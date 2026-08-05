"""Self-tests for the optimization harness.

A harness that cannot fail is worse than no harness, so each helper is
checked against both a case it must accept and a case it must reject.
"""

import dask.array as dask_array
import numpy as np
import pytest
import xarray as xr

from tests import optimization_helpers as opt


class TestAssertLazy:
    """assert_lazy must accept dask-backed data and reject computed data."""

    def test_accepts_dask_backed_dataarray(self):
        da = xr.DataArray(dask_array.zeros((4, 4), chunks=2), dims=["x", "y"])
        opt.assert_lazy(da)

    def test_rejects_numpy_backed_dataarray(self):
        da = xr.DataArray(np.zeros((4, 4)), dims=["x", "y"])
        with pytest.raises(AssertionError, match="expected a lazy"):
            opt.assert_lazy(da)

    def test_rejects_computed_result_of_a_dask_graph(self):
        da = xr.DataArray(dask_array.zeros((4, 4), chunks=2), dims=["x", "y"])
        with pytest.raises(AssertionError, match="expected a lazy"):
            opt.assert_lazy(da.compute())

    def test_reports_the_backing_type_in_the_message(self):
        da = xr.DataArray(np.zeros((2, 2)), dims=["x", "y"])
        with pytest.raises(AssertionError, match="ndarray"):
            opt.assert_lazy(da)


class TestGraphSize:
    """graph_size must grow with the number of chunks in the graph."""

    def test_more_chunks_produce_a_larger_graph(self):
        base = np.zeros((64,))
        few = xr.DataArray(dask_array.from_array(base, chunks=32), dims=["x"])
        many = xr.DataArray(dask_array.from_array(base, chunks=2), dims=["x"])
        assert opt.graph_size(many) > opt.graph_size(few)

    def test_rejects_non_dask_input(self):
        da = xr.DataArray(np.zeros(4), dims=["x"])
        with pytest.raises(AssertionError, match="requires a dask collection"):
            opt.graph_size(da)


class TestCountTaskExecutions:
    """Task counting must distinguish one compute from a repeated compute."""

    def test_counts_more_tasks_when_a_graph_is_computed_twice(self):
        da = xr.DataArray(dask_array.ones((16,), chunks=4), dims=["x"])

        with opt.count_task_executions() as once:
            (da + 1).compute()

        with opt.count_task_executions() as twice:
            doubled = da + 1
            doubled.compute()
            doubled.compute()

        assert twice.count > once.count

    def test_persisted_intermediate_is_not_recomputed(self):
        """A shared intermediate computed once costs fewer tasks than one
        rebuilt by each consumer. This is the contract the AR fix relies on."""
        da = xr.DataArray(dask_array.ones((16,), chunks=4), dims=["x"])

        with opt.count_task_executions() as recomputed:
            shared = (da + 1) * 2
            xr.Dataset({"a": shared.sum(), "b": shared.mean()}).compute()

        with opt.count_task_executions() as reused:
            shared = ((da + 1) * 2).persist()
            xr.Dataset({"a": shared.sum(), "b": shared.mean()}).compute()

        assert reused.count < recomputed.count


class TestSpy:
    """The spy must count calls while leaving real behavior intact."""

    def test_counts_calls_without_changing_the_result(self):
        import extremeweatherbench.utils as utils

        with opt.spy(utils, "convert_longitude_to_360") as counter:
            assert utils.convert_longitude_to_360(-10.0) == pytest.approx(350.0)
            assert utils.convert_longitude_to_360(370.0) == pytest.approx(10.0)

        assert counter.count == 2

    def test_records_zero_calls_when_the_function_is_not_used(self):
        import extremeweatherbench.utils as utils

        with opt.spy(utils, "convert_longitude_to_360") as counter:
            pass

        assert counter.count == 0

    def test_restores_the_original_function_on_exit(self):
        import extremeweatherbench.utils as utils

        original = utils.convert_longitude_to_360
        with opt.spy(utils, "convert_longitude_to_360"):
            pass
        assert utils.convert_longitude_to_360 is original
