"""Tests for the GHCNh station quality-control checks."""

from unittest import mock

import pandas as pd
import polars as pl
import pytest

from extremeweatherbench import cases, evaluate, inputs, regions, station_qc


def _station_frame(temps, station="S1", start="2020-11-04", freq="1h", **columns):
    """Build an hourly single-station GHCNh-like frame."""
    n = len(temps)
    return pl.DataFrame(
        {
            "station": [station] * n,
            "valid_time": pd.date_range(start, periods=n, freq=freq),
            "latitude": [47.0] * n,
            "longitude": [19.0] * n,
            "surface_air_temperature": temps,
            **columns,
        }
    )


class TestMaskSentinelValues:
    def test_nulls_sentinels(self):
        df = _station_frame([10.0, -999.0, 11.0, -9999.0, 9999.0, -999.9])
        out = station_qc.mask_sentinel_values(
            df.lazy(), ["surface_air_temperature"]
        ).collect()
        assert out["surface_air_temperature"].to_list() == [
            10.0,
            None,
            11.0,
            None,
            None,
            None,
        ]

    def test_keeps_valid_pressure_of_999_9_hpa(self):
        df = _station_frame(
            [10.0, 10.0], air_pressure_at_mean_sea_level=[999.9, -999.0]
        )
        out = station_qc.apply_ghcnh_quality_control(df.lazy()).collect()
        assert out["air_pressure_at_mean_sea_level"].to_list() == [999.9, None]

    def test_ignores_missing_columns(self):
        df = _station_frame([10.0])
        out = station_qc.mask_sentinel_values(df.lazy(), ["not_a_column"]).collect()
        assert out.equals(df)


class TestMaskOutOfRange:
    def test_nulls_impossible_values(self):
        df = _station_frame(
            [20.0, 500.0, -95.0, -89.0],
            surface_relative_humidity=[50.0, 13871565.0, 101.0, -1.0],
            surface_wind_from_direction=[180.0, 999.0, 360.0, 0.0],
        )
        out = station_qc.mask_out_of_range(df.lazy()).collect()
        assert out["surface_air_temperature"].to_list() == [20.0, None, None, -89.0]
        assert out["surface_relative_humidity"].to_list() == [50.0, None, 101.0, None]
        assert out["surface_wind_from_direction"].to_list() == [180.0, None, 360.0, 0.0]

    def test_keeps_nulls(self):
        df = _station_frame([None, 20.0])
        out = station_qc.mask_out_of_range(df.lazy()).collect()
        assert out["surface_air_temperature"].to_list() == [None, 20.0]


class TestMaskSpikes:
    def test_nulls_isolated_spike(self):
        df = _station_frame([10.0, 11.0, 35.0, 12.0, 12.5])
        out = station_qc.mask_spikes(df.lazy()).collect()
        assert out["surface_air_temperature"].to_list() == [
            10.0,
            11.0,
            None,
            12.0,
            12.5,
        ]

    def test_nulls_negative_spike(self):
        df = _station_frame([10.0, 11.0, -15.0, 12.0])
        out = station_qc.mask_spikes(df.lazy()).collect()
        assert out["surface_air_temperature"].to_list()[2] is None

    def test_keeps_sustained_step(self):
        # a front: large drop that persists, not an up-then-down spike
        temps = [20.0, 20.5, 8.0, 7.5, 7.0]
        out = station_qc.mask_spikes(_station_frame(temps).lazy()).collect()
        assert out["surface_air_temperature"].to_list() == temps

    def test_keeps_spike_with_distant_neighbors(self):
        temps = [10.0, 35.0, 12.0]
        df = _station_frame(temps, freq="6h")
        out = station_qc.mask_spikes(df.lazy()).collect()
        assert out["surface_air_temperature"].to_list() == temps

    def test_spike_check_skips_over_nulls(self):
        df = _station_frame([10.0, None, 35.0, None, 12.0])
        out = station_qc.mask_spikes(df.lazy()).collect()
        assert out["surface_air_temperature"].to_list() == [
            10.0,
            None,
            None,
            None,
            12.0,
        ]

    def test_neighbors_are_per_station(self):
        a = _station_frame([10.0, 11.0], station="A")
        b = _station_frame([35.0, 36.0], station="B")
        df = pl.concat([a, b])
        out = station_qc.mask_spikes(df.lazy()).collect()
        assert out["surface_air_temperature"].to_list() == [10.0, 11.0, 35.0, 36.0]

    def test_unsorted_input(self):
        df = _station_frame([10.0, 11.0, 35.0, 12.0, 12.5]).reverse()
        out = station_qc.mask_spikes(df.lazy()).collect()
        assert out["surface_air_temperature"].null_count() == 1
        assert out.filter(pl.col("surface_air_temperature").is_null())[
            "valid_time"
        ].item() == pd.Timestamp("2020-11-04 02:00")

    def test_missing_station_column_is_noop(self):
        df = _station_frame([10.0, 35.0, 12.0]).drop("station")
        out = station_qc.mask_spikes(df.lazy()).collect()
        assert out.equals(df)


class TestApplyGhcnhRowChecks:
    def test_row_checks_skip_spikes(self):
        df = _station_frame(
            [10.0, -999.0, 11.0, 40.0, 12.0], name=["A (MOVING ICE STATION)"] * 5
        )
        assert station_qc.apply_ghcnh_row_checks(df.lazy()).collect().height == 0
        out = station_qc.apply_ghcnh_row_checks(df.drop("name").lazy()).collect()
        assert out["surface_air_temperature"].to_list() == [
            10.0,
            None,
            11.0,
            40.0,
            12.0,
        ]


class TestApplyGhcnhQualityControl:
    def test_all_checks(self):
        df = _station_frame(
            [10.0, -999.0, 11.0, 928.0, 11.5, 40.0, 12.0],
            surface_dew_point=[5.0, 5.0, 550.0, 5.0, 5.0, 5.0, 5.0],
        )
        out = station_qc.apply_ghcnh_quality_control(df.lazy()).collect()
        assert out["surface_air_temperature"].to_list() == [
            10.0,
            None,
            11.0,
            None,
            11.5,
            None,
            12.0,
        ]
        assert out["surface_dew_point"].to_list() == [5.0, 5.0, None] + [5.0] * 4
        assert out.height == df.height


class TestGHCNQualityControl:
    @pytest.fixture
    def case(self):
        mock_case = mock.Mock()
        mock_case.start_date = pd.Timestamp("2020-11-05")
        mock_case.end_date = pd.Timestamp("2020-11-06")
        mock_case.location.as_geopandas.return_value.total_bounds = [10, 40, 30, 55]
        return mock_case

    @pytest.fixture
    def frame(self):
        good = _station_frame([10.0, 11.0, 12.0, 13.0], station="GOOD")
        bad = _station_frame([-999.0] * 4, station="HUU00061700")
        return pl.concat([good, bad]).lazy()

    def _open_and_subset(self, ghcn, frame, case):
        with mock.patch("polars.scan_parquet", return_value=frame):
            data = ghcn._open_data_from_source()
        data = inputs.maybe_subset_variables(
            data,
            ghcn.variables,
            keep_if_present=ghcn.columns_kept_through_variable_subset(),
        )
        return ghcn.subset_data_to_case(data, case).collect()

    def test_quality_control_on_by_default(self, case, frame):
        ghcn = inputs.GHCN(variables=["surface_air_temperature"])
        assert ghcn.quality_control
        out = self._open_and_subset(ghcn, frame, case)
        assert out["valid_time"].is_sorted()
        assert out.height == 8
        bad = out.filter(pl.col("station") == "HUU00061700")
        assert bad["surface_air_temperature"].null_count() == 4
        good = out.filter(pl.col("station") == "GOOD")
        assert good["surface_air_temperature"].to_list() == [10.0, 11.0, 12.0, 13.0]

    def test_quality_control_can_be_disabled(self, case, frame):
        ghcn = inputs.GHCN(variables=["surface_air_temperature"], quality_control=False)
        out = self._open_and_subset(ghcn, frame, case)
        assert (out["surface_air_temperature"] == -999.0).sum() == 4
        assert "station" not in out.columns

    def test_station_kept_through_variable_subset(self, frame):
        ghcn = inputs.GHCN(variables=["surface_air_temperature"])
        out = inputs.maybe_subset_variables(
            frame,
            ghcn.variables,
            keep_if_present=ghcn.columns_kept_through_variable_subset(),
        )
        assert "station" in out.collect_schema().names()

    def test_metadata_columns_dropped_before_conversion(self, frame):
        ghcn = inputs.GHCN(variables=["surface_air_temperature"])
        named = frame.with_columns(name=pl.lit("GENEVA"))
        ds = ghcn._custom_convert_to_dataset(named)
        assert list(ds.data_vars) == ["surface_air_temperature"]


class TestGHCNQualityControlPipeline:
    """Run QC through ``evaluate.run_pipeline``, as the default GHCN targets do."""

    @pytest.fixture
    def source(self, tmp_path):
        times = pd.date_range("2020-11-03", periods=96, freq="1h")
        good = [10.0] * 96
        good[50] = 40.0  # isolated spike at 2020-11-05 02:00

        def station(station_id, name, lat, lon, temps):
            n = len(temps)
            return pl.DataFrame(
                {
                    "station": [station_id] * n,
                    "name": [name] * n,
                    "latitude": [lat] * n,
                    "longitude": [lon] * n,
                    "surface_air_temperature": temps,
                    "valid_time": times[:n],
                }
            )

        frame = pl.concat(
            [
                station("GOOD", "GENEVA", 47.0, 19.0, good),
                station(
                    "GLU00000037", "NEM (MOVING ICE STATION)", 48.0, 20.0, [4.0] * 96
                ),
            ]
        )
        path = tmp_path / "ghcnh.parquet"
        frame.write_parquet(path)
        return str(path)

    @pytest.fixture
    def case(self):
        return cases.IndividualCase(
            case_id_number=1,
            title="test",
            start_date=pd.Timestamp("2020-11-04"),
            end_date=pd.Timestamp("2020-11-05"),
            location=regions.BoundingBoxRegion(40, 55, 10, 30),
            event_type="heat_wave",
        )

    def test_excludes_ice_station_and_masks_spike(self, source, case):
        ghcn = inputs.GHCN(
            source=source, variables=["surface_air_temperature"], storage_options={}
        )
        ds = evaluate.run_pipeline(case, ghcn)
        assert list(ds.data_vars) == ["surface_air_temperature"]
        assert ds.sizes["location"] == 1
        assert float(ds["latitude"].item()) == 47.0
        temps = ds["surface_air_temperature"].isel(location=0).dropna("valid_time")
        # the spike's row has no values left, so it is dropped before conversion
        assert pd.Timestamp("2020-11-05 02:00") not in temps["valid_time"].values
        assert temps.sizes["valid_time"] == 95
        assert float(temps.max()) == pytest.approx(283.15)

    def test_disabled_keeps_ice_station_and_spike(self, source, case):
        ghcn = inputs.GHCN(
            source=source,
            variables=["surface_air_temperature"],
            storage_options={},
            quality_control=False,
        )
        ds = evaluate.run_pipeline(case, ghcn)
        assert ds.sizes["location"] == 2
        assert float(ds["surface_air_temperature"].max()) == pytest.approx(313.15)


class TestExcludeStations:
    def test_drops_moving_ice_stations_by_default(self):
        good = _station_frame([10.0, 11.0], station="GOOD", name=["GENEVA"] * 2)
        ice = _station_frame(
            [4.0, 5.0], station="GLU00000037", name=["NEM (MOVING ICE STATION)"] * 2
        )
        out = station_qc.exclude_stations(pl.concat([good, ice]).lazy()).collect()
        assert out["station"].unique().to_list() == ["GOOD"]

    def test_applied_by_apply_ghcnh_quality_control(self):
        ice = _station_frame(
            [4.0, 5.0], station="GLU00000037", name=["NEM (MOVING ICE STATION)"] * 2
        )
        out = station_qc.apply_ghcnh_quality_control(ice.lazy()).collect()
        assert out.height == 0

    def test_custom_patterns_and_null_names(self):
        df = _station_frame([1.0, 2.0, 3.0], name=["A STATION", None, "B STATION"])
        out = station_qc.exclude_stations(df.lazy(), ["A STA"]).collect()
        assert out["name"].to_list() == [None, "B STATION"]

    def test_noop_without_patterns_or_name_column(self):
        df = _station_frame([1.0], name=["NEM (MOVING ICE STATION)"])
        assert station_qc.exclude_stations(df.lazy(), []).collect().equals(df)
        no_name = df.drop("name")
        assert station_qc.exclude_stations(no_name.lazy()).collect().equals(no_name)


class TestSkippedCheckWarnings:
    def test_warns_when_name_column_missing(self, caplog):
        df = _station_frame([1.0])
        with caplog.at_level("WARNING", logger="extremeweatherbench.station_qc"):
            station_qc.exclude_stations(df.lazy())
        assert "Station exclusion skipped" in caplog.text

    def test_warns_when_station_column_missing(self, caplog):
        df = _station_frame([1.0]).drop("station")
        with caplog.at_level("WARNING", logger="extremeweatherbench.station_qc"):
            station_qc.mask_spikes(df.lazy())
        assert "Spike check skipped" in caplog.text
        assert "station" in caplog.text

    def test_no_warning_without_checked_variables(self, caplog):
        df = pl.DataFrame({"valid_time": [pd.Timestamp("2020-01-01")], "x": [1.0]})
        with caplog.at_level("WARNING", logger="extremeweatherbench.station_qc"):
            station_qc.mask_spikes(df.lazy())
            station_qc.exclude_stations(df.lazy(), [])
        assert caplog.text == ""


class TestGHCNDuplicatesAfterQualityControl:
    def test_valid_duplicate_wins_over_nulled_row(self):
        # two colocated stations report at the same time; the first one's
        # sentinel is nulled by QC and must not hide the second one's value
        t = pd.Timestamp("2020-11-04")
        frame = pl.DataFrame(
            {
                "station": ["A", "B"],
                "valid_time": [t, t],
                "latitude": [47.0, 47.0],
                "longitude": [19.0, 19.0],
                "surface_air_temperature": [-999.0, 12.0],
            }
        ).lazy()
        ghcn = inputs.GHCN(variables=["surface_air_temperature"])
        checked = station_qc.apply_ghcnh_row_checks(frame)
        ds = ghcn._custom_convert_to_dataset(checked)
        assert ds.sizes["location"] == 1
        assert float(ds["surface_air_temperature"].item()) == pytest.approx(285.15)


class TestGHCNEmptyCase:
    def test_warns_when_quality_control_empties_case(self, caplog):
        ice = _station_frame(
            [4.0, 5.0], station="GLU00000037", name=["NEM (MOVING ICE STATION)"] * 2
        )
        ghcn = inputs.GHCN(variables=["surface_air_temperature"])
        checked = station_qc.apply_ghcnh_row_checks(ice.lazy())
        with caplog.at_level("WARNING", logger="extremeweatherbench.inputs"):
            ds = ghcn._custom_convert_to_dataset(checked)
        assert "No GHCN station observations" in caplog.text
        assert "after quality control" in caplog.text
        assert 0 in ds.sizes.values()
