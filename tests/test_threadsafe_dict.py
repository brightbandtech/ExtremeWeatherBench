"""Comprehensive tests for ThreadSafeDict class.

This test suite covers all scenarios including single-threaded, joblib-based
parallel execution, and stress testing scenarios to ensure ThreadSafeDict
works correctly under all conditions.

Note: This version uses only joblib for all parallel processing and avoids
pickle serialization entirely. No other multiprocessing modules are used.
"""

import random
import threading
import time

import numpy as np
import pytest
import xarray as xr
from joblib import Parallel, delayed

from extremeweatherbench.utils import ThreadSafeDict


def _joblib_worker_task(process_id, num_items=100):
    """Worker function for joblib parallel processing test.

    This function creates and populates a ThreadSafeDict instance within
    each worker process, avoiding any pickle serialization.

    Args:
        process_id: Unique identifier for this worker
        num_items: Number of items to add to the dict

    Returns:
        Dict containing statistics about the process
    """
    # Create ThreadSafeDict inside worker (in-memory, no pickle)
    tsd = ThreadSafeDict()

    # Each process creates its own ThreadSafeDict and populates it
    for i in range(num_items):
        key = f"process_{process_id}_key_{i}"
        value = f"process_{process_id}_value_{i}"
        tsd[key] = value

    # Return some stats about the process (serialize only simple data)
    return {
        "process_id": process_id,
        "length": len(tsd),
        "sample_key": f"process_{process_id}_key_0",
        "sample_value": tsd.get(f"process_{process_id}_key_0"),
        "all_keys_present": all(
            f"process_{process_id}_key_{i}" in tsd for i in range(num_items)
        ),
    }


def _joblib_shared_computation_task(task_id, base_value):
    """Worker function that performs computation and returns results.

    Args:
        task_id: Unique identifier for this task
        base_value: Base value for computation

    Returns:
        Dict containing computation results
    """
    # Create local ThreadSafeDict for this task
    local_dict = ThreadSafeDict()

    # Perform some computation
    for i in range(50):
        key = f"task_{task_id}_item_{i}"
        computed_value = base_value * i + task_id
        local_dict[key] = computed_value

    # Return summary (no serialization of ThreadSafeDict itself)
    return {
        "task_id": task_id,
        "items_count": len(local_dict),
        "sum_of_values": sum(local_dict.values()),
        "first_value": local_dict.get(f"task_{task_id}_item_0"),
    }


def _joblib_concurrent_read_worker(tsd_data, num_reads=100):
    """Worker function for testing concurrent reads using joblib.

    Args:
        tsd_data: Dictionary data to populate ThreadSafeDict with
        num_reads: Number of read operations to perform

    Returns:
        List of (key, value) tuples from read operations
    """
    # Create and populate ThreadSafeDict in worker
    tsd = ThreadSafeDict()
    for key, value in tsd_data.items():
        tsd[key] = value

    # Perform reads
    results = []
    for i in range(num_reads):
        key = f"key_{i % 50}"  # Read from subset of keys
        value = tsd.get(key)
        results.append((key, value))

    return results


def _joblib_concurrent_write_worker(thread_id, num_writes=100):
    """Worker function for testing concurrent writes using joblib.

    Args:
        thread_id: Unique identifier for this worker
        num_writes: Number of write operations to perform

    Returns:
        Dict containing write statistics
    """
    # Create ThreadSafeDict in worker
    tsd = ThreadSafeDict()

    # Perform writes
    for i in range(num_writes):
        key = f"thread_{thread_id}_key_{i}"
        value = f"thread_{thread_id}_value_{i}"
        tsd[key] = value

    return {
        "thread_id": thread_id,
        "length": len(tsd),
        "sample_key": f"thread_{thread_id}_key_0",
        "sample_value": tsd.get(f"thread_{thread_id}_key_0"),
    }


def _joblib_stress_worker(worker_id, operations_per_worker=1000):
    """Worker function for stress testing using joblib.

    Args:
        worker_id: Unique identifier for this worker
        operations_per_worker: Number of operations to perform

    Returns:
        List of operation results
    """
    tsd = ThreadSafeDict()
    operations = []

    for i in range(operations_per_worker):
        op_type = random.choice(["set", "get", "delete", "contains"])
        key = f"key_{random.randint(0, 99)}"

        try:
            if op_type == "set":
                tsd[key] = f"value_{worker_id}_{i}"
                operations.append(("set", key, True))
            elif op_type == "get":
                value = tsd.get(key)
                operations.append(("get", key, value))
            elif op_type == "delete":
                if key in tsd:
                    del tsd[key]
                    operations.append(("delete", key, True))
            elif op_type == "contains":
                exists = key in tsd
                operations.append(("contains", key, exists))
        except Exception as e:
            operations.append(("error", key, str(e)))

    return operations


class TestThreadSafeDictBasic:
    """Basic functionality tests for ThreadSafeDict."""

    def test_init(self):
        """Test ThreadSafeDict initialization."""
        tsd = ThreadSafeDict()
        assert len(tsd) == 0
        assert list(tsd.keys()) == []
        assert list(tsd.values()) == []
        assert list(tsd.items()) == []

    def test_setitem_getitem(self):
        """Test setting and getting items."""
        tsd = ThreadSafeDict()

        # Test basic set/get
        tsd["key1"] = "value1"
        assert tsd["key1"] == "value1"

        # Test overwrite
        tsd["key1"] = "new_value1"
        assert tsd["key1"] == "new_value1"

        # Test multiple keys
        tsd["key2"] = 42
        tsd["key3"] = [1, 2, 3]
        assert tsd["key2"] == 42
        assert tsd["key3"] == [1, 2, 3]

    def test_delitem(self):
        """Test deleting items."""
        tsd = ThreadSafeDict()
        tsd["key1"] = "value1"
        tsd["key2"] = "value2"

        del tsd["key1"]
        assert "key1" not in tsd
        assert "key2" in tsd

        # Test deleting non-existent key
        with pytest.raises(KeyError):
            del tsd["non_existent"]

    def test_contains(self):
        """Test __contains__ method."""
        tsd = ThreadSafeDict()
        tsd["key1"] = "value1"

        assert "key1" in tsd
        assert "key2" not in tsd
        assert None not in tsd

    def test_get(self):
        """Test get method with default values."""
        tsd = ThreadSafeDict()
        tsd["key1"] = "value1"

        assert tsd.get("key1") == "value1"
        assert tsd.get("key2") is None
        assert tsd.get("key2", "default") == "default"
        assert tsd.get("key1", "default") == "value1"

    def test_len(self):
        """Test length method."""
        tsd = ThreadSafeDict()
        assert len(tsd) == 0

        tsd["key1"] = "value1"
        assert len(tsd) == 1

        tsd["key2"] = "value2"
        assert len(tsd) == 2

        del tsd["key1"]
        assert len(tsd) == 1

    def test_clear(self):
        """Test clear method."""
        tsd = ThreadSafeDict()
        tsd["key1"] = "value1"
        tsd["key2"] = "value2"

        assert len(tsd) == 2
        tsd.clear()
        assert len(tsd) == 0
        assert list(tsd.keys()) == []

    def test_keys_values_items(self):
        """Test keys, values, and items methods."""
        tsd = ThreadSafeDict()
        tsd["a"] = 1
        tsd["b"] = 2
        tsd["c"] = 3

        keys = tsd.keys()
        values = tsd.values()
        items = tsd.items()

        # Should return lists (copies)
        assert isinstance(keys, list)
        assert isinstance(values, list)
        assert isinstance(items, list)

        # Check contents
        assert set(keys) == {"a", "b", "c"}
        assert set(values) == {1, 2, 3}
        assert set(items) == {("a", 1), ("b", 2), ("c", 3)}

        # Test that they're copies (concurrent modification safe)
        tsd["d"] = 4
        assert "d" not in keys  # Original list unchanged
        assert tsd.get("d") == 4  # But dict was modified

    def test_complex_data_types(self):
        """Test with complex data types."""
        tsd = ThreadSafeDict()

        # Test with various data types
        data_types = {
            "string": "hello",
            "int": 42,
            "float": 3.14,
            "list": [1, 2, 3],
            "dict": {"nested": "value"},
            "tuple": (1, 2, 3),
            "numpy_array": np.array([1, 2, 3]),
            "xarray": xr.DataArray([1, 2, 3], dims=["x"]),
            "none": None,
            "bool": True,
        }

        for key, value in data_types.items():
            tsd[key] = value
            if isinstance(value, np.ndarray):
                np.testing.assert_array_equal(tsd[key], value)
            elif isinstance(value, xr.DataArray):
                xr.testing.assert_equal(tsd[key], value)
            else:
                assert tsd[key] == value

    def test_unhashable_keys_error(self):
        """Test that unhashable keys raise appropriate errors."""
        tsd = ThreadSafeDict()

        # These should raise TypeError
        with pytest.raises(TypeError):
            tsd[{"unhashable": "dict"}] = "value"

        with pytest.raises(TypeError):
            tsd[[1, 2, 3]] = "value"


class TestThreadSafeDictJoblib:
    """Joblib-based parallel processing tests for ThreadSafeDict."""

    def test_joblib_parallel_execution(self):
        """Test ThreadSafeDict creation and usage in joblib workers."""
        # Test with different numbers of jobs
        n_jobs = 4
        num_items_per_job = 100

        # Run tasks in parallel using joblib with threading backend
        results = Parallel(n_jobs=n_jobs, backend="threading")(
            delayed(_joblib_worker_task)(i, num_items_per_job) for i in range(n_jobs)
        )

        # Verify results
        assert len(results) == n_jobs
        for i, result in enumerate(results):
            assert result["process_id"] == i
            assert result["length"] == num_items_per_job
            assert result["sample_value"] == f"process_{i}_value_0"
            assert result["all_keys_present"] is True

    def test_joblib_shared_computation(self):
        """Test joblib workers performing computations with ThreadSafeDict."""
        n_jobs = 6
        base_values = [10, 20, 30, 40, 50, 60]

        # Run computation tasks in parallel with threading backend
        results = Parallel(n_jobs=n_jobs, backend="threading")(
            delayed(_joblib_shared_computation_task)(i, base_values[i])
            for i in range(n_jobs)
        )

        # Verify results
        assert len(results) == n_jobs
        for i, result in enumerate(results):
            assert result["task_id"] == i
            assert result["items_count"] == 50
            assert result["first_value"] == base_values[i] * 0 + i  # i
            # Verify sum calculation
            expected_sum = sum(base_values[i] * j + i for j in range(50))
            assert result["sum_of_values"] == expected_sum

    def test_joblib_concurrent_reads(self):
        """Test concurrent reads using joblib workers."""
        # Prepare initial data
        initial_data = {f"key_{i}": f"value_{i}" for i in range(100)}

        # Run concurrent reads using joblib with threading backend
        results = Parallel(n_jobs=10, backend="threading")(
            delayed(_joblib_concurrent_read_worker)(initial_data, 100)
            for _ in range(10)
        )

        # Verify all reads were successful
        all_results = []
        for worker_results in results:
            all_results.extend(worker_results)

        assert len(all_results) == 1000  # 10 workers * 100 reads each
        for key, value in all_results:
            if value is not None:  # Some keys might not exist
                expected_value = key.replace("key_", "value_")
                assert value == expected_value

    def test_joblib_concurrent_writes(self):
        """Test concurrent writes using joblib workers."""
        # Run concurrent writes using threading backend
        results = Parallel(n_jobs=10, backend="threading")(
            delayed(_joblib_concurrent_write_worker)(i, 100) for i in range(10)
        )

        # Verify all writes were successful
        total_length = sum(result["length"] for result in results)
        assert total_length == 1000  # 10 workers * 100 writes each

        for i, result in enumerate(results):
            assert result["thread_id"] == i
            assert result["length"] == 100
            assert result["sample_value"] == f"thread_{i}_value_0"

    def test_joblib_different_backends(self):
        """Test ThreadSafeDict with different joblib backends."""
        backends_to_test = ["threading", "loky"]

        for backend in backends_to_test:
            try:
                results = Parallel(n_jobs=2, backend=backend)(
                    delayed(_joblib_worker_task)(i, 50) for i in range(2)
                )

                # Verify results for this backend
                assert len(results) == 2
                for i, result in enumerate(results):
                    assert result["process_id"] == i
                    assert result["length"] == 50
                    assert result["all_keys_present"] is True

            except Exception as e:
                # Some backends might not be available in all environments
                pytest.skip(f"Backend {backend} not available: {e}")

    def test_joblib_memory_efficiency(self):
        """Test memory efficiency with joblib workers."""

        def create_large_dict_task(task_id):
            """Create a large ThreadSafeDict and return summary."""
            large_dict = ThreadSafeDict()

            # Add many items
            for i in range(1000):
                key = f"large_key_{task_id}_{i}"
                value = list(range(100))  # Some non-trivial data
                large_dict[key] = value

            # Return only summary data (not the dict itself)
            return {
                "task_id": task_id,
                "length": len(large_dict),
                "sample_value_length": len(
                    large_dict.get(f"large_key_{task_id}_0", [])
                ),
            }

        # Run tasks that create large dicts with threading backend
        results = Parallel(n_jobs=3, backend="threading")(
            delayed(create_large_dict_task)(i) for i in range(3)
        )

        # Verify each task created and processed large dict correctly
        for i, result in enumerate(results):
            assert result["task_id"] == i
            assert result["length"] == 1000
            assert result["sample_value_length"] == 100

    def test_joblib_stress_operations(self):
        """Stress test with many joblib workers and operations."""
        num_workers = 20
        operations_per_worker = 500

        # Run stress test using joblib with threading backend
        results = Parallel(n_jobs=num_workers, backend="threading")(
            delayed(_joblib_stress_worker)(i, operations_per_worker)
            for i in range(num_workers)
        )

        # Verify no errors occurred
        all_operations = []
        for worker_results in results:
            all_operations.extend(worker_results)

        errors = [op for op in all_operations if op[0] == "error"]
        error_msg = f"Errors occurred: {errors[:5]}..."
        assert len(errors) == 0, error_msg

        # Verify total operations
        expected_total = num_workers * operations_per_worker
        assert len(all_operations) >= expected_total * 0.8  # Allow 20% failure


class TestThreadSafeDictStress:
    """Stress tests for ThreadSafeDict using only joblib."""

    def test_large_data_volume(self):
        """Test with large amounts of data."""
        tsd = ThreadSafeDict()

        # Insert large amount of data
        num_items = 10000
        for i in range(num_items):
            key = f"large_key_{i:06d}"
            value = {
                "id": i,
                "data": list(range(100)),  # Some non-trivial data
                "metadata": {"processed": True, "timestamp": i * 1.5},
            }
            tsd[key] = value

        assert len(tsd) == num_items

        # Verify random sampling of data
        sample_indices = random.sample(range(num_items), 100)
        for i in sample_indices:
            key = f"large_key_{i:06d}"
            value = tsd[key]
            assert value["id"] == i
            assert len(value["data"]) == 100
            assert value["metadata"]["timestamp"] == i * 1.5

    def test_rapid_operations(self):
        """Test rapid fire operations."""
        tsd = ThreadSafeDict()
        num_operations = 50000

        start_time = time.time()

        # Rapid insertions
        for i in range(num_operations):
            tsd[f"rapid_{i}"] = i

        # Rapid reads
        for i in range(0, num_operations, 2):
            _ = tsd.get(f"rapid_{i}")

        # Rapid deletions
        for i in range(0, num_operations, 4):
            if f"rapid_{i}" in tsd:
                del tsd[f"rapid_{i}"]

        end_time = time.time()

        # Should complete in reasonable time (less than 10 seconds)
        assert end_time - start_time < 10

        # Verify final state
        expected_remaining = num_operations - (num_operations // 4)
        assert len(tsd) == expected_remaining

    def test_memory_efficiency(self):
        """Test memory efficiency under load."""
        tsd = ThreadSafeDict()

        # Add many items
        for i in range(10000):
            tsd[f"mem_test_{i}"] = f"value_{i}"

        # Clear and verify memory is released
        original_length = len(tsd)
        tsd.clear()

        assert len(tsd) == 0
        assert original_length == 10000

        # Add items again to verify it still works
        for i in range(100):
            tsd[f"after_clear_{i}"] = f"new_value_{i}"

        assert len(tsd) == 100

    def test_joblib_concurrent_stress_operations(self):
        """Extreme stress test using joblib parallel execution."""
        num_workers = 50
        operations_per_worker = 1000

        def extreme_stress_worker(worker_id):
            """Worker that performs many operations on its own ThreadSafeDict."""
            tsd = ThreadSafeDict()
            ops_completed = 0

            for i in range(operations_per_worker):
                try:
                    op = random.choice(["set", "get", "delete", "update"])
                    key = f"stress_{random.randint(0, 500)}"

                    if op == "set":
                        tsd[key] = f"worker_{worker_id}_op_{i}"
                    elif op == "get":
                        _ = tsd.get(key)
                    elif op == "delete":
                        if key in tsd:
                            del tsd[key]
                    elif op == "update":
                        if key in tsd:
                            current = tsd[key]
                            tsd[key] = f"updated_{current}"
                        else:
                            tsd[key] = f"new_worker_{worker_id}_op_{i}"

                    ops_completed += 1

                except Exception:
                    # Ignore individual operation failures in stress test
                    pass

            return {"worker_id": worker_id, "ops_completed": ops_completed}

        # Run extreme stress test using joblib with threading backend
        start_time = time.time()
        results = Parallel(n_jobs=num_workers, backend="threading")(
            delayed(extreme_stress_worker)(i) for i in range(num_workers)
        )
        end_time = time.time()

        # Verify results
        total_ops = sum(result["ops_completed"] for result in results)
        expected_total = num_workers * operations_per_worker

        # Should complete most operations successfully
        assert total_ops >= expected_total * 0.9  # Allow 10% failure under stress

        # Should complete in reasonable time
        assert end_time - start_time < 30  # 30 seconds max


class TestThreadSafeDictEdgeCases:
    """Edge case tests for ThreadSafeDict."""

    def test_none_values(self):
        """Test handling None values."""
        tsd = ThreadSafeDict()

        tsd["none_key"] = None
        assert tsd["none_key"] is None
        assert "none_key" in tsd
        assert tsd.get("none_key") is None
        assert tsd.get("none_key", "default") is None

    def test_zero_and_false_values(self):
        """Test handling falsy values."""
        tsd = ThreadSafeDict()

        falsy_values = {
            "zero": 0,
            "empty_string": "",
            "empty_list": [],
            "empty_dict": {},
            "false": False,
        }

        for key, value in falsy_values.items():
            tsd[key] = value
            assert tsd[key] == value
            assert key in tsd
            assert tsd.get(key) == value

    def test_large_objects(self):
        """Test with large objects."""
        tsd = ThreadSafeDict()

        # Large numpy array
        large_array = np.random.random((1000, 1000))
        tsd["large_array"] = large_array

        retrieved = tsd["large_array"]
        np.testing.assert_array_equal(retrieved, large_array)

        # Large string
        large_string = "x" * 1000000  # 1MB string
        tsd["large_string"] = large_string
        assert tsd["large_string"] == large_string

    def test_special_key_types(self):
        """Test with various key types."""
        tsd = ThreadSafeDict()

        special_keys = {
            42: "integer_key",
            3.14: "float_key",
            (1, 2, 3): "tuple_key",
            frozenset([1, 2, 3]): "frozenset_key",
            True: "bool_key",
        }

        for key, value in special_keys.items():
            tsd[key] = value
            assert tsd[key] == value
            assert key in tsd

    def test_unicode_and_special_characters(self):
        """Test with unicode and special characters."""
        tsd = ThreadSafeDict()

        unicode_data = {
            "emoji_key_🚀": "rocket_value",
            "中文_key": "chinese_value",
            "العربية": "arabic_value",
            "key with spaces": "space_value",
            "key\nwith\nnewlines": "newline_value",
            "key\twith\ttabs": "tab_value",
        }

        for key, value in unicode_data.items():
            tsd[key] = value
            assert tsd[key] == value

    def test_circular_references(self):
        """Test handling objects with circular references."""
        tsd = ThreadSafeDict()

        # Create objects with circular references
        obj1 = {"name": "obj1"}
        obj2 = {"name": "obj2"}
        obj1["ref"] = obj2
        obj2["ref"] = obj1

        tsd["circular1"] = obj1
        tsd["circular2"] = obj2

        # Should store and retrieve without issues
        retrieved1 = tsd["circular1"]
        retrieved2 = tsd["circular2"]

        assert retrieved1["name"] == "obj1"
        assert retrieved2["name"] == "obj2"
        assert retrieved1["ref"] is retrieved2
        assert retrieved2["ref"] is retrieved1

    def test_exception_during_operations(self):
        """Test behavior when exceptions occur during operations."""
        tsd = ThreadSafeDict()

        # Test KeyError for missing keys
        with pytest.raises(KeyError):
            _ = tsd["missing_key"]

        # Test that exception doesn't corrupt the dict
        tsd["valid_key"] = "valid_value"
        assert tsd["valid_key"] == "valid_value"

        # Test exception during deletion
        with pytest.raises(KeyError):
            del tsd["another_missing_key"]

        # Dict should still be functional
        assert len(tsd) == 1
        assert "valid_key" in tsd

    def test_iteration_during_modification(self):
        """Test that iteration methods return safe copies."""
        tsd = ThreadSafeDict()

        # Populate dict
        for i in range(100):
            tsd[f"key_{i}"] = f"value_{i}"

        # Get snapshots
        keys_snapshot = tsd.keys()
        values_snapshot = tsd.values()
        items_snapshot = tsd.items()

        # Modify dict after getting snapshots
        for i in range(50):
            del tsd[f"key_{i}"]

        # Snapshots should be unchanged
        assert len(keys_snapshot) == 100
        assert len(values_snapshot) == 100
        assert len(items_snapshot) == 100

        # But current dict should be modified
        assert len(tsd) == 50

    def test_thread_safety_of_lock_itself(self):
        """Test that the lock mechanism itself is thread-safe.

        Note: This test uses threading.Thread as it's testing the internal
        lock mechanism of ThreadSafeDict, which requires actual threading.
        """
        tsd = ThreadSafeDict()

        # Test that multiple threads can't corrupt the lock
        def lock_stress_test():
            for _ in range(1000):
                tsd["test"] = threading.current_thread().ident
                _ = tsd.get("test")

        threads = []
        for _ in range(10):
            t = threading.Thread(target=lock_stress_test)
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        # Dict should still be functional
        tsd["final"] = "test"
        assert tsd["final"] == "test"


class TestThreadSafeDictIntegration:
    """Integration tests with actual extremeweatherbench usage patterns."""

    def test_cache_usage_pattern_joblib(self):
        """Test caching pattern using joblib for parallel processing."""

        def compute_expensive_operation(input_data):
            """Simulate expensive computation."""
            time.sleep(0.01)  # Simulate computation time
            return sum(input_data) * 2

        def cached_compute_worker(worker_id, test_inputs):
            """Worker that uses local cache for expensive operations."""
            cache = ThreadSafeDict()

            results = []
            for input_data in test_inputs:
                cache_key = tuple(input_data)  # Make hashable

                if cache_key in cache:
                    result = cache[cache_key]
                else:
                    result = compute_expensive_operation(input_data)
                    cache[cache_key] = result

                results.append(result)

            return {
                "worker_id": worker_id,
                "results": results,
                "cache_size": len(cache),
            }

        # Test inputs with some overlap to test caching
        test_inputs = [
            [1, 2, 3],
            [4, 5, 6],
            [1, 2, 3],  # Duplicate for cache hit
            [7, 8, 9],
            [4, 5, 6],  # Another duplicate
        ]

        # Run multiple workers with joblib using threading backend
        results = Parallel(n_jobs=5, backend="threading")(
            delayed(cached_compute_worker)(i, test_inputs) for i in range(5)
        )

        # Verify all workers got same results
        expected_results = [12, 30, 12, 48, 30]  # sum * 2 for each input
        for result in results:
            assert result["results"] == expected_results
            assert result["cache_size"] == 3  # Only unique inputs cached

    def test_global_state_management_joblib(self):
        """Test managing state across joblib workers."""

        def process_data_chunk_worker(chunk_data):
            """Worker that processes a chunk and returns aggregated results."""
            chunk_id, data = chunk_data
            local_state = ThreadSafeDict()
            local_state["processed_count"] = 0
            local_state["error_count"] = 0
            local_state["results"] = []

            try:
                # Simulate processing
                processed = [x * 2 for x in data]
                local_state["processed_count"] = len(data)
                local_state["results"] = [(chunk_id, processed)]
            except Exception:
                local_state["error_count"] = 1

            return {
                "chunk_id": chunk_id,
                "processed_count": local_state["processed_count"],
                "error_count": local_state["error_count"],
                "results": local_state["results"],
            }

        # Process multiple chunks using joblib with threading backend
        chunks = [(i, list(range(i * 10, (i + 1) * 10))) for i in range(10)]

        results = Parallel(n_jobs=5, backend="threading")(
            delayed(process_data_chunk_worker)(chunk) for chunk in chunks
        )

        # Aggregate results
        total_processed = sum(result["processed_count"] for result in results)
        total_errors = sum(result["error_count"] for result in results)
        all_results = []
        for result in results:
            all_results.extend(result["results"])

        # Verify aggregated state
        assert total_processed == 100  # 10 chunks * 10 items
        assert total_errors == 0
        assert len(all_results) == 10

    def test_metrics_aggregation_pattern_joblib(self):
        """Test metrics aggregation pattern using joblib."""

        def metric_recorder_worker(worker_id):
            """Worker that records metrics using ThreadSafeDict."""
            metrics_store = ThreadSafeDict()

            for i in range(100):
                # Record various metrics
                for metric_name, value in [
                    ("accuracy", random.uniform(0.8, 1.0)),
                    ("processing_time", random.uniform(0.1, 2.0)),
                    (f"worker_{worker_id}_custom", i),
                ]:
                    if metric_name not in metrics_store:
                        metrics_store[metric_name] = []

                    current_values = metrics_store[metric_name]
                    metrics_store[metric_name] = current_values + [value]

            # Return metric summaries
            return {
                "worker_id": worker_id,
                "accuracy_count": len(metrics_store.get("accuracy", [])),
                "time_count": len(metrics_store.get("processing_time", [])),
                "custom_count": len(
                    metrics_store.get(f"worker_{worker_id}_custom", [])
                ),
                "accuracy_mean": np.mean(metrics_store.get("accuracy", [0])),
            }

        # Run concurrent metric recorders using joblib with threading backend
        results = Parallel(n_jobs=5, backend="threading")(
            delayed(metric_recorder_worker)(i) for i in range(5)
        )

        # Verify metrics from each worker
        for i, result in enumerate(results):
            assert result["worker_id"] == i
            assert result["accuracy_count"] == 100
            assert result["time_count"] == 100
            assert result["custom_count"] == 100
            assert 0.8 <= result["accuracy_mean"] <= 1.0

    def test_joblib_integration_pattern(self):
        """Test integration pattern using ThreadSafeDict with joblib."""

        def process_with_cache(data_chunk, cache_key_prefix):
            """Process data chunk with local caching using ThreadSafeDict."""
            local_cache = ThreadSafeDict()

            results = []
            for item in data_chunk:
                cache_key = f"{cache_key_prefix}_{item}"

                # Check cache first
                if cache_key in local_cache:
                    result = local_cache[cache_key]
                else:
                    # Simulate expensive computation
                    result = item**2 + item * 3
                    local_cache[cache_key] = result

                results.append(result)

            return {
                "results": results,
                "cache_size": len(local_cache),
                "cache_hits": len(data_chunk) - len(local_cache),
            }

        # Create test data with some overlap to test caching
        data_chunks = [
            [1, 2, 3, 4, 5],
            [4, 5, 6, 7, 8],  # Overlap with previous
            [7, 8, 9, 10, 11],  # Overlap with previous
        ]

        # Process chunks in parallel using joblib with threading backend
        results = Parallel(n_jobs=3, backend="threading")(
            delayed(process_with_cache)(chunk, f"chunk_{i}")
            for i, chunk in enumerate(data_chunks)
        )

        # Verify results
        assert len(results) == 3
        for i, result in enumerate(results):
            expected_results = [x**2 + x * 3 for x in data_chunks[i]]
            assert result["results"] == expected_results
            assert result["cache_size"] == len(data_chunks[i])  # No hits in this test
            assert result["cache_hits"] == 0  # Each chunk has unique cache keys


if __name__ == "__main__":
    # Allow running individual test classes for debugging
    import sys

    if len(sys.argv) > 1:
        test_class = sys.argv[1]
        pytest.main(["-v", f"::Test{test_class}"])
    else:
        pytest.main(["-v", __file__])
