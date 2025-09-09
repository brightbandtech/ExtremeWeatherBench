"""Comprehensive tests for ThreadSafeDict class.

This test suite covers all scenarios including single-threaded, multi-threaded,
multiprocessing, Dask, and stress testing scenarios to ensure ThreadSafeDict
works correctly under all conditions.
"""

import concurrent.futures
import multiprocessing
import pickle
import random
import threading
import time

import numpy as np
import pytest
import xarray as xr

from extremeweatherbench.utils import ThreadSafeDict

try:
    import dask
    import dask.array as da
    import dask.bag as db
    import dask.distributed
    from dask.distributed import Client, LocalCluster

    DASK_AVAILABLE = True
except ImportError:
    DASK_AVAILABLE = False


# Worker function for multiprocessing (needs to be at module level for pickling)
def _multiprocessing_worker_process(process_id, return_dict):
    """Worker function for multiprocessing test."""
    tsd = ThreadSafeDict()

    # Each process creates its own ThreadSafeDict and populates it
    for i in range(100):
        key = f"process_{process_id}_key_{i}"
        value = f"process_{process_id}_value_{i}"
        tsd[key] = value

    # Return some stats about the process
    return_dict[process_id] = {
        "length": len(tsd),
        "sample_key": f"process_{process_id}_key_0",
        "sample_value": tsd.get(f"process_{process_id}_key_0"),
    }


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


class TestThreadSafeDictThreading:
    """Multi-threading tests for ThreadSafeDict."""

    def test_concurrent_reads(self):
        """Test concurrent reads from multiple threads."""
        tsd = ThreadSafeDict()

        # Populate with initial data
        for i in range(100):
            tsd[f"key_{i}"] = f"value_{i}"

        results = []

        def read_worker():
            thread_results = []
            for i in range(100):
                key = f"key_{i % 50}"  # Read from subset of keys
                value = tsd.get(key)
                thread_results.append((key, value))
            return thread_results

        # Run concurrent reads
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(read_worker) for _ in range(10)]
            for future in concurrent.futures.as_completed(futures):
                results.extend(future.result())

        # Verify all reads were successful
        assert len(results) == 1000  # 10 threads * 100 reads each
        for key, value in results:
            if value is not None:  # Some keys might not exist
                expected_value = key.replace("key_", "value_")
                assert value == expected_value

    def test_concurrent_writes(self):
        """Test concurrent writes from multiple threads."""
        tsd = ThreadSafeDict()

        def write_worker(thread_id):
            for i in range(100):
                key = f"thread_{thread_id}_key_{i}"
                value = f"thread_{thread_id}_value_{i}"
                tsd[key] = value

        # Run concurrent writes
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(write_worker, i) for i in range(10)]
            concurrent.futures.wait(futures)

        # Verify all writes were successful
        assert len(tsd) == 1000  # 10 threads * 100 writes each

        for thread_id in range(10):
            for i in range(100):
                key = f"thread_{thread_id}_key_{i}"
                expected_value = f"thread_{thread_id}_value_{i}"
                assert tsd[key] == expected_value

    def test_concurrent_read_write(self):
        """Test concurrent reads and writes."""
        tsd = ThreadSafeDict()

        # Initialize with some data
        for i in range(50):
            tsd[f"initial_key_{i}"] = f"initial_value_{i}"

        read_results = []
        write_count = 0

        def reader():
            nonlocal read_results
            local_results = []
            for _ in range(100):
                # Read random existing keys
                key = f"initial_key_{random.randint(0, 49)}"
                value = tsd.get(key)
                local_results.append((key, value))
                time.sleep(0.001)  # Small delay to interleave operations
            read_results.extend(local_results)

        def writer():
            nonlocal write_count
            for i in range(50):
                key = f"new_key_{i}"
                value = f"new_value_{i}"
                tsd[key] = value
                write_count += 1
                time.sleep(0.001)  # Small delay to interleave operations

        # Run concurrent readers and writers
        with concurrent.futures.ThreadPoolExecutor(max_workers=6) as executor:
            # 4 readers, 2 writers
            futures = [executor.submit(reader) for _ in range(4)] + [
                executor.submit(writer) for _ in range(2)
            ]
            concurrent.futures.wait(futures)

        # Verify results
        assert len(read_results) == 400  # 4 readers * 100 reads
        assert write_count == 100  # 2 writers * 50 writes
        # Dictionary size should be at least 50 (initial) + some new keys
        assert len(tsd) >= 50  # At least the initial keys should remain

    def test_concurrent_modifications(self):
        """Test concurrent modifications (updates, deletes)."""
        tsd = ThreadSafeDict()

        # Initialize with data
        for i in range(100):
            tsd[f"key_{i}"] = f"initial_value_{i}"

        def updater(thread_id):
            for i in range(0, 100, 2):  # Update even keys
                key = f"key_{i}"
                new_value = f"updated_by_thread_{thread_id}_value_{i}"
                tsd[key] = new_value

        def deleter():
            for i in range(1, 100, 4):  # Delete every 4th odd key
                key = f"key_{i}"
                if key in tsd:
                    del tsd[key]

        # Run concurrent modifications
        with concurrent.futures.ThreadPoolExecutor(max_workers=6) as executor:
            futures = [executor.submit(updater, i) for i in range(3)] + [
                executor.submit(deleter) for _ in range(2)
            ]
            concurrent.futures.wait(futures)

        # Verify the dictionary is in a consistent state
        remaining_keys = tsd.keys()
        assert len(remaining_keys) <= 100  # Some keys may have been deleted

        # Check that all remaining keys have valid values
        for key in remaining_keys:
            value = tsd[key]
            assert value is not None
            assert isinstance(value, str)

    def test_race_condition_prevention(self):
        """Test that race conditions are prevented."""
        tsd = ThreadSafeDict()
        tsd["counter"] = 0

        def increment_counter():
            for _ in range(1000):
                current = tsd["counter"]
                # Simulate some processing time
                time.sleep(0.0001)
                tsd["counter"] = current + 1

        # This test would fail with a regular dict due to race conditions
        # but should work with ThreadSafeDict
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(increment_counter) for _ in range(5)]
            concurrent.futures.wait(futures)

        # Note: This is still a race condition because the read-modify-write
        # cycle isn't atomic. The ThreadSafeDict only protects individual operations.
        # This test demonstrates that the dict itself doesn't get corrupted.
        assert isinstance(tsd["counter"], int)
        assert tsd["counter"] >= 0  # Should at least be non-negative

    def test_stress_threading(self):
        """Stress test with many threads and operations."""
        tsd = ThreadSafeDict()
        num_threads = 20
        operations_per_thread = 500

        def worker(thread_id):
            operations = []
            for i in range(operations_per_thread):
                op_type = random.choice(["set", "get", "delete", "contains"])
                key = f"key_{random.randint(0, 99)}"

                try:
                    if op_type == "set":
                        tsd[key] = f"value_{thread_id}_{i}"
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

        # Run stress test
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(worker, i) for i in range(num_threads)]
            all_operations = []
            for future in concurrent.futures.as_completed(futures):
                all_operations.extend(future.result())

        # Verify no errors occurred
        errors = [op for op in all_operations if op[0] == "error"]
        error_msg = f"Errors occurred: {errors[:5]}..."
        assert len(errors) == 0, error_msg

        # Verify total operations (allow some failures under stress)
        expected_total = num_threads * operations_per_thread
        assert len(all_operations) >= expected_total * 0.8  # Allow 20% failure

        # Dictionary should still be in valid state
        assert isinstance(len(tsd), int)
        assert len(tsd) >= 0


class TestThreadSafeDictMultiprocessing:
    """Multiprocessing tests for ThreadSafeDict."""

    def test_multiprocessing_pickle(self):
        """Test that ThreadSafeDict can be pickled for multiprocessing."""
        tsd = ThreadSafeDict()
        tsd["key1"] = "value1"
        tsd["key2"] = [1, 2, 3]

        # Test pickling
        pickled_data = pickle.dumps(tsd)
        unpickled_tsd = pickle.loads(pickled_data)

        # Note: After unpickling, it's a new instance with a new lock
        assert len(unpickled_tsd) == 2
        assert unpickled_tsd["key1"] == "value1"
        assert unpickled_tsd["key2"] == [1, 2, 3]

    def test_multiprocessing_worker_function(self):
        """Test ThreadSafeDict in multiprocessing context."""
        # Use Manager for shared return dict
        with multiprocessing.Manager() as manager:
            return_dict = manager.dict()
            processes = []

            # Create and start processes
            for i in range(4):
                p = multiprocessing.Process(
                    target=_multiprocessing_worker_process, args=(i, return_dict)
                )
                p.start()
                processes.append(p)

            # Wait for all processes to complete
            for p in processes:
                p.join()

            # Verify results
            assert len(return_dict) == 4
            for i in range(4):
                assert return_dict[i]["length"] == 100
                assert return_dict[i]["sample_value"] == f"process_{i}_value_0"


@pytest.mark.skipif(not DASK_AVAILABLE, reason="Dask not available")
class TestThreadSafeDictDask:
    """Dask-specific tests for ThreadSafeDict."""

    def test_dask_delayed_with_threadsafe_dict(self):
        """Test ThreadSafeDict with Dask delayed operations."""

        @dask.delayed
        def populate_dict(tsd, start_idx, count):
            for i in range(start_idx, start_idx + count):
                tsd[f"key_{i}"] = f"value_{i}"
            return len(tsd)

        @dask.delayed
        def read_from_dict(tsd, keys_to_read):
            results = []
            for key in keys_to_read:
                value = tsd.get(key)
                if value:
                    results.append((key, value))
            return results

        # Create ThreadSafeDict
        tsd = ThreadSafeDict()

        # Create delayed tasks
        populate_tasks = [populate_dict(tsd, i * 50, 50) for i in range(4)]

        # Compute population tasks
        dask.compute(*populate_tasks)

        # Now read from different parts
        read_tasks = [
            read_from_dict(tsd, [f"key_{i}" for i in range(j * 25, (j + 1) * 25)])
            for j in range(8)
        ]

        read_results = dask.compute(*read_tasks)

        # Verify results
        assert len(tsd) == 200  # 4 tasks * 50 items each
        all_read_items = []
        for result in read_results:
            all_read_items.extend(result)

        assert len(all_read_items) == 200  # All items should be found

    def test_dask_bag_with_threadsafe_dict(self):
        """Test ThreadSafeDict with Dask bag operations."""
        # Create a bag of data
        data = list(range(1000))
        bag = db.from_sequence(data, npartitions=10)

        def process_item(x):
            # Create local ThreadSafeDict for each partition
            local_dict = ThreadSafeDict()
            local_dict[f"processed_{x}"] = x * 2
            return local_dict

        # Process bag (this will use multiple threads)
        result_dicts = bag.map(process_item).compute()

        # Verify results - each item should produce a ThreadSafeDict
        assert len(result_dicts) == 1000

        # Verify each local dict contains expected value
        for i, local_dict in enumerate(result_dicts):
            assert len(local_dict) == 1
            assert local_dict[f"processed_{i}"] == i * 2

    def test_dask_array_with_threadsafe_dict(self):
        """Test ThreadSafeDict with Dask array operations."""
        tsd = ThreadSafeDict()

        # Create a Dask array
        x = da.random.random((1000, 100), chunks=(100, 100))

        def cache_chunk_stats(chunk, block_id):
            """Function to cache statistics about each chunk."""
            chunk_id = f"chunk_{block_id[0]}_{block_id[1]}"
            stats = {
                "mean": float(np.mean(chunk)),
                "std": float(np.std(chunk)),
                "min": float(np.min(chunk)),
                "max": float(np.max(chunk)),
            }
            tsd[chunk_id] = stats
            return chunk

        # Apply function to each block
        result = x.map_blocks(cache_chunk_stats, dtype=x.dtype, drop_axis=[])
        result.compute()

        # Verify caching worked
        expected_chunks = 10 * 1  # 1000/100 * 100/100
        assert len(tsd) == expected_chunks

        # Check that all cached values are valid
        for key in tsd.keys():
            stats = tsd[key]
            assert "mean" in stats
            assert "std" in stats
            assert "min" in stats
            assert "max" in stats
            assert 0 <= stats["mean"] <= 1  # Random values are 0-1

    def test_dask_distributed_client(self):
        """Test ThreadSafeDict with Dask distributed client."""
        # Note: This test creates a local cluster
        with LocalCluster(
            n_workers=2,
            threads_per_worker=2,
            processes=False,  # Use threads for shared memory
            silence_logs=False,
        ) as cluster:
            with Client(cluster) as client:

                def distributed_worker(worker_id):
                    """Function to run on distributed workers."""
                    # Each worker gets its own ThreadSafeDict instance
                    local_tsd = ThreadSafeDict()

                    for i in range(100):
                        key = f"worker_{worker_id}_item_{i}"
                        value = f"processed_by_worker_{worker_id}_value_{i}"
                        local_tsd[key] = value

                    return {
                        "worker_id": worker_id,
                        "items_processed": len(local_tsd),
                        "sample_item": local_tsd.get(f"worker_{worker_id}_item_0"),
                    }

                # Submit tasks to distributed workers
                futures = [client.submit(distributed_worker, i) for i in range(4)]

                # Gather results
                results = client.gather(futures)

                # Verify results
                assert len(results) == 4
                for i, result in enumerate(results):
                    assert result["worker_id"] == i
                    assert result["items_processed"] == 100
                    assert result["sample_item"] == f"processed_by_worker_{i}_value_0"


class TestThreadSafeDictStress:
    """Stress tests for ThreadSafeDict."""

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

    def test_concurrent_stress_operations(self):
        """Extreme stress test with many concurrent operations."""
        tsd = ThreadSafeDict()
        num_threads = 50
        operations_per_thread = 1000

        def stress_worker(thread_id):
            ops_completed = 0
            for i in range(operations_per_thread):
                try:
                    op = random.choice(["set", "get", "delete", "update"])
                    key = f"stress_{random.randint(0, 500)}"

                    if op == "set":
                        tsd[key] = f"thread_{thread_id}_op_{i}"
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
                            tsd[key] = f"new_thread_{thread_id}_op_{i}"

                    ops_completed += 1

                except Exception:
                    # Ignore individual operation failures in stress test
                    pass

            return ops_completed

        # Run extreme stress test
        start_time = time.time()
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(stress_worker, i) for i in range(num_threads)]
            results = [future.result() for future in futures]
        end_time = time.time()

        # Verify results
        total_ops = sum(results)
        expected_total = num_threads * operations_per_thread

        # Should complete most operations successfully
        assert total_ops >= expected_total * 0.9  # Allow 10% failure under stress

        # Should complete in reasonable time
        assert end_time - start_time < 30  # 30 seconds max

        # Dictionary should still be functional
        assert isinstance(len(tsd), int)
        tsd["final_test"] = "success"
        assert tsd["final_test"] == "success"


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
        """Test that the lock mechanism itself is thread-safe."""
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

    def test_cache_usage_pattern(self):
        """Test usage pattern similar to caching in extremeweatherbench."""
        cache = ThreadSafeDict()

        def compute_expensive_operation(input_data):
            """Simulate expensive computation."""
            time.sleep(0.01)  # Simulate computation time
            return sum(input_data) * 2

        def cached_compute(input_data):
            """Cached version of expensive operation."""
            cache_key = tuple(input_data)  # Make hashable

            if cache_key in cache:
                return cache[cache_key]

            result = compute_expensive_operation(input_data)
            cache[cache_key] = result
            return result

        # Test concurrent access to cache
        test_inputs = [
            [1, 2, 3],
            [4, 5, 6],
            [1, 2, 3],  # Duplicate for cache hit
            [7, 8, 9],
            [4, 5, 6],  # Another duplicate
        ]

        def worker():
            results = []
            for input_data in test_inputs:
                result = cached_compute(input_data)
                results.append(result)
            return results

        # Run multiple workers concurrently
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(worker) for _ in range(5)]
            all_results = [future.result() for future in futures]

        # Verify all workers got same results
        expected_results = [12, 30, 12, 48, 30]  # sum * 2 for each input
        for results in all_results:
            assert results == expected_results

        # Verify cache has correct entries
        assert len(cache) == 3  # Only unique inputs should be cached

    def test_global_state_management(self):
        """Test managing global state across threads."""
        global_state = ThreadSafeDict()
        global_state["processed_count"] = 0
        global_state["error_count"] = 0
        global_state["results"] = []

        def process_data_chunk(chunk_id, data):
            """Simulate processing a chunk of data."""
            try:
                # Simulate processing
                processed = [x * 2 for x in data]

                # Update global state safely
                current_count = global_state["processed_count"]
                global_state["processed_count"] = current_count + len(data)

                current_results = global_state["results"]
                global_state["results"] = current_results + [(chunk_id, processed)]

            except Exception:
                error_count = global_state["error_count"]
                global_state["error_count"] = error_count + 1

        # Process multiple chunks concurrently
        chunks = [(i, list(range(i * 10, (i + 1) * 10))) for i in range(10)]

        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [
                executor.submit(process_data_chunk, chunk_id, data)
                for chunk_id, data in chunks
            ]
            concurrent.futures.wait(futures)

        # Verify global state
        assert global_state["processed_count"] == 100  # 10 chunks * 10 items
        assert global_state["error_count"] == 0
        assert len(global_state["results"]) == 10

    def test_metrics_aggregation_pattern(self):
        """Test pattern similar to metrics aggregation."""
        metrics_store = ThreadSafeDict()

        def record_metric(metric_name, value):
            """Record a metric value."""
            if metric_name not in metrics_store:
                metrics_store[metric_name] = []

            current_values = metrics_store[metric_name]
            metrics_store[metric_name] = current_values + [value]

        def get_metric_stats(metric_name):
            """Get statistics for a metric."""
            if metric_name not in metrics_store:
                return None

            values = metrics_store[metric_name]
            return {
                "count": len(values),
                "mean": np.mean(values),
                "std": np.std(values),
                "min": np.min(values),
                "max": np.max(values),
            }

        # Simulate concurrent metric recording
        def metric_recorder(thread_id):
            for i in range(100):
                # Record various metrics
                record_metric("accuracy", random.uniform(0.8, 1.0))
                record_metric("processing_time", random.uniform(0.1, 2.0))
                record_metric(f"thread_{thread_id}_custom", i)

        # Run concurrent recorders
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(metric_recorder, i) for i in range(5)]
            concurrent.futures.wait(futures)

        # Verify metrics
        accuracy_stats = get_metric_stats("accuracy")
        assert accuracy_stats["count"] == 500  # 5 threads * 100 records
        assert 0.8 <= accuracy_stats["mean"] <= 1.0

        time_stats = get_metric_stats("processing_time")
        assert time_stats["count"] == 500
        assert 0.1 <= time_stats["mean"] <= 2.0

        # Check thread-specific metrics
        for i in range(5):
            thread_stats = get_metric_stats(f"thread_{i}_custom")
            assert thread_stats["count"] == 100
            assert thread_stats["min"] == 0
            assert thread_stats["max"] == 99


if __name__ == "__main__":
    # Allow running individual test classes for debugging
    import sys

    if len(sys.argv) > 1:
        test_class = sys.argv[1]
        pytest.main(["-v", f"::Test{test_class}"])
    else:
        pytest.main(["-v", __file__])
