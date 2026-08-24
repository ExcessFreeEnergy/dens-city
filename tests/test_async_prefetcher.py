"""
Unit tests for AsyncBatchPrefetcher, ProcessPoolExecutor parallel assembly, and execute_prepared_batch.
"""

import tempfile
from pathlib import Path

from dens_city.utils.pipeline import (
    AsyncArtifactWriter,
    AsyncBatchPrefetcher,
    MaterialPipelineTask,
    PipelineStatus,
    execute_prepared_batch,
)


def test_async_prefetcher_streaming():
    """Validates that AsyncBatchPrefetcher correctly streams pre-assembled batches across workers."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tasks = [
            MaterialPipelineTask(
                material_path_or_name="argon",
                out_dir=tmpdir,
                cdft_steps=5,
                bg_steps=5,
                bg_samples=8,
            ),
            MaterialPipelineTask(
                material_path_or_name="water",
                out_dir=tmpdir,
                cdft_steps=5,
                bg_steps=5,
                bg_samples=8,
            ),
            MaterialPipelineTask(
                material_path_or_name="benzene",
                out_dir=tmpdir,
                cdft_steps=5,
                bg_steps=5,
                bg_samples=8,
            ),
        ]

        # Chunk into batch_size=2
        task_chunks = [tasks[0:2], tasks[2:3]]
        prefetcher = AsyncBatchPrefetcher(
            task_chunks=task_chunks,
            batch_size=2,
            prefetch_depth=2,
        ).start()

        batches = list(prefetcher)
        assert len(batches) == 2, f"Expected 2 prepared batches, got {len(batches)}"

        # Verify Batch 0
        b0 = batches[0]
        assert len(b0.tasks) == 2
        assert len(b0.loaded_materials) == 2
        assert b0.batch_size == 2

        # Verify Batch 1
        b1 = batches[1]
        assert len(b1.tasks) == 1
        assert len(b1.loaded_materials) == 1
        assert b1.batch_size == 2


def test_execute_prepared_batch_execution():
    """Validates that execute_prepared_batch executes cDFT and Boltzmann Generator on a pre-assembled batch."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tasks = [
            MaterialPipelineTask(
                material_path_or_name="argon",
                out_dir=tmpdir,
                cdft_steps=10,
                bg_steps=5,
                bg_samples=8,
            ),
            MaterialPipelineTask(
                material_path_or_name="water",
                out_dir=tmpdir,
                cdft_steps=10,
                bg_steps=5,
                bg_samples=8,
            ),
        ]

        prefetcher = AsyncBatchPrefetcher(
            task_chunks=[tasks],
            batch_size=2,
        ).start()

        prepared_batch = next(prefetcher)
        async_writer = AsyncArtifactWriter()

        try:
            results = execute_prepared_batch(prepared_batch, async_writer=async_writer)
        finally:
            prefetcher.close()
            async_writer.flush()
            async_writer.close()

        assert len(results) == 2
        assert results[0].status == PipelineStatus.SUCCESS.value
        assert results[1].status == PipelineStatus.SUCCESS.value
        assert results[0].num_sites == 1
        assert results[1].num_sites == 3
        assert (Path(tmpdir) / "argon" / "trajectory.xyz").exists()
        assert (Path(tmpdir) / "water" / "density_profile.npy").exists()
