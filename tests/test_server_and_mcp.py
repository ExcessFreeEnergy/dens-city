"""
Comprehensive Test Suite for dens-city FastAPI Server and Model Context Protocol (MCP) Integration.
Tests synchronous intake validation, semantic error boundaries, long-polling,
artifact pool chunking & lifecycle pruning, and MCP tool registration.
"""

import asyncio

import pytest
from fastapi.testclient import TestClient

from dens_city.server.app import app
from dens_city.server.diagnostics import translate_exception_to_semantic_error
from dens_city.server.jobs import JobManager
from dens_city.server.mcp_server import mcp
from dens_city.server.models import JobStatus, JobType
from dens_city.server.pools import ArtifactPoolStore
from dens_city.server.validators import (
    validate_single_smiles,
    validate_smiles_list,
    validate_target_spec_dict,
)


@pytest.fixture
def client():
    return TestClient(app)


# =============================================================================
# 1. Chemical Intake Gate Tests
# =============================================================================


def test_valid_smiles_validation():
    valid_smiles = [
        "CC(=O)Oc1ccccc1C(=O)O",  # Aspirin
        "C(CO)(CO)(CO)CO",  # Pentaerythritol
        "c1ccc(cc1)N(c2ccccc2)c3ccccc3",  # Triphenylamine
    ]
    is_valid, invalid_entries = validate_smiles_list(valid_smiles)
    assert is_valid is True
    assert len(invalid_entries) == 0


def test_invalid_valence_smiles_rejection():
    # Pentavalent carbon
    invalid_smi = "C=C(C)(C)(C)C"
    err = validate_single_smiles(invalid_smi)
    assert err is not None
    assert "valence" in err.reason.lower() or "violation" in err.reason.lower()


def test_unclosed_ring_smiles_rejection():
    broken_smi = "c1ccccc"
    err = validate_single_smiles(broken_smi)
    assert err is not None
    assert "invalid" in err.reason.lower()


def test_target_spec_validation():
    valid_spec = {
        "group_name": "shoe_sole_elastomer",
        "rl_reward_targets": {
            "target_elasticity": 0.95,
            "min_wall_pressure_bar": 15.0,
            "max_molecular_weight": 500.0,
        },
    }
    is_valid, errors = validate_target_spec_dict(valid_spec)
    assert is_valid is True
    assert len(errors) == 0

    invalid_spec = {
        "group_name": "broken_spec",
        "rl_reward_targets": {
            "max_molecular_weight": -100.0,  # Negative MW
            "min_wall_pressure_bar": -5.0,  # Negative pressure
        },
    }
    is_valid, errors = validate_target_spec_dict(invalid_spec)
    assert is_valid is False
    assert len(errors) >= 2


def test_real_yaml_specs_validation():
    """Verifies that all reference material YAMLs in tests/data satisfy the validator."""
    from pathlib import Path

    import yaml

    spec_dir = Path("tests/data")
    yaml_files = list(spec_dir.glob("*.yaml"))
    assert len(yaml_files) >= 5

    for y_file in yaml_files:
        spec_data = yaml.safe_load(y_file.read_text(encoding="utf-8"))
        is_valid, errors = validate_target_spec_dict(spec_data)
        assert is_valid is True, f"Reference spec {y_file.name} failed validation: {errors}"


# =============================================================================
# 2. Semantic Error Translation Tests
# =============================================================================


def test_semantic_error_translation_cuda_oom():
    cuda_exc = RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB")
    sem_err = translate_exception_to_semantic_error(cuda_exc, stage_name="run_egnn")
    assert sem_err.error_code == "CUDA_OUT_OF_MEMORY"
    assert "batch_size" in sem_err.agent_action_required.lower()


def test_semantic_error_translation_cdft_divergence():
    div_exc = RuntimeError("cDFT density profile calculation diverged: NaN encountered at grid point 42")
    sem_err = translate_exception_to_semantic_error(div_exc, stage_name="run_cdft")
    assert sem_err.error_code == "CDFT_DIVERGENCE_STERIC_CLASH"
    assert "temperature" in sem_err.agent_action_required.lower()


def test_semantic_error_translation_sa_gate():
    sa_exc = RuntimeError("Synthesizability safety gate dropped all 512 candidates")
    sem_err = translate_exception_to_semantic_error(sa_exc, stage_name="rank_pareto")
    assert sem_err.error_code == "ALL_CANDIDATES_DROPPED_SA"
    assert "max_sa_score" in sem_err.agent_action_required.lower()


# =============================================================================
# 3. Artifact Pool Store & Chunking Tests
# =============================================================================


def test_artifact_pool_store_lifecycle(tmp_path):
    store = ArtifactPoolStore(root_dir=tmp_path / "pools")

    meta = [
        {
            "index": 0,
            "name": "mol_0",
            "smiles": "CCO",
            "num_atoms": 9,
            "mw": 46.0,
            "mol2": "@<TRIPOS>MOLECULE\nmol_0\n",
        },
        {
            "index": 1,
            "name": "mol_1",
            "smiles": "CCN",
            "num_atoms": 10,
            "mw": 45.0,
            "mol2": "@<TRIPOS>MOLECULE\nmol_1\n",
        },
    ]
    cand_pool = store.create_candidate_pool(meta)
    assert cand_pool.startswith("pool_cand_")

    loaded_meta, _ = store.load_pool_candidates(cand_pool)
    assert len(loaded_meta) == 2
    assert loaded_meta[0]["name"] == "mol_0"

    # Test intelligent power-of-2 chunking
    ten_thousand_items = list(range(10000))
    chunks = ArtifactPoolStore.chunk_molecules(ten_thousand_items, chunk_size=1024)
    assert len(chunks) == 10  # 10 chunks of 1024 (last chunk 784)
    assert len(chunks[0]) == 1024
    assert len(chunks[-1]) == 10000 - 9 * 1024

    # Test cleanup
    deleted, freed = store.prune_intermediate_pools(keep_pareto=True)
    assert cand_pool in deleted
    assert freed > 0


# =============================================================================
# 4. Job Manager & Long-Polling Tests
# =============================================================================


def test_job_manager_lifecycle(tmp_path):
    async def _async_test():
        mgr = JobManager(db_dir=tmp_path / "jobs")
        job_id = mgr.enqueue_job(JobType.TRAIN_SWARM, {"spec": "oled"})
        assert job_id.startswith("job_")

        job = mgr.get_job(job_id)
        assert job is not None
        assert job.status == JobStatus.PENDING

        # Test cooperative cancel
        cancelled = mgr.cancel_job(job_id)
        assert cancelled is True
        job_after_cancel = mgr.get_job(job_id)
        assert job_after_cancel.status == JobStatus.CANCELLED

        # Test long-polling immediately resolves for terminal status
        res = await mgr.wait_for_job(job_id, timeout_seconds=5)
        assert res.status == JobStatus.CANCELLED

    asyncio.run(_async_test())


# =============================================================================
# 5. FastAPI Endpoints Tests
# =============================================================================


def test_fastapi_healthz(client):
    resp = client.get("/healthz")
    assert resp.status_code == 200
    data = resp.json()
    assert "status" in data
    assert "timestamp" in data


def test_fastapi_synchronous_chemical_rejection(client):
    # Pass an invalid SMILES with pentavalent carbon
    bad_payload = {
        "smiles_list": ["C=C(C)(C)(C)C"],
        "solvent_id": "water",
    }
    resp = client.post("/api/v1/stages/run-cdft", json=bad_payload)
    assert resp.status_code == 400
    err_detail = resp.json()["detail"]
    assert err_detail["error_code"] == "CHEMICAL_INTAKE_REJECTED"


def test_fastapi_job_enqueue_and_get(client):
    payload = {
        "target_spec": {"group_name": "test_mat"},
        "train_steps": 1000,
        "num_candidates": 16,
    }
    resp = client.post("/api/v1/pipeline/full", json=payload)
    assert resp.status_code == 202
    data = resp.json()
    job_id = data["job_id"]
    assert job_id.startswith("job_")

    # Get job status without long-polling
    status_resp = client.get(f"/api/v1/jobs/{job_id}?wait_for_completion=false")
    assert status_resp.status_code == 200
    st_data = status_resp.json()
    assert st_data["job_id"] == job_id


def test_fastapi_cleanup_endpoint(client):
    resp = client.post("/api/v1/pools/cleanup", json={"mode": "keep_pareto_only"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["mode"] == "keep_pareto_only"
    assert "freed_mb" in data


# =============================================================================
# 6. MCP Tool Registration Tests
# =============================================================================


def test_mcp_tools_registered():
    # Verify all 10 tools exist in the MCPServer
    tool_names = [
        "run_full_pipeline",
        "train_swarm_agent",
        "sample_candidates",
        "run_cdft_thermo",
        "run_egnn_quantum",
        "rank_pareto_frontier",
        "get_job_status",
        "cancel_job",
        "validate_spec",
        "cleanup_artifacts",
    ]
    # Check tool names using list_tools
    tools = asyncio.run(mcp.list_tools())
    registered_names = [t.name for t in tools]

    for expected in tool_names:
        assert expected in registered_names, f"Expected tool '{expected}' not found in MCP server."
