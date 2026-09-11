"""
Semantic Error Translation Layer for dens-city server.
Transforms low-level CUDA, compiler, and physics engine exceptions into
structured, agent-actionable diagnostics with prescriptive recovery guidance.
"""

from __future__ import annotations

import traceback
from typing import Optional

from dens_city.server.models import SemanticError


def translate_exception_to_semantic_error(
    exc: BaseException,
    stage_name: str = "pipeline",
    context: Optional[dict] = None,
) -> SemanticError:
    """
    Translates an unhandled exception or simulation failure into a structured SemanticError.
    """
    err_str = str(exc)
    err_type = type(exc).__name__
    tb_str = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))

    # 1. CUDA Out Of Memory
    if "out of memory" in err_str.lower() or "cuda oom" in err_str.lower():
        return SemanticError(
            error_code="CUDA_OUT_OF_MEMORY",
            summary=f"GPU VRAM exhausted during {stage_name} batch tensor allocation.",
            agent_action_required=(
                "Re-run with a smaller batch size (e.g. batch_size=16 or egnn_batch_size=16). "
                "If evaluating thousands of candidates, consider chunked execution via smaller candidate pools."
            ),
            technical_detail=f"{err_type}: {err_str}",
        )

    # 2. cDFT Solver Divergence / Steric Clash
    if "divergence" in err_str.lower() or "nan" in err_str.lower() or "density profile" in err_str.lower():
        return SemanticError(
            error_code="CDFT_DIVERGENCE_STERIC_CLASH",
            summary="Classical DFT Euler-Lagrange density profile diverged due to severe atomic steric overlap or extreme chemical potential.",
            agent_action_required=(
                "Re-run Stage 2 (sample_candidates) with higher temperature (e.g. temperature=1.2) "
                "to escape trapped strained conformations, or check that target_spec min_wall_pressure_bar is physically reasonable (< 50 bar)."
            ),
            technical_detail=f"{err_type}: {err_str}",
        )

    # 3. All Candidates Dropped by Synthesizability Gate
    if "synthesizability safety gate" in err_str.lower() or "all candidates dropped" in err_str.lower():
        return SemanticError(
            error_code="ALL_CANDIDATES_DROPPED_SA",
            summary="All candidate molecules were rejected by the Stage 5 Synthesizability Safety Gate because their RDKit SA Score exceeded the threshold.",
            agent_action_required=(
                "Increase max_sa_score (e.g. max_sa_score=7.0 or disable_sa_filter=True), "
                "or re-train the Stage 1 swarm policy with sa_penalty=True to bias growth toward accessible chemistry."
            ),
            technical_detail=f"{err_type}: {err_str}",
        )

    # 4. Zero Valid Candidates Generated in Stage 2
    if "no valid candidate molecules" in err_str.lower() or "zero valid candidates" in err_str.lower():
        return SemanticError(
            error_code="ZERO_VALID_CANDIDATES_GENERATED",
            summary="Stage 2 sampling produced zero chemically valid candidate molecules within the rollout step budget.",
            agent_action_required=(
                "Inspect target_spec constraints: verify max_molecular_weight is large enough for scaffolds "
                "and allowed_atomic_numbers contains compatible elements. Alternatively, train Stage 1 policy for more steps (> 50,000)."
            ),
            technical_detail=f"{err_type}: {err_str}",
        )

    # 5. Missing / Corrupt Artifact Pool
    if "pool not found" in err_str.lower() or "pool_id" in err_str.lower():
        return SemanticError(
            error_code="ARTIFACT_POOL_NOT_FOUND",
            summary="The specified pool_id could not be resolved from persistent storage.",
            agent_action_required=(
                "Verify the pool_id string against the result of previous stage tool calls, "
                "or call sample_candidates or run_cdft_thermo to generate a fresh candidate pool."
            ),
            technical_detail=f"{err_type}: {err_str}",
        )

    # 6. Chemical Valence / SMILES Failure
    if "valence" in err_str.lower() or "smiles" in err_str.lower():
        return SemanticError(
            error_code="CHEMICAL_VALENCE_VIOLATION",
            summary="One or more chemical structures violated fundamental valence rules or ring closure constraints.",
            agent_action_required=(
                "Call validate_spec or inspect SMILES list with standard RDKit rules. "
                "Ensure carbon valency is <= 4 and all aromatic rings are closed."
            ),
            technical_detail=f"{err_type}: {err_str}",
        )

    # 7. Job Cancelled
    if "cancelled" in err_str.lower() or "abort" in err_str.lower():
        return SemanticError(
            error_code="JOB_CANCELLED_BY_AGENT",
            summary=f"The {stage_name} job was cancelled via agent request.",
            agent_action_required="No recovery required. You may submit a new job with updated parameters.",
            technical_detail=f"{err_type}: {err_str}",
        )

    # Generic Fallback
    return SemanticError(
        error_code=f"STAGE_{stage_name.upper()}_FAILURE",
        summary=f"Execution halted during {stage_name}: {err_str[:200]}",
        agent_action_required=(
            "Review latest logs from get_job_status. Check specification parameters, "
            "reduce batch size, or re-run the stage."
        ),
        technical_detail=tb_str[-1000:] if len(tb_str) > 1000 else tb_str,
    )
