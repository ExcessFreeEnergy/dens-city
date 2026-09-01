"""
WikiSkill: Persistent Knowledge Co-Evolution System for dens-city.

Implements the three-layer knowledge architecture:
- Raw Layer: Immutable execution traces and failure logs
- Wiki Layer: Persistent, compounding knowledge base (patterns, logs, skill-impact tracker)
- Skills & Rules Layer: Active procedural instructions and directory rules
"""

from dens_city.wikiskill.gating import GatingHarness, GatingResult
from dens_city.wikiskill.maintainer import WikiMaintainer
from dens_city.wikiskill.proposer import Proposal, ProposalAction, SkillProposer
from dens_city.wikiskill.trace_recorder import ExecutionTrace, RawTraceRecorder
from dens_city.wikiskill.wiki_manager import PatchOperation, WikiManager, WikiPattern

__all__ = [
    "RawTraceRecorder",
    "ExecutionTrace",
    "WikiManager",
    "WikiPattern",
    "PatchOperation",
    "WikiMaintainer",
    "SkillProposer",
    "Proposal",
    "ProposalAction",
    "GatingHarness",
    "GatingResult",
]
