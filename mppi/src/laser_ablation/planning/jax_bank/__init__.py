"""Optional JAX proposal, trajectory-bank, and state-tail infrastructure."""

from laser_ablation.planning.jax_bank.contracts import (
    BankTrajectory,
    MatchingTailBank,
    ProvisionalPlan,
    QuickRolloutBatch,
    StaticTaskTensors,
    mpc_handoff,
)
from laser_ablation.planning.jax_bank.planner import JaxPlanBankPlanner
from laser_ablation.planning.jax_bank.mppi_repairer import MPPIPlanRepairer
from laser_ablation.planning.jax_bank.global_seeds import HybridGlobalSeeds
from laser_ablation.planning.jax_bank.exact_segment import ExactSegmentBeam
from laser_ablation.planning.jax_bank.comparison_library import (
    ComparisonROILibrary,
    build_comparison_roi_library,
    load_comparison_roi_library,
    save_comparison_roi_library,
)
from laser_ablation.planning.jax_bank.linear_contracts import (
    DeviceLinearizationWorkspace,
    LinearizationLibrary,
    LinearizedActionBatch,
    load_linearization_library,
    save_linearization_library,
)
from laser_ablation.planning.jax_bank.linear_rollout import (
    rollout_linearized_batch,
    rollout_linearized_in_batches,
)
from laser_ablation.planning.jax_bank.linearization import linearize_seed_batch
from laser_ablation.planning.jax_bank.recentered_workspace import build_recentered_workspace
from laser_ablation.planning.jax_bank.segment_beam import SegmentBeam
from laser_ablation.planning.jax_bank.terminal_rollout import TerminalRolloutBatch
from laser_ablation.planning.jax_bank.similarity import roi_cosine_similarity
from laser_ablation.planning.jax_bank.repair import (
    GlobalReplanRequired,
    MPPIRepairConfig,
    PlanRepairer,
    RepairRequest,
    RepairTrigger,
)

__all__ = [
    "BankTrajectory",
    "ComparisonROILibrary",
    "JaxPlanBankPlanner",
    "HybridGlobalSeeds",
    "DeviceLinearizationWorkspace",
    "ExactSegmentBeam",
    "LinearizationLibrary",
    "LinearizedActionBatch",
    "MatchingTailBank",
    "MPPIPlanRepairer",
    "MPPIRepairConfig",
    "PlanRepairer",
    "ProvisionalPlan",
    "QuickRolloutBatch",
    "RepairRequest",
    "RepairTrigger",
    "StaticTaskTensors",
    "TerminalRolloutBatch",
    "SegmentBeam",
    "GlobalReplanRequired",
    "mpc_handoff",
    "build_comparison_roi_library",
    "linearize_seed_batch",
    "build_recentered_workspace",
    "rollout_linearized_batch",
    "rollout_linearized_in_batches",
    "roi_cosine_similarity",
    "save_linearization_library",
    "load_linearization_library",
    "save_comparison_roi_library",
    "load_comparison_roi_library",
]
