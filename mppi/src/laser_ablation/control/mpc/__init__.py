"""Dependency-light contracts for the isolated DEVELOPMENT energy MPC.

CasADi-dependent transition, NLP, and controller classes are imported from
their explicit submodules so ordinary framework imports do not require the
optional MPC dependency. The implementation is not part of the frozen method.
"""

from laser_ablation.control.mpc.config import (
    EnergyMPCConfig,
    MPCFingerprints,
    WeightSchedule,
    energy_mpc_config_fingerprint,
    fingerprints_from_shared_configs,
    load_energy_mpc_config,
)
from laser_ablation.control.mpc.contracts import (
    EnergyMPCFailureReason,
    EnergyMPCProblem,
    EnergyMPCSolution,
)

__all__ = [
    "EnergyMPCConfig",
    "EnergyMPCFailureReason",
    "EnergyMPCProblem",
    "EnergyMPCSolution",
    "MPCFingerprints",
    "WeightSchedule",
    "energy_mpc_config_fingerprint",
    "fingerprints_from_shared_configs",
    "load_energy_mpc_config",
]
