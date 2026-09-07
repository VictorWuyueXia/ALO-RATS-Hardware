"""Exact-only Geometry-Aware Global finalist authority."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import importlib
import json
from pathlib import Path
import platform
import struct
from typing import Any

import numpy as np
import scipy

from laser_ablation.core.plan import PlanVerification
from laser_ablation.core.state import VoxelState
from laser_ablation.planning.global_3d.interface import (
    PlanCandidate,
    VerifiedGlobalPlan,
    canonical_action_sequence_hash,
    physical_action_sequences_equal,
)
from laser_ablation.planning.global_3d.terminal_verification import (
    ExactPlanVerifier,
    exact_terminal_key,
)


_AUTHORITY_DOMAIN = b"laser-ablation/exact-authority/v1\0"
_STATE_DOMAIN = b"laser-ablation/voxel-state/v1\0"
_REQUIRED_SOURCE_MODULES = (
    "laser_ablation.physics.exact_voxel",
    "laser_ablation.physics.super_gaussian",
    "laser_ablation.metrics",
    "laser_ablation.metrics.volumetric",
    "laser_ablation.planning.global_3d.terminal_verification",
    "laser_ablation.planning.global_3d.interface",
    "laser_ablation.planning.global_3d.exact_finalist",
    "laser_ablation.core.actions",
    "laser_ablation.core.state",
    "laser_ablation.core.metrics",
    "laser_ablation.core.plan",
)


@dataclass(frozen=True)
class VerificationAuthorityIdentity:
    """Live-worktree identity of the exact physical and metric authority."""

    digest: str
    manifest_json: str
    cache_reusable: bool

    def __post_init__(self) -> None:
        expected = sha256(
            _AUTHORITY_DOMAIN + self.manifest_json.encode("ascii")
        ).hexdigest()
        if self.digest != expected:
            raise ValueError("authority digest does not bind manifest_json")
        manifest = json.loads(self.manifest_json)
        if manifest.get("schema") != "laser-ablation/exact-authority/v1":
            raise ValueError("unsupported exact-authority manifest schema")


@dataclass(frozen=True)
class ExactVerificationCacheEntry:
    """Exact rollout plus hashes binding it to plan, state, and authority."""

    verification: PlanVerification
    planned_action_hash: str
    executed_prefix_hash: str
    state_hash: str
    verification_authority_hash: str

    def __post_init__(self) -> None:
        if (
            canonical_action_sequence_hash(self.verification.executed_actions)
            != self.executed_prefix_hash
        ):
            raise ValueError("cache entry prefix hash does not bind verification")
        for value in (
            self.planned_action_hash,
            self.executed_prefix_hash,
            self.state_hash,
            self.verification_authority_hash,
        ):
            if len(value) != 64 or any(
                character not in "0123456789abcdef" for character in value
            ):
                raise ValueError(
                    "cache entry identities must be lowercase SHA-256 digests"
                )


def _canonical_float(value: float) -> str:
    number = float(value)
    if not np.isfinite(number):
        raise ValueError("authority configuration floats must be finite")
    if number == 0.0:
        number = 0.0
    return number.hex()


def _canonical_value(value: Any) -> Any:
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return value
    if isinstance(value, (float, np.floating)):
        return {"float64_hex": _canonical_float(float(value))}
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, dict):
        return {
            str(key): _canonical_value(value[key])
            for key in sorted(value, key=str)
        }
    if isinstance(value, (tuple, list)):
        return [_canonical_value(item) for item in value]
    raise TypeError(f"unsupported authority configuration type: {type(value)!r}")


def _source_digest(module_name: str) -> str:
    module = importlib.import_module(module_name)
    source_path = Path(module.__file__ or "")
    if not source_path.is_file():
        raise RuntimeError(f"cannot fingerprint imported source module {module_name}")
    return sha256(source_path.read_bytes()).hexdigest()


def verification_authority_identity(
    verifier: ExactPlanVerifier,
) -> VerificationAuthorityIdentity:
    """Hash live source bytes and exact numerical configuration, not Git HEAD."""
    evaluator = verifier.evaluator
    simulator = verifier.simulator
    if (
        evaluator.simulator is not simulator
        or evaluator.completion_remaining_pct != verifier.completion_remaining_pct
        or evaluator.hard_margin_mm != verifier.hard_margin_mm
    ):
        raise ValueError("verifier and evaluator exact-authority contracts disagree")
    implementation_modules = {
        type(simulator).__module__,
        type(evaluator).__module__,
        type(verifier).__module__,
    }
    source_hashes = {
        module_name: _source_digest(module_name)
        for module_name in sorted(
            set(_REQUIRED_SOURCE_MODULES) | implementation_modules
        )
    }
    ndimage_binary = importlib.import_module("scipy.ndimage._nd_image")
    binary_path = Path(ndimage_binary.__file__ or "")
    if not binary_path.is_file():
        raise RuntimeError("cannot fingerprint the loaded SciPy ndimage authority")
    manifest = {
        "schema": "laser-ablation/exact-authority/v1",
        "implementation_classes": {
            "simulator": (
                f"{type(simulator).__module__}.{type(simulator).__qualname__}"
            ),
            "evaluator": (
                f"{type(evaluator).__module__}.{type(evaluator).__qualname__}"
            ),
            "verifier": (
                f"{type(verifier).__module__}.{type(verifier).__qualname__}"
            ),
        },
        "physics_config": asdict(simulator.config),
        "physical_action_bounds": asdict(simulator.bounds),
        "completion_remaining_pct": verifier.completion_remaining_pct,
        "hard_margin_mm": verifier.hard_margin_mm,
        "simulator_runtime": {
            "response_noise_std": simulator.response_noise_std,
            "response_scale_bounds": simulator.response_scale_bounds,
            "random_seed": simulator.random_seed,
        },
        "runtime_dependencies": {
            "python_implementation": platform.python_implementation(),
            "python_version": platform.python_version(),
            "numpy_version": np.__version__,
            "scipy_version": scipy.__version__,
            "scipy_ndimage_binary_sha256": sha256(
                binary_path.read_bytes()
            ).hexdigest(),
        },
        "worktree_source_sha256": source_hashes,
    }
    canonical = json.dumps(
        _canonical_value(manifest),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )
    digest = sha256(_AUTHORITY_DOMAIN + canonical.encode("ascii")).hexdigest()
    return VerificationAuthorityIdentity(
        digest=digest,
        manifest_json=canonical,
        cache_reusable=simulator.response_noise_std == 0.0,
    )


def _update_array_digest(
    digest,
    name: str,
    values: np.ndarray,
    *,
    boolean: bool = False,
) -> None:
    digest.update(name.encode("utf-8"))
    array = np.asarray(values)
    digest.update(struct.pack("<Q", array.ndim))
    digest.update(struct.pack("<" + "Q" * array.ndim, *array.shape))
    if boolean:
        if array.dtype != np.bool_:
            raise ValueError(f"{name} must be Boolean")
        canonical = np.ascontiguousarray(array.astype(np.uint8, copy=False))
        digest.update(b"bool-u8\0")
    else:
        canonical = np.asarray(array, dtype=np.float64)
        if not np.all(np.isfinite(canonical)):
            raise ValueError(f"{name} must be finite")
        canonical = canonical.copy()
        canonical[canonical == 0.0] = 0.0
        canonical = np.ascontiguousarray(
            canonical.astype(np.dtype("<f8"), copy=False)
        )
        digest.update(b"little-endian-f64\0")
    digest.update(canonical.tobytes(order="C"))


def _update_scalar_digest(digest, name: str, value: float) -> None:
    number = float(value)
    if not np.isfinite(number):
        raise ValueError(f"{name} must be finite")
    if number == 0.0:
        number = 0.0
    digest.update(name.encode("utf-8"))
    digest.update(b"little-endian-f64\0")
    digest.update(struct.pack("<d", number))


def _update_optional_array_digest(
    digest, name: str, values: np.ndarray | None
) -> None:
    digest.update(name.encode("utf-8"))
    if values is None:
        digest.update(b"absent\0")
        return
    digest.update(b"present\0")
    _update_array_digest(digest, f"{name}:value", values)


def voxel_state_hash(state: VoxelState) -> str:
    """Hash the complete physical state, excluding descriptive provenance."""
    digest = sha256()
    digest.update(_STATE_DOMAIN)
    _update_array_digest(digest, "x_axis_mm", state.x_axis_mm)
    _update_array_digest(digest, "y_axis_mm", state.y_axis_mm)
    _update_array_digest(digest, "z_axis_mm", state.z_axis_mm)
    _update_array_digest(digest, "tissue", state.tissue, boolean=True)
    _update_array_digest(
        digest, "initial_tissue", state.initial_tissue, boolean=True
    )
    _update_array_digest(digest, "target_mask", state.target_mask, boolean=True)
    _update_array_digest(
        digest, "constraint_mask", state.constraint_mask, boolean=True
    )
    _update_scalar_digest(digest, "spacing_mm", state.spacing_mm)
    _update_scalar_digest(digest, "plane_z_mm", state.plane_z_mm)
    digest.update(b"supports_vertical_surface\0")
    digest.update(b"\1" if state.supports_vertical_surface else b"\0")
    _update_optional_array_digest(
        digest, "vertical_surface_z_mm", state.vertical_surface_z_mm
    )
    _update_optional_array_digest(
        digest, "constraint_surface_z_mm", state.constraint_surface_z_mm
    )
    return digest.hexdigest()


def _validate_executed_prefix(
    planned_actions,
    verification: PlanVerification,
    expected_prefix_hash: str,
) -> str | None:
    executed = verification.executed_actions
    if len(executed) > len(planned_actions):
        return "executed_prefix_longer_than_plan"
    planned_prefix = planned_actions[: len(executed)]
    if not physical_action_sequences_equal(executed, planned_prefix):
        return "executed_prefix_actions_mismatch"
    if canonical_action_sequence_hash(planned_prefix) != expected_prefix_hash:
        return "executed_prefix_hash_mismatch"
    if canonical_action_sequence_hash(executed) != expected_prefix_hash:
        return "verification_prefix_hash_mismatch"
    return None


def _make_exact_cache_entry(
    state: VoxelState,
    candidate: PlanCandidate,
    verification: PlanVerification,
    verifier: ExactPlanVerifier,
    *,
    authority: VerificationAuthorityIdentity | None = None,
    state_digest: str | None = None,
) -> ExactVerificationCacheEntry:
    """Internal bridge for an exact evaluation produced at the call site."""
    authority = authority or verification_authority_identity(verifier)
    state_digest = state_digest or voxel_state_hash(state)
    planned_hash = canonical_action_sequence_hash(candidate.actions)
    prefix_hash = canonical_action_sequence_hash(verification.executed_actions)
    mismatch = _validate_executed_prefix(
        candidate.actions, verification, prefix_hash
    )
    if mismatch is not None:
        raise ValueError(f"fresh exact verification is inconsistent: {mismatch}")
    return ExactVerificationCacheEntry(
        verification=verification,
        planned_action_hash=planned_hash,
        executed_prefix_hash=prefix_hash,
        state_hash=state_digest,
        verification_authority_hash=authority.digest,
    )


def _cache_miss_reason(
    candidate: PlanCandidate,
    cached: ExactVerificationCacheEntry | None,
    authority: VerificationAuthorityIdentity,
    state_digest: str,
) -> str | None:
    if not authority.cache_reusable:
        return "nondeterministic_exact_authority"
    if cached is None:
        return "missing_exact_metadata"
    planned_hash = canonical_action_sequence_hash(candidate.actions)
    if cached.planned_action_hash != planned_hash:
        return "planned_action_hash_mismatch"
    prefix_mismatch = _validate_executed_prefix(
        candidate.actions,
        cached.verification,
        cached.executed_prefix_hash,
    )
    if prefix_mismatch is not None:
        return prefix_mismatch
    if cached.state_hash != state_digest:
        return "state_hash_mismatch"
    if cached.verification_authority_hash != authority.digest:
        return "verification_authority_hash_mismatch"
    return None


def _bind_verified_candidate_with_context(
    state: VoxelState,
    candidate: PlanCandidate,
    verifier: ExactPlanVerifier,
    *,
    cached: ExactVerificationCacheEntry | None = None,
    authority: VerificationAuthorityIdentity,
    state_digest: str,
    provenance: tuple[tuple[str, str], ...] = (),
) -> VerifiedGlobalPlan:
    miss_reason = _cache_miss_reason(
        candidate, cached, authority, state_digest
    )
    if miss_reason is None:
        assert cached is not None
        entry = cached
        cache_hit = True
        replay_count = 0
    else:
        verification = verifier.evaluator.evaluate(state, candidate.actions)
        entry = _make_exact_cache_entry(
            state,
            candidate,
            verification,
            verifier,
            authority=authority,
            state_digest=state_digest,
        )
        cache_hit = False
        replay_count = 1
    key = exact_terminal_key(
        entry.verification,
        verifier.completion_remaining_pct,
        verifier.hard_margin_mm,
    )
    return VerifiedGlobalPlan(
        actions=candidate.actions,
        verification=entry.verification,
        exact_terminal_key=key,
        completion_remaining_pct=verifier.completion_remaining_pct,
        hard_margin_mm=verifier.hard_margin_mm,
        planned_action_hash=entry.planned_action_hash,
        executed_prefix_hash=entry.executed_prefix_hash,
        state_hash=entry.state_hash,
        verification_authority_hash=entry.verification_authority_hash,
        candidate_id=f"geometry-aware:{entry.planned_action_hash}",
        layout_id=candidate.layout_id,
        tilt_pattern=candidate.tilt_pattern,
        energy_pattern=candidate.energy_pattern,
        repeat_depth_fraction=candidate.repeat_depth_fraction,
        pulse_fraction=candidate.pulse_fraction,
        order_pattern=candidate.order_pattern,
        provenance=tuple(provenance),
        verification_cache_hit=cache_hit,
        cache_miss_reason=miss_reason,
        exact_replay_count=replay_count,
    )


def bind_verified_candidate(
    state: VoxelState,
    candidate: PlanCandidate,
    verifier: ExactPlanVerifier,
    *,
    cached: ExactVerificationCacheEntry | None = None,
    provenance: tuple[tuple[str, str], ...] = (),
) -> VerifiedGlobalPlan:
    """Bind a candidate under freshly computed state and authority identities.

    State and authority digests are intentionally not caller-overridable.  A
    candidate without trusted exact metadata receives one authoritative replay.
    """
    return _bind_verified_candidate_with_context(
        state,
        candidate,
        verifier,
        cached=cached,
        authority=verification_authority_identity(verifier),
        state_digest=voxel_state_hash(state),
        provenance=provenance,
    )


def select_exact_global_finalist(
    candidates: tuple[VerifiedGlobalPlan, ...],
) -> VerifiedGlobalPlan:
    """Select by exact key alone, with action hash only as a final tie-break."""
    if not candidates:
        raise ValueError("exact Global finalist set cannot be empty")
    return min(
        candidates,
        key=lambda candidate: (
            candidate.exact_terminal_key,
            candidate.planned_action_hash,
        ),
    )


def resolve_exact_downstream_candidate(
    global_reference: VerifiedGlobalPlan,
    downstream_candidate: object | None,
    *,
    state: VoxelState,
    verifier: ExactPlanVerifier,
) -> VerifiedGlobalPlan:
    """Preserve Global unless an exact-bound downstream plan is strictly better."""
    current_state_hash = voxel_state_hash(state)
    current_authority_hash = verification_authority_identity(verifier).digest
    if (
        global_reference.state_hash != current_state_hash
        or global_reference.verification_authority_hash != current_authority_hash
        or global_reference.completion_remaining_pct
        != verifier.completion_remaining_pct
        or global_reference.hard_margin_mm != verifier.hard_margin_mm
    ):
        raise ValueError("VerifiedGlobalPlan is stale for the current exact authority")
    if not isinstance(downstream_candidate, VerifiedGlobalPlan):
        return global_reference
    if (
        downstream_candidate.completion_remaining_pct
        != global_reference.completion_remaining_pct
        or downstream_candidate.hard_margin_mm != global_reference.hard_margin_mm
        or downstream_candidate.state_hash != current_state_hash
        or downstream_candidate.verification_authority_hash
        != current_authority_hash
    ):
        return global_reference
    if (
        downstream_candidate.exact_terminal_key
        < global_reference.exact_terminal_key
    ):
        return downstream_candidate
    return global_reference


__all__ = [
    "VerificationAuthorityIdentity",
    "bind_verified_candidate",
    "resolve_exact_downstream_candidate",
    "select_exact_global_finalist",
    "verification_authority_identity",
    "voxel_state_hash",
]
