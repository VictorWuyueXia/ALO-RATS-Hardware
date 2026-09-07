"""Immutable local-linearization data shared by future segment rollouts."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path

import numpy as np

from laser_ablation.planning.jax_bank.contracts import ACTION_DIMENSION


def linearization_fingerprint(
    task_fingerprint: str,
    nominal_actions: np.ndarray,
    step_mask: np.ndarray,
    nominal_states: np.ndarray,
    roi_indices: np.ndarray,
    roi_mask: np.ndarray,
    state_gain: np.ndarray,
    action_jacobian: np.ndarray,
    action_support_lower: np.ndarray,
    action_support_upper: np.ndarray,
    canonical_seed_ids: tuple[str, ...],
) -> str:
    """Identify every immutable coefficient closed over by a local executor."""
    digest = sha256()
    digest.update(b"local-affine-sdf-v1")
    digest.update(task_fingerprint.encode())
    for name, values in (
        ("nominal_actions", nominal_actions), ("step_mask", step_mask),
        ("nominal_states", nominal_states), ("roi_indices", roi_indices),
        ("roi_mask", roi_mask), ("state_gain", state_gain),
        ("action_jacobian", action_jacobian),
        ("action_support_lower", action_support_lower),
        ("action_support_upper", action_support_upper),
    ):
        array = np.ascontiguousarray(values)
        digest.update(name.encode())
        digest.update(str(array.shape).encode())
        digest.update(str(array.dtype).encode())
        digest.update(array.tobytes())
    for identifier in canonical_seed_ids:
        digest.update(identifier.encode())
    return digest.hexdigest()


@dataclass(frozen=True)
class LinearizationLibrary:
    """Padded sparse affine dynamics around unique nominal action sequences."""

    nominal_actions: np.ndarray
    step_mask: np.ndarray
    nominal_states: np.ndarray
    nominal_descriptors: np.ndarray
    roi_indices: np.ndarray
    roi_mask: np.ndarray
    state_gain: np.ndarray
    action_jacobian: np.ndarray
    action_support_lower: np.ndarray
    action_support_upper: np.ndarray
    canonical_seed_ids: tuple[str, ...]
    task_fingerprint: str
    library_fingerprint: str

    def __post_init__(self) -> None:
        actions = np.asarray(self.nominal_actions, dtype=np.float32)
        mask = np.asarray(self.step_mask, dtype=bool)
        states = np.asarray(self.nominal_states, dtype=np.float32)
        descriptors = np.asarray(self.nominal_descriptors, dtype=np.float32)
        indices = np.asarray(self.roi_indices, dtype=np.int32)
        roi_mask = np.asarray(self.roi_mask, dtype=bool)
        gain = np.asarray(self.state_gain, dtype=np.float32)
        jacobian = np.asarray(self.action_jacobian, dtype=np.float32)
        lower = np.asarray(self.action_support_lower, dtype=np.float32)
        upper = np.asarray(self.action_support_upper, dtype=np.float32)
        if actions.ndim != 3 or actions.shape[-1] != ACTION_DIMENSION:
            raise ValueError("nominal_actions must have shape (seed, step, 5)")
        seeds, steps, _ = actions.shape
        if mask.shape != (seeds, steps):
            raise ValueError("step_mask must have shape (seed, step)")
        if states.ndim != 5 or states.shape[:2] != (seeds, steps + 1):
            raise ValueError("nominal_states must have shape (seed, step + 1, x, y, z)")
        if descriptors.ndim != 5 or descriptors.shape[:2] != (seeds, steps + 1):
            raise ValueError("nominal_descriptors must align with nominal_states")
        if indices.ndim != 3 or indices.shape[:2] != (seeds, steps):
            raise ValueError("roi_indices must have shape (seed, step, roi)")
        if roi_mask.shape != indices.shape or gain.shape != indices.shape:
            raise ValueError("ROI masks and state gains must align with roi_indices")
        if jacobian.shape != indices.shape + (ACTION_DIMENSION,):
            raise ValueError("action_jacobian must have shape (seed, step, roi, 5)")
        if lower.shape != actions.shape or upper.shape != actions.shape:
            raise ValueError("action support bounds must align with nominal_actions")
        if len(self.canonical_seed_ids) != seeds or len(set(self.canonical_seed_ids)) != seeds:
            raise ValueError("canonical_seed_ids must contain one unique value per seed")
        voxel_count = int(np.prod(states.shape[2:]))
        if np.any(indices[roi_mask] < 0) or np.any(indices[roi_mask] >= voxel_count):
            raise ValueError("ROI indices must address nominal state voxels")
        if np.any(lower > upper):
            raise ValueError("action support lower bounds must not exceed upper bounds")
        if not all(np.all(np.isfinite(values)) for values in (
            actions, states, descriptors, gain, jacobian, lower, upper,
        )):
            raise ValueError("linearization library arrays must be finite")
        for name, values in (
            ("nominal_actions", actions), ("step_mask", mask),
            ("nominal_states", states), ("nominal_descriptors", descriptors),
            ("roi_indices", indices), ("roi_mask", roi_mask),
            ("state_gain", gain), ("action_jacobian", jacobian),
            ("action_support_lower", lower), ("action_support_upper", upper),
        ):
            values.setflags(write=False)
            object.__setattr__(self, name, values)
        object.__setattr__(self, "canonical_seed_ids", tuple(self.canonical_seed_ids))

    @property
    def seed_count(self) -> int:
        return int(self.nominal_actions.shape[0])

    @property
    def maximum_pulses(self) -> int:
        return int(self.nominal_actions.shape[1])

    @property
    def state_shape(self) -> tuple[int, int, int]:
        return tuple(int(value) for value in self.nominal_states.shape[2:])

    @property
    def maximum_roi_voxels(self) -> int:
        return int(self.roi_indices.shape[-1])

    @property
    def storage_bytes(self) -> int:
        return int(sum(values.nbytes for values in (
            self.nominal_actions, self.step_mask, self.nominal_states,
            self.nominal_descriptors, self.roi_indices, self.roi_mask,
            self.state_gain, self.action_jacobian, self.action_support_lower,
            self.action_support_upper,
        )))

    def per_device_storage_bytes(self, device_count: int) -> int:
        """Return the conservative replicated-library allocation for one device."""
        if device_count <= 0:
            raise ValueError("device_count must be positive")
        return self.storage_bytes


@dataclass(frozen=True)
class LinearizedActionBatch:
    """Padded actions with a nominal library transition selected for every pulse."""

    actions: np.ndarray
    action_mask: np.ndarray
    linearization_path: np.ndarray
    initial_current_sdf: np.ndarray
    source_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        actions = np.asarray(self.actions, dtype=np.float32)
        mask = np.asarray(self.action_mask, dtype=bool)
        path = np.asarray(self.linearization_path, dtype=np.int32)
        initial = np.asarray(self.initial_current_sdf, dtype=np.float32)
        if actions.ndim != 3 or actions.shape[-1] != ACTION_DIMENSION:
            raise ValueError("actions must have shape (batch, step, 5)")
        if mask.shape != actions.shape[:2]:
            raise ValueError("action_mask must have shape (batch, step)")
        if path.shape != actions.shape[:2] + (2,):
            raise ValueError("linearization_path must have shape (batch, step, 2)")
        if initial.ndim != 4 or initial.shape[0] != actions.shape[0]:
            raise ValueError("initial_current_sdf must have shape (batch, x, y, z)")
        if len(self.source_ids) != actions.shape[0]:
            raise ValueError("source_ids must contain one value per batch row")
        if np.any(path[mask] < 0):
            raise ValueError("active linearization paths must be nonnegative")
        if np.any(path[~mask] != -1):
            raise ValueError("masked linearization paths must equal [-1, -1]")
        if not np.all(np.isfinite(actions)) or not np.all(np.isfinite(initial)):
            raise ValueError("linearized action batches require finite arrays")
        for values in (actions, mask, path, initial):
            values.setflags(write=False)
        object.__setattr__(self, "actions", actions)
        object.__setattr__(self, "action_mask", mask)
        object.__setattr__(self, "linearization_path", path)
        object.__setattr__(self, "initial_current_sdf", initial)
        object.__setattr__(self, "source_ids", tuple(self.source_ids))

    @property
    def maximum_pulses(self) -> int:
        return int(self.actions.shape[1])

    @property
    def storage_bytes(self) -> int:
        return int(sum(values.nbytes for values in (
            self.actions, self.action_mask, self.linearization_path,
            self.initial_current_sdf,
        )))


@dataclass(frozen=True)
class DeviceLinearizationWorkspace:
    """One repair-local, fixed-shape affine workspace centred on live SDF tails.

    The persisted :class:`LinearizationLibrary` remains the canonical source of
    base coefficients.  This object only carries its padded device copy, the
    four observed-state centre scans, and sparse overlays for centre/nominal
    pairs that leave the base trust region.  ``anchor_routes`` stay canonical;
    ``refresh_mask`` selects an anchor-local overlay without changing them.
    """

    base_nominal_actions: object
    base_step_mask: object
    base_nominal_states: object
    base_roi_indices: object
    base_roi_mask: object
    base_state_gain: object
    base_action_jacobian: object
    base_action_support_lower: object
    base_action_support_upper: object
    center_states: object
    center_actions: object
    center_action_mask: object
    anchor_routes: object
    anchor_valid: object
    refresh_mask: object
    overlay_roi_indices: object
    overlay_roi_mask: object
    overlay_state_gain: object
    overlay_action_jacobian: object
    overlay_action_support_lower: object
    overlay_action_support_upper: object
    task_fingerprint: str
    library_fingerprint: str
    linearization_cosine_min: float
    base_seed_count: int
    anchor_count: int
    rollout_rows: int
    phase_seconds: tuple[tuple[str, float], ...] = ()

    def __post_init__(self) -> None:
        arrays = {
            name: getattr(self, name)
            for name in (
                "base_nominal_actions", "base_step_mask", "base_nominal_states",
                "base_roi_indices", "base_roi_mask", "base_state_gain",
                "base_action_jacobian", "base_action_support_lower",
                "base_action_support_upper", "center_states", "center_actions",
                "center_action_mask", "anchor_routes", "anchor_valid", "refresh_mask",
                "overlay_roi_indices", "overlay_roi_mask", "overlay_state_gain",
                "overlay_action_jacobian", "overlay_action_support_lower",
                "overlay_action_support_upper",
            )
        }
        actions = arrays["base_nominal_actions"]
        if actions.ndim != 3 or actions.shape[-1] != ACTION_DIMENSION:
            raise ValueError("workspace base actions must have shape (seed, pulse, 5)")
        seeds, pulses, _ = actions.shape
        states = arrays["base_nominal_states"]
        if states.ndim != 5 or states.shape[:2] != (seeds, pulses + 1):
            raise ValueError("workspace base states must align with base actions")
        state_shape = states.shape[2:]
        roi_indices = arrays["base_roi_indices"]
        if roi_indices.ndim != 3 or roi_indices.shape[:2] != (seeds, pulses):
            raise ValueError("workspace base ROI indices must align with base actions")
        roi = roi_indices.shape[-1]
        if arrays["base_roi_mask"].shape != roi_indices.shape:
            raise ValueError("workspace base ROI mask must align with indices")
        if arrays["base_state_gain"].shape != roi_indices.shape:
            raise ValueError("workspace base state gain must align with indices")
        if arrays["base_action_jacobian"].shape != roi_indices.shape + (ACTION_DIMENSION,):
            raise ValueError("workspace base action Jacobian must have shape (seed, pulse, roi, 5)")
        for name in ("base_action_support_lower", "base_action_support_upper"):
            if arrays[name].shape != actions.shape:
                raise ValueError("workspace base action support must align with actions")
        centers = arrays["center_states"]
        if centers.ndim != 5 or centers.shape[1:] != (pulses + 1,) + state_shape:
            raise ValueError("workspace centre states must have shape (anchor, pulse + 1, x, y, z)")
        anchors = centers.shape[0]
        if arrays["center_actions"].shape != (anchors, pulses, ACTION_DIMENSION):
            raise ValueError("workspace centre actions must align with centre states")
        for name in ("center_action_mask", "anchor_routes", "refresh_mask"):
            expected = (anchors, pulses) + ((2,) if name == "anchor_routes" else ())
            if arrays[name].shape != expected:
                raise ValueError(f"workspace {name} must align with anchor pulse rows")
        if arrays["anchor_valid"].shape != (anchors,):
            raise ValueError("workspace anchor validity must contain one value per anchor")
        for name in ("overlay_roi_indices", "overlay_roi_mask", "overlay_state_gain"):
            if arrays[name].shape != (anchors, pulses, roi):
                raise ValueError(f"workspace {name} must align with the fixed ROI capacity")
        if arrays["overlay_action_jacobian"].shape != (anchors, pulses, roi, ACTION_DIMENSION):
            raise ValueError("workspace overlay Jacobian must align with the fixed ROI capacity")
        for name in ("overlay_action_support_lower", "overlay_action_support_upper"):
            if arrays[name].shape != (anchors, pulses, ACTION_DIMENSION):
                raise ValueError("workspace overlay action support must align with centre actions")
        if not 0 < self.base_seed_count <= seeds or not 0 < self.anchor_count <= anchors:
            raise ValueError("workspace active seed and anchor counts must fit their padded capacities")
        if self.rollout_rows <= 0:
            raise ValueError("workspace rollout row capacity must be positive")
        phases = tuple((str(name), float(seconds)) for name, seconds in self.phase_seconds)
        if any(not name or not np.isfinite(seconds) or seconds < 0.0 for name, seconds in phases):
            raise ValueError("workspace phase timings must be finite nonnegative named values")
        host_arrays = all(isinstance(values, np.ndarray) for values in arrays.values())
        active = arrays["center_action_mask"]
        routes = arrays["anchor_routes"]
        if host_arrays and (np.any(routes[active] < 0) or np.any(routes[~active] != -1)):
            raise ValueError("workspace active routes must be canonical and masked routes must be [-1, -1]")
        if host_arrays and np.any(arrays["refresh_mask"] & ~active):
            raise ValueError("workspace overlays may exist only at active centre steps")
        if host_arrays and np.any(active & ~arrays["anchor_valid"][:, None]):
            raise ValueError("inactive workspace anchor slots may not carry actions")
        if not self.task_fingerprint or not self.library_fingerprint:
            raise ValueError("workspace fingerprints must be nonempty")
        if not -1.0 <= self.linearization_cosine_min <= 1.0:
            raise ValueError("workspace trust cosine must lie in [-1, 1]")
        finite = (
            "base_nominal_actions", "base_nominal_states", "base_state_gain",
            "base_action_jacobian", "base_action_support_lower", "base_action_support_upper",
            "center_states", "center_actions", "overlay_state_gain",
            "overlay_action_jacobian", "overlay_action_support_lower",
            "overlay_action_support_upper",
        )
        if host_arrays and not all(np.all(np.isfinite(arrays[name])) for name in finite):
            raise ValueError("workspace floating arrays must be finite")
        object.__setattr__(self, "phase_seconds", phases)

    @property
    def base_seed_capacity(self) -> int:
        return int(self.base_nominal_actions.shape[0])

    @property
    def anchor_capacity(self) -> int:
        return int(self.center_states.shape[0])

    @property
    def maximum_pulses(self) -> int:
        return int(self.base_nominal_actions.shape[1])

    @property
    def roi_capacity(self) -> int:
        return int(self.base_roi_indices.shape[-1])

    @property
    def state_shape(self) -> tuple[int, int, int]:
        return tuple(int(value) for value in self.base_nominal_states.shape[2:])

    @property
    def array_bytes(self) -> dict[str, int]:
        """Return exact allocations for every device-resident workspace array."""
        return {
            name: int(getattr(self, name).nbytes)
            for name in (
                "base_nominal_actions", "base_step_mask",
                "base_roi_indices", "base_roi_mask", "base_state_gain",
                "base_action_jacobian", "base_action_support_lower",
                "base_action_support_upper", "center_states", "center_actions",
                "center_action_mask", "anchor_routes", "anchor_valid", "refresh_mask",
                "overlay_roi_indices", "overlay_roi_mask", "overlay_state_gain",
                "overlay_action_jacobian", "overlay_action_support_lower",
                "overlay_action_support_upper",
            )
        }

    @property
    def host_array_bytes(self) -> dict[str, int]:
        """Report the one persisted-state mirror deliberately not copied to devices."""
        return {"base_nominal_states": int(self.base_nominal_states.nbytes)}

    @property
    def storage_bytes(self) -> int:
        return int(sum(self.array_bytes.values()))

    @property
    def build_phase_seconds(self) -> dict[str, float]:
        """Expose completed host-side workspace phases for evidence artifacts."""
        return dict(self.phase_seconds)

    def per_device_storage_bytes(self, device_count: int) -> int:
        if device_count <= 0:
            raise ValueError("device_count must be positive")
        return self.storage_bytes


def save_linearization_library(directory: Path, library: LinearizationLibrary) -> Path:
    """Persist one library with its own SHA-256 manifest beside a plan bank."""
    location = Path(directory)
    location.mkdir(parents=True, exist_ok=True)
    archive_path = location / "linearization_library.npz"
    metadata_path = location / "linearization_library.json"
    np.savez_compressed(
        archive_path,
        nominal_actions=library.nominal_actions, step_mask=library.step_mask,
        nominal_states=library.nominal_states, nominal_descriptors=library.nominal_descriptors,
        roi_indices=library.roi_indices, roi_mask=library.roi_mask,
        state_gain=library.state_gain, action_jacobian=library.action_jacobian,
        action_support_lower=library.action_support_lower,
        action_support_upper=library.action_support_upper,
    )
    metadata_path.write_text(json.dumps({
        "task_fingerprint": library.task_fingerprint,
        "library_fingerprint": library.library_fingerprint,
        "canonical_seed_ids": library.canonical_seed_ids,
    }, indent=2, sort_keys=True), encoding="utf-8")
    manifest = {
        path.name: sha256(path.read_bytes()).hexdigest()
        for path in (archive_path, metadata_path)
    }
    (location / "linearization_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    return location


def load_linearization_library(directory: Path) -> LinearizationLibrary:
    """Load one manifest-checked local-linearization library."""
    location = Path(directory)
    manifest = json.loads((location / "linearization_manifest.json").read_text(encoding="utf-8"))
    for filename, expected in manifest.items():
        if sha256((location / filename).read_bytes()).hexdigest() != expected:
            raise ValueError(f"linearization manifest hash mismatch for {filename}")
    metadata = json.loads((location / "linearization_library.json").read_text(encoding="utf-8"))
    archive = np.load(location / "linearization_library.npz", allow_pickle=False)
    library = LinearizationLibrary(
        nominal_actions=archive["nominal_actions"], step_mask=archive["step_mask"],
        nominal_states=archive["nominal_states"],
        nominal_descriptors=archive["nominal_descriptors"],
        roi_indices=archive["roi_indices"], roi_mask=archive["roi_mask"],
        state_gain=archive["state_gain"], action_jacobian=archive["action_jacobian"],
        action_support_lower=archive["action_support_lower"],
        action_support_upper=archive["action_support_upper"],
        canonical_seed_ids=tuple(metadata["canonical_seed_ids"]),
        task_fingerprint=metadata["task_fingerprint"],
        library_fingerprint=metadata["library_fingerprint"],
    )
    archive.close()
    expected = linearization_fingerprint(
        library.task_fingerprint, library.nominal_actions, library.step_mask,
        library.nominal_states, library.roi_indices, library.roi_mask,
        library.state_gain, library.action_jacobian, library.action_support_lower,
        library.action_support_upper, library.canonical_seed_ids,
    )
    if expected != library.library_fingerprint:
        raise ValueError("linearization library fingerprint mismatch")
    return library
