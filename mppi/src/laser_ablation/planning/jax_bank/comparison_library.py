"""Persist the routed local-geometry views used by every active state comparison."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path

import numpy as np


def comparison_fingerprint(
    task_fingerprint: str,
    state_shape: tuple[int, int, int],
    step_mask: np.ndarray,
    nominal_energy_j: np.ndarray,
    roi_indices: np.ndarray,
    roi_mask: np.ndarray,
    canonical_seed_ids: tuple[str, ...],
) -> str:
    """Identify the complete immutable routed-ROI comparison authority."""
    digest = sha256(b"routed-response-roi-v1")
    digest.update(task_fingerprint.encode())
    digest.update(str(state_shape).encode())
    for name, values in (
        ("step_mask", step_mask), ("nominal_energy_j", nominal_energy_j),
        ("roi_indices", roi_indices), ("roi_mask", roi_mask),
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
class ComparisonROILibrary:
    """Padded response-ROI routes without dynamics coefficients or duplicate SDF states."""

    step_mask: np.ndarray
    nominal_energy_j: np.ndarray
    roi_indices: np.ndarray
    roi_mask: np.ndarray
    canonical_seed_ids: tuple[str, ...]
    state_shape: tuple[int, int, int]
    task_fingerprint: str
    library_fingerprint: str

    def __post_init__(self) -> None:
        step_mask = np.asarray(self.step_mask, dtype=bool)
        nominal_energy = np.asarray(self.nominal_energy_j, dtype=np.float32)
        indices = np.asarray(self.roi_indices, dtype=np.int32)
        roi_mask = np.asarray(self.roi_mask, dtype=bool)
        shape = tuple(int(value) for value in self.state_shape)
        if step_mask.ndim != 2:
            raise ValueError("comparison step_mask must have shape (seed, step)")
        if nominal_energy.shape != step_mask.shape or not np.all(np.isfinite(nominal_energy)):
            raise ValueError("comparison nominal energy must align with every seed step")
        if indices.ndim != 3 or indices.shape[:2] != step_mask.shape:
            raise ValueError("comparison ROI indices must have shape (seed, step, roi)")
        if roi_mask.shape != indices.shape:
            raise ValueError("comparison ROI mask must align with its indices")
        if len(shape) != 3 or any(value <= 0 for value in shape):
            raise ValueError("comparison state_shape must contain three positive dimensions")
        if len(self.canonical_seed_ids) != step_mask.shape[0]:
            raise ValueError("comparison seed IDs must align with step_mask")
        if len(set(self.canonical_seed_ids)) != len(self.canonical_seed_ids):
            raise ValueError("comparison seed IDs must be unique")
        voxel_count = int(np.prod(shape))
        if np.any(indices[roi_mask] < 0) or np.any(indices[roi_mask] >= voxel_count):
            raise ValueError("comparison ROI indices must address the SDF lattice")
        if np.any(step_mask & ~np.any(roi_mask, axis=-1)):
            raise ValueError("every active comparison route requires a nonempty ROI")
        for values in (step_mask, nominal_energy, indices, roi_mask):
            values.setflags(write=False)
        object.__setattr__(self, "step_mask", step_mask)
        object.__setattr__(self, "nominal_energy_j", nominal_energy)
        object.__setattr__(self, "roi_indices", indices)
        object.__setattr__(self, "roi_mask", roi_mask)
        object.__setattr__(self, "canonical_seed_ids", tuple(self.canonical_seed_ids))
        object.__setattr__(self, "state_shape", shape)

    @property
    def seed_count(self) -> int:
        return int(self.step_mask.shape[0])

    @property
    def maximum_pulses(self) -> int:
        return int(self.step_mask.shape[1])

    @property
    def storage_bytes(self) -> int:
        return int(
            self.step_mask.nbytes + self.nominal_energy_j.nbytes
            + self.roi_indices.nbytes + self.roi_mask.nbytes
        )


def build_comparison_roi_library(
    nominal_states: np.ndarray,
    requested_step_mask: np.ndarray,
    nominal_energy_j: np.ndarray,
    canonical_seed_ids: tuple[str, ...],
    task_fingerprint: str,
) -> ComparisonROILibrary:
    """Build response ROIs from nominal tissue removal and its one-voxel halo."""
    states = np.asarray(nominal_states, dtype=np.float32)
    requested = np.asarray(requested_step_mask, dtype=bool)
    nominal_energy = np.asarray(nominal_energy_j, dtype=np.float32)
    if states.ndim != 5 or states.shape[:2] != (
        requested.shape[0], requested.shape[1] + 1,
    ):
        raise ValueError("nominal states must have shape (seed, step + 1, x, y, z)")
    if nominal_energy.shape != requested.shape or not np.all(np.isfinite(nominal_energy)):
        raise ValueError("nominal energy must have shape (seed, step)")
    if not np.all(np.isfinite(states)):
        raise ValueError("comparison-library nominal states must be finite")
    removed = (states[:, :-1] <= 0.0) & (states[:, 1:] > 0.0) & requested[..., None, None, None]
    flat_removed = removed.reshape((-1,) + states.shape[2:])
    padded = np.pad(flat_removed, ((0, 0), (1, 1), (1, 1), (1, 1)))
    halo = np.zeros_like(flat_removed)
    for x_shift in range(3):
        for y_shift in range(3):
            for z_shift in range(3):
                halo |= padded[
                    :, x_shift:x_shift + states.shape[2],
                    y_shift:y_shift + states.shape[3],
                    z_shift:z_shift + states.shape[4],
                ]
    response = np.any(halo, axis=(1, 2, 3)).reshape(requested.shape)
    step_mask = requested & response
    active = halo.reshape((len(halo), -1))[step_mask.ravel()]
    if not len(active):
        raise ValueError("comparison library requires at least one tissue-removing nominal pulse")
    maximum = int(np.max(np.count_nonzero(active, axis=1)))
    indices = np.zeros(requested.shape + (maximum,), dtype=np.int32)
    roi_mask = np.zeros(requested.shape + (maximum,), dtype=bool)
    for flat_row in np.flatnonzero(step_mask.ravel()):
        seed, step = np.unravel_index(flat_row, requested.shape)
        selected = np.flatnonzero(halo[flat_row].ravel())
        indices[seed, step, :len(selected)] = selected
        roi_mask[seed, step, :len(selected)] = True
    shape = tuple(int(value) for value in states.shape[2:])
    fingerprint = comparison_fingerprint(
        task_fingerprint, shape, step_mask, nominal_energy, indices, roi_mask,
        canonical_seed_ids,
    )
    return ComparisonROILibrary(
        step_mask, nominal_energy, indices, roi_mask, canonical_seed_ids, shape,
        task_fingerprint, fingerprint,
    )


def save_comparison_roi_library(
    directory: Path, library: ComparisonROILibrary,
) -> Path:
    """Persist one lightweight comparison library under SHA-256 coverage."""
    location = Path(directory)
    location.mkdir(parents=True, exist_ok=True)
    archive = location / "comparison_roi_library.npz"
    metadata = location / "comparison_roi_library.json"
    np.savez_compressed(
        archive, step_mask=library.step_mask, nominal_energy_j=library.nominal_energy_j,
        roi_indices=library.roi_indices, roi_mask=library.roi_mask,
    )
    metadata.write_text(json.dumps({
        "canonical_seed_ids": library.canonical_seed_ids,
        "state_shape": library.state_shape,
        "task_fingerprint": library.task_fingerprint,
        "library_fingerprint": library.library_fingerprint,
    }, indent=2, sort_keys=True), encoding="utf-8")
    manifest = {
        path.name: sha256(path.read_bytes()).hexdigest() for path in (archive, metadata)
    }
    (location / "comparison_roi_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    return location


def load_comparison_roi_library(directory: Path) -> ComparisonROILibrary:
    """Load and fingerprint-check a routed ROI comparison library."""
    location = Path(directory)
    manifest = json.loads((location / "comparison_roi_manifest.json").read_text(encoding="utf-8"))
    for filename, expected in manifest.items():
        if sha256((location / filename).read_bytes()).hexdigest() != expected:
            raise ValueError(f"comparison ROI manifest hash mismatch for {filename}")
    metadata = json.loads((location / "comparison_roi_library.json").read_text(encoding="utf-8"))
    with np.load(location / "comparison_roi_library.npz", allow_pickle=False) as archive:
        library = ComparisonROILibrary(
            archive["step_mask"], archive["nominal_energy_j"],
            archive["roi_indices"], archive["roi_mask"],
            tuple(metadata["canonical_seed_ids"]), tuple(metadata["state_shape"]),
            metadata["task_fingerprint"], metadata["library_fingerprint"],
        )
    expected = comparison_fingerprint(
        library.task_fingerprint, library.state_shape, library.step_mask,
        library.nominal_energy_j,
        library.roi_indices, library.roi_mask, library.canonical_seed_ids,
    )
    if expected != library.library_fingerprint:
        raise ValueError("comparison ROI library fingerprint mismatch")
    return library


__all__ = [
    "ComparisonROILibrary", "build_comparison_roi_library",
    "load_comparison_roi_library", "save_comparison_roi_library",
]
