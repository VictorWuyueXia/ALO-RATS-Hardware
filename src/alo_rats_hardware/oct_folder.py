"""Deterministic import of Lumedica B-scan folders from the mounted Windows share."""

from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
import json
from pathlib import Path
import re
from time import monotonic, sleep, time

import cv2
import numpy as np

from .scan_adapter import lattice_axes
from .site import SiteConfiguration
from .surface_scan import SurfaceScan, VolumetricScan, rigid_transform, save_scan


class OCTFolderAdapter:
    """Wait for one new stable scan folder and reconstruct a registered height-field occupancy."""

    def __init__(self, site: SiteConfiguration, records):
        self.site, self.records = site, records
        self.root = Path(site.oct["shared_root"])
        if not self.root.is_dir():
            raise FileNotFoundError(f"Mounted OCT shared root does not exist: {self.root}")
        self.known_directories = {path.name for path in self.root.iterdir() if path.is_dir()}
        self.sequence = 0

    def acquire(self, command_id, scan_pose_base_m):
        """Accept one completed new folder, then convert its surface to the controller lattice."""
        scan_pose = rigid_transform(scan_pose_base_m)
        values = self.site.oct
        expected = int(values["expected_b_scans"])
        stable_path = stable_signature = None
        stable_count = 0
        started = monotonic()
        while monotonic() - started <= values["scan_timeout_s"]:
            new_directories = sorted(path for path in self.root.iterdir()
                                     if path.is_dir() and path.name not in self.known_directories)
            complete = []
            for directory in new_directories:
                files = tuple(directory.glob(values["file_pattern"]))
                if len(files) == expected:
                    complete.append((directory, tuple(sorted(
                        (path.name, path.stat().st_size, path.stat().st_mtime_ns) for path in files))))
            if len(complete) > 1:
                raise RuntimeError("Multiple new complete OCT folders are ambiguous")
            if complete:
                directory, signature = complete[0]
                if directory == stable_path and signature == stable_signature:
                    stable_count += 1
                else:
                    stable_path, stable_signature, stable_count = directory, signature, 1
                if stable_count >= values["stable_observations"]:
                    break
            print(f"Waiting for one stable new OCT folder under {self.root}; "
                  f"elapsed {monotonic() - started:.1f} s", flush=True)
            sleep(values["poll_interval_s"])
        else:
            raise TimeoutError("No complete stable OCT folder appeared before scan_timeout_s")

        indexed = []
        for path in stable_path.glob(values["file_pattern"]):
            numbers = re.findall(r"\d+", path.stem)
            if not numbers:
                raise ValueError(f"OCT B-scan filename has no numeric index: {path.name}")
            indexed.append((int(numbers[-1]), path))
        indexed.sort()
        indices = [index for index, _ in indexed]
        if len(set(indices)) != expected or indices != list(range(indices[0], indices[0] + expected)):
            raise ValueError("OCT B-scan indices must be unique and contiguous")
        files = [path for _, path in indexed]

        with ThreadPoolExecutor() as pool:
            images = list(pool.map(lambda path: cv2.imread(str(path), cv2.IMREAD_GRAYSCALE), files))
        expected_shape = tuple(values["image_shape_px"])
        if any(image is None or image.shape != expected_shape for image in images):
            raise ValueError(f"Every OCT B-scan must be a readable grayscale image of shape {expected_shape}")
        with ThreadPoolExecutor() as pool:
            filtered = list(pool.map(lambda image: cv2.bilateralFilter(image, 8, 300, 150), images))
        margin = int(values["surface_margin_px"])
        cropped = np.stack([image[margin:-margin] for image in filtered])
        surface_rows = np.argmax(cropped, axis=1) + margin
        peaks = np.take_along_axis(cropped, (surface_rows - margin)[:, None, :], axis=1)[:, 0, :]
        if np.any(peaks < values["surface_threshold_u8"]):
            raise ValueError("OCT surface extraction found columns below surface_threshold_u8")

        scan_count, width = surface_rows.shape
        lateral_spacing, scan_spacing, depth_spacing = values[
            "pixel_spacing_lateral_scan_depth_mm"]
        lateral = (np.arange(width) - (width - 1) / 2) * lateral_spacing
        scan_axis = (np.arange(scan_count) - (scan_count - 1) / 2) * scan_spacing
        source = np.column_stack((
            np.tile(lateral, scan_count),
            np.repeat(scan_axis, width),
            -surface_rows.reshape(-1) * depth_spacing,
        ))
        oct_points = source[:, values["axis_order"]] * values["axis_signs"]
        planning_from_oct = (np.linalg.inv(self.site.registration["base_from_planning_m"])
                             @ scan_pose @ self.site.registration["tcp_from_oct_m"])
        planning_points = (oct_points / 1000 @ planning_from_oct[:3, :3].T
                           + planning_from_oct[:3, 3]) * 1000

        accepted_at = time()
        hashes = []
        digest = sha256()
        for path in files:
            file_hash = sha256(path.read_bytes()).hexdigest()
            hashes.append({"name": path.name, "bytes": path.stat().st_size, "sha256": file_hash})
            digest.update(path.name.encode())
            digest.update(bytes.fromhex(file_hash))
        scan_id = digest.hexdigest()
        bounds = values["planning_volume_bounds_mm"]
        axes = lattice_axes(bounds)
        surface_scan = SurfaceScan(
            planning_points, np.ones(len(planning_points), dtype=bool), scan_id, accepted_at,
            values["planning_frame_id"], np.eye(4), scan_pose, float(bounds[2, 0]),
            f"Lumedica shared folder {stable_path}", "mm", "complete_height_field",
        )
        surface = surface_scan.surface_on(*axes[:2])
        tissue = axes[2][None, None, :] <= surface[:, :, None] + 1e-8
        if np.any(tissue.sum(axis=2) == 0):
            raise ValueError("Registered OCT surface leaves an empty planning column")
        scan = VolumetricScan(
            *axes, tissue, np.ones(tissue.shape, dtype=bool), scan_id, accepted_at,
            values["planning_frame_id"], scan_pose,
            f"segmented height field from Lumedica shared folder {stable_path}",
            "mm", "segmented_occupancy_volume",
        )
        manifest = {
            "scan_id": scan_id, "accepted_at_s": accepted_at, "command_id": command_id,
            "source_directory": str(stable_path), "file_pattern": values["file_pattern"],
            "files": hashes, "scan_pose_base_m": scan_pose,
            "planning_from_oct_m": planning_from_oct,
            "surface_threshold_u8": values["surface_threshold_u8"],
        }
        name = f"prefix_{self.sequence:03d}"
        save_scan(scan, self.records.output / "scans" / f"{name}.npz")
        with (self.records.output / "scans" / f"{name}_manifest.json").open(
                "x", encoding="utf-8") as stream:
            json.dump({key: value.tolist() if isinstance(value, np.ndarray) else value
                       for key, value in manifest.items()}, stream, indent=2)
            stream.write("\n")
        self.records.event("oct_scan", sequence=self.sequence, scan_id=scan_id,
                           command_id=command_id, source_directory=stable_path,
                           occupied_voxels=int(tissue.sum()))
        self.known_directories.add(stable_path.name)
        self.sequence += 1
        return scan
