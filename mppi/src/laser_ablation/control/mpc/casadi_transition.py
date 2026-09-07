"""Beam-aligned CasADi SDF transition for fixed-pose energy refinement."""

from __future__ import annotations

from dataclasses import dataclass

import casadi as ca
import numpy as np

from laser_ablation.control.mpc.beam_geometry import (
    BeamMPCROI,
    beam_contact_bracket,
    maximum_active_radius_mm,
)
from laser_ablation.core.actions import PhysicalAction
from laser_ablation.core.sdf import SDFState
from laser_ablation.core.state import VoxelState
from laser_ablation.metrics import evaluate_ablation
from laser_ablation.physics.super_gaussian import PhysicsConfig, laser_axis


@dataclass(frozen=True)
class SymbolicBeamRollout:
    """Single-shooting expressions generated from fixed poses and variable energies."""

    phi_by_step: tuple[ca.MX, ...]
    contact_distance_mm: tuple[ca.MX, ...]
    contact_outside_phi_mm: tuple[ca.MX, ...]
    contact_inside_phi_mm: tuple[ca.MX, ...]
    new_clearance_mm: tuple[ca.MX, ...]
    cumulative_clearance_mm: tuple[ca.MX, ...]
    workspace_margin_mm: tuple[ca.MX, ...]


class CasadiBeamTransition:
    """Construct local CSG rollout expressions for arbitrary fixed beam directions."""

    def __init__(
        self,
        current_voxel_state: VoxelState,
        current_sdf_state: SDFState,
        nominal_actions: tuple[PhysicalAction, ...],
        reference_tissue_sdf_mm: tuple[np.ndarray, ...],
        physics: PhysicsConfig,
        roi: BeamMPCROI,
        *,
        inclusion_epsilon_mm: float,
        clearance_samples_per_axis: int,
    ) -> None:
        if len(reference_tissue_sdf_mm) != len(nominal_actions):
            raise ValueError("one pre-action tissue SDF is required per horizon action")
        if inclusion_epsilon_mm <= 0.0 or clearance_samples_per_axis < 3:
            raise ValueError("transition discretization configuration is invalid")
        self.voxel_state = current_voxel_state
        self.sdf_state = current_sdf_state
        self.actions = nominal_actions
        self.references = tuple(np.asarray(value, dtype=float) for value in reference_tissue_sdf_mm)
        self.physics = physics
        self.roi = roi
        self.inclusion_epsilon_mm = float(inclusion_epsilon_mm)
        self.clearance_samples_per_axis = int(clearance_samples_per_axis)
        self.brackets = tuple(
            beam_contact_bracket(reference, current_voxel_state, action)
            for reference, action in zip(self.references, nominal_actions, strict=True)
        )
        self._tissue_interpolant = self._interpolant(
            "mpc_tissue", current_sdf_state.observation.tissue_sdf_mm
        )
        self._clearance_interpolant = self._interpolant(
            "mpc_clearance", current_sdf_state.observation.physical_clearance_mm
        )

    def _interpolant(self, prefix: str, values: np.ndarray) -> ca.Function:
        observation = self.sdf_state.observation
        suffix = self.roi.hash_sha256[:12] + str(id(self))[-6:]
        return ca.interpolant(
            prefix + suffix,
            "linear",
            [
                observation.x_axis_mm.astype(float),
                observation.y_axis_mm.astype(float),
                observation.z_axis_mm.astype(float),
            ],
            np.asarray(values, dtype=float).ravel(order="F"),
        )

    def rollout(self, energies_j: ca.MX) -> SymbolicBeamRollout:
        if energies_j.shape != (len(self.actions), 1):
            raise ValueError("energy vector must have shape (H_eff, 1)")
        initial_phi = self.sdf_state.observation.tissue_sdf_mm[tuple(self.roi.indices.T)]
        phi_roi = ca.DM(initial_phi.astype(float).reshape(-1, 1))
        phi_steps: list[ca.MX] = []
        contacts: list[ca.MX] = []
        distances: list[ca.MX] = []
        outside_values: list[ca.MX] = []
        inside_values: list[ca.MX] = []
        new_clearances: list[ca.MX] = []
        cumulative_clearances: list[ca.MX] = []
        workspace_margins: list[ca.MX] = []
        current_clearance = evaluate_ablation(self.voxel_state).minimum_clearance_mm
        if not np.isfinite(current_clearance):
            current_clearance = float(np.ptp(self.voxel_state.z_axis_mm) + self.voxel_state.spacing_mm)
        cumulative = ca.DM(current_clearance)

        for step, (action, bracket) in enumerate(zip(self.actions, self.brackets, strict=True)):
            points = ca.DM(np.column_stack((bracket.outside_point_mm, bracket.inside_point_mm)))
            bracket_phi = self._phi_after_prior_craters(points, contacts, energies_j, step)
            phi_outside, phi_inside = bracket_phi[0], bracket_phi[1]
            fraction = -phi_outside / (phi_inside - phi_outside)
            contact = ca.DM(bracket.outside_point_mm) + fraction * ca.DM(
                bracket.inside_point_mm - bracket.outside_point_mm
            )
            distance = bracket.outside_distance_mm + fraction * (
                bracket.inside_distance_mm - bracket.outside_distance_mm
            )
            contacts.append(contact)
            distances.append(distance)
            outside_values.append(phi_outside)
            inside_values.append(phi_inside)
            psi = self._crater_level_set(ca.DM(self.roi.points_mm.T), action, contact, energies_j[step])
            phi_roi = ca.fmax(phi_roi, -psi + self.inclusion_epsilon_mm)
            phi_steps.append(phi_roi)
            clearance, workspace = self._new_crater_clearance(action, contact, energies_j[step])
            cumulative = ca.fmin(cumulative, clearance)
            new_clearances.append(clearance)
            cumulative_clearances.append(cumulative)
            workspace_margins.append(workspace)
        return SymbolicBeamRollout(
            tuple(phi_steps), tuple(distances), tuple(outside_values), tuple(inside_values),
            tuple(new_clearances), tuple(cumulative_clearances), tuple(workspace_margins),
        )

    def _phi_after_prior_craters(
        self, points: ca.MX, contacts: list[ca.MX], energies_j: ca.MX, count: int
    ) -> ca.MX:
        phi = self._tissue_interpolant(points).T
        for index in range(count):
            psi = self._crater_level_set(points, self.actions[index], contacts[index], energies_j[index])
            phi = ca.fmax(phi, -psi + self.inclusion_epsilon_mm)
        return phi

    def _crater_level_set(
        self, points: ca.MX, action: PhysicalAction, contact_mm: ca.MX, energy_j: ca.MX
    ) -> ca.MX:
        beam = ca.DM(laser_axis(action.tilt_x_rad, action.tilt_y_rad))
        offsets = points - ca.repmat(contact_mm, 1, points.shape[1])
        axial = ca.mtimes(beam.T, offsets)
        radial_squared = ca.fmax(ca.sum1(offsets * offsets) - axial * axial, 0.0)
        profile = ca.exp(-ca.power(
            radial_squared / (2.0 * self.physics.spot_size_mm**2),
            self.physics.super_gaussian_power,
        ))
        raw_depth = self.physics.depth_scale_mm_per_j * (
            energy_j * profile - self.physics.ablation_threshold
        )
        lower_face = -axial - 0.5 * self.voxel_state.spacing_mm
        return ca.fmax(ca.fmax(lower_face, axial - raw_depth), -raw_depth).T

    def _new_crater_clearance(
        self, action: PhysicalAction, contact_mm: ca.MX, energy_j: ca.MX
    ) -> tuple[ca.MX, ca.MX]:
        beam = laser_axis(action.tilt_x_rad, action.tilt_y_rad)
        reference = np.array((1.0, 0.0, 0.0)) if abs(beam[0]) < 0.9 else np.array((0.0, 1.0, 0.0))
        transverse_u = np.cross(beam, reference)
        transverse_u /= np.linalg.norm(transverse_u)
        transverse_v = np.cross(beam, transverse_u)
        transverse = np.linspace(
            -maximum_active_radius_mm(self.physics),
            maximum_active_radius_mm(self.physics),
            self.clearance_samples_per_axis,
        )
        uu, vv = np.meshgrid(transverse, transverse, indexing="ij")
        radial_squared = uu.ravel() ** 2 + vv.ravel() ** 2
        active = radial_squared <= maximum_active_radius_mm(self.physics) ** 2
        uu = uu.ravel()[active]
        vv = vv.ravel()[active]
        radial_squared = radial_squared[active]
        profile = np.exp(-(
            radial_squared / (2.0 * self.physics.spot_size_mm**2)
        ) ** self.physics.super_gaussian_power)
        depth = self.physics.depth_scale_mm_per_j * ca.fmax(
            energy_j * ca.DM(profile) - self.physics.ablation_threshold, 0.0
        )
        transverse_points = (
            uu[:, None] * transverse_u[None]
            + vv[:, None] * transverse_v[None]
        ).T
        points = ca.repmat(contact_mm, 1, depth.shape[0]) + ca.DM(transverse_points) + ca.DM(beam) @ depth.T
        observation = self.sdf_state.observation
        half_spacing = 0.5 * self.voxel_state.spacing_mm
        lower = ca.DM([
            observation.x_axis_mm[0] - half_spacing,
            observation.y_axis_mm[0] - half_spacing,
            observation.z_axis_mm[0] - half_spacing,
        ])
        upper = ca.DM([
            observation.x_axis_mm[-1] + half_spacing,
            observation.y_axis_mm[-1] + half_spacing,
            observation.z_axis_mm[-1] + half_spacing,
        ])
        workspace = ca.mmin(ca.vertcat(
            points - ca.repmat(lower, 1, points.shape[1]),
            ca.repmat(upper, 1, points.shape[1]) - points,
        ))
        clearance = self._clearance_interpolant(points)
        if self.voxel_state.constraint_surface_z_mm is not None:
            clearance = ca.fmax(clearance, -float(self.voxel_state.spacing_mm))
        return ca.mmin(clearance), workspace
