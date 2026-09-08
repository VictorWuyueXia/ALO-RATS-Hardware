"""Compiled JAX sampling, rollout, cost, and segment-beam execution."""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from laser_ablation.planning.jax_bank.contracts import QuickRolloutBatch, StaticTaskTensors
from laser_ablation.planning.jax_bank.linear_contracts import DeviceLinearizationWorkspace, LinearizationLibrary
from laser_ablation.planning.jax_bank.mppi_cost import (
    path_integral_weights,
    tail_costs,
    update_nominal,
)
from laser_ablation.planning.jax_bank.mppi_sampling import sample_correlated_actions
from laser_ablation.planning.jax_bank.repair import (
    FeasibleRepairTrajectory,
    MPPIRepairConfig,
)
from laser_ablation.planning.jax_bank.terminal_rollout import rollout_terminal_in_batches
from laser_ablation.planning.jax_bank.segment_beam import (
    SegmentBeam,
    stream_linearized_segment as _stream_linearized_segment,
)
from laser_ablation.planning.jax_bank.workspace_rollout import (
    WORKSPACE_FLAG_NAMES,
    build_workspace_executor,
    workspace_array_tuple,
)


@dataclass(frozen=True)
class WorkspaceRolloutSummary:
    """Terminal device-screening values transferred without a full SDF trace."""

    remaining_fraction: np.ndarray
    overcut_fraction: np.ndarray
    healthy_overcut_fraction: np.ndarray
    clearance_mm: np.ndarray
    pulse_count: np.ndarray
    energy_j: np.ndarray
    flags: dict[str, np.ndarray]
    constraint_feasible: np.ndarray
    accepted: np.ndarray
    mppi_cost: np.ndarray
    cost_components: np.ndarray


class MPPIExecutor:
    """Keep compiled array primitives reusable across every repair iteration."""

    def __init__(self, config: MPPIRepairConfig, devices: tuple[object, ...]) -> None:
        import jax

        self.config = config
        if not devices:
            raise ValueError("MPPI execution requires an explicit JAX device set")
        self.devices = devices
        self.sample_compiled = jax.jit(
            sample_correlated_actions,
            static_argnames=("samples_per_anchor", "beta"),
        )
        self.cost_compiled = jax.jit(tail_costs)
        self.weight_compiled = jax.jit(
            path_integral_weights, static_argnames=("temperature",)
        )
        self.update_compiled = jax.jit(update_nominal)
        self.workspace: DeviceLinearizationWorkspace | None = None
        self._workspace_arrays: tuple[object, ...] | None = None
        self._workspace_executor: object | None = None
        self._workspace_summary_executor: object | None = None
        self._workspace_signature: tuple[object, ...] | None = None
        self._workspace_roi_capacity: int | None = None
        self.workspace_kernel_creations = 0
        self.workspace_kernel_families = {"full_trace": 0, "terminal_summary": 0}
        self.workspace_summary_rows_evaluated = 0
        self.workspace_full_trace_rows_fetched = 0
        self.workspace_summary_prewarm_count = 0
        self.last_workspace_cosine: np.ndarray | None = None
        self.last_workspace_contact_displacement: np.ndarray | None = None
        self.last_workspace_contact_has_mismatch: np.ndarray | None = None
        self.last_workspace_contact_entry_mismatch: np.ndarray | None = None
        self.last_workspace_contact_interval_mismatch: np.ndarray | None = None
        self.last_workspace_contact_guard_failure: np.ndarray | None = None

    @property
    def workspace_roi_capacity(self) -> int | None:
        """Expose the first installed fixed ROI width to the next repair build."""
        return self._workspace_roi_capacity

    def install_workspace(
        self, task: StaticTaskTensors, workspace: DeviceLinearizationWorkspace,
    ) -> DeviceLinearizationWorkspace:
        """Place one repair-local workspace on devices and retain one shape executor."""
        import gc
        import jax

        if workspace.task_fingerprint != task.fingerprint:
            raise ValueError("installed workspace and task fingerprints disagree")
        if self._workspace_roi_capacity is None:
            self._workspace_roi_capacity = workspace.roi_capacity
        elif workspace.roi_capacity != self._workspace_roi_capacity:
            raise ValueError("fresh_roi_capacity")
        if self.workspace is not None:
            self.workspace = None
            self._workspace_arrays = None
            gc.collect()
        if len(self.devices) == 1:
            placed = tuple(jax.device_put(values, self.devices[0]) for values in workspace_array_tuple(workspace))
        else:
            from jax.sharding import Mesh, NamedSharding, PartitionSpec

            sharding = NamedSharding(Mesh(np.asarray(self.devices), ("plan",)), PartitionSpec())
            placed = tuple(jax.device_put(values, sharding) for values in workspace_array_tuple(workspace))
        device_workspace = replace(
            workspace,
            **{
                name: values for name, values in zip(
                    (
                        "base_nominal_actions", "base_step_mask",
                        "base_roi_indices", "base_roi_mask", "base_state_gain",
                        "base_action_jacobian", "base_action_support_lower",
                        "base_action_support_upper", "center_states", "center_actions",
                        "center_action_mask", "anchor_routes", "anchor_valid", "refresh_mask",
                        "overlay_roi_indices", "overlay_roi_mask", "overlay_state_gain",
                        "overlay_action_jacobian", "overlay_action_support_lower",
                        "overlay_action_support_upper",
                    ),
                    placed,
                    strict=True,
                )
            },
        )
        signature = (
            task.fingerprint, device_workspace.base_seed_capacity,
            device_workspace.anchor_capacity, device_workspace.maximum_pulses,
            device_workspace.roi_capacity, device_workspace.rollout_rows, len(self.devices),
        )
        if self._workspace_executor is None or self._workspace_signature != signature:
            self._workspace_executor = build_workspace_executor(
                task, device_workspace, self.devices, trace=True,
            )
            self._workspace_summary_executor = build_workspace_executor(
                task, device_workspace, self.devices, trace=False,
            )
            self._workspace_signature = signature
            self.workspace_kernel_creations += 1
            self.workspace_kernel_families["full_trace"] += 1
            self.workspace_kernel_families["terminal_summary"] += 1
        self.workspace = device_workspace
        self._workspace_arrays = workspace_array_tuple(device_workspace)
        return device_workspace

    def prewarm_workspace(self, task: StaticTaskTensors) -> None:
        """Compile the terminal-screening path without fetching an SDF trace."""
        import jax
        import jax.numpy as jnp

        if self.workspace is None:
            raise RuntimeError("a recentered workspace must be installed before prewarm")
        workspace = self.workspace
        actions = np.zeros((1, workspace.maximum_pulses, 5), dtype=np.float32)
        actions[:, 0] = 0.5 * (task.lower_bounds + task.upper_bounds)
        mask = np.zeros((1, workspace.maximum_pulses), dtype=bool)
        mask[:, 0] = True
        path = np.full((1, workspace.maximum_pulses, 2), -1, dtype=np.int32)
        initial = np.broadcast_to(task.initial_current_sdf, (1,) + task.shape).copy()
        before = self.workspace_summary_rows_evaluated
        summary = self.rollout_workspace_summary(
            actions, actions.copy(), mask, path, np.asarray((-1,), dtype=np.int32), initial, task,
        )
        jax.block_until_ready(jnp.asarray(summary.mppi_cost))
        self.workspace_summary_rows_evaluated = before
        self.workspace_summary_prewarm_count += 1

    def _workspace_inputs(
        self,
        actions: np.ndarray,
        action_mask: np.ndarray,
        linearization_path: np.ndarray,
        anchor_slots: np.ndarray,
        initial_current_sdf: np.ndarray,
    ) -> tuple[int, tuple[object, object, object, object, object]]:
        """Pad one host microbatch once for either scalar or full-trace execution."""
        import jax.numpy as jnp

        if self.workspace is None or self._workspace_arrays is None:
            raise RuntimeError("a recentered workspace must be installed before segment rollout")
        count = int(actions.shape[0])
        workspace = self.workspace
        if (
            actions.shape != (count, workspace.maximum_pulses, 5)
            or action_mask.shape != (count, workspace.maximum_pulses)
            or linearization_path.shape != (count, workspace.maximum_pulses, 2)
            or anchor_slots.shape != (count,)
            or initial_current_sdf.shape != (count,) + workspace.state_shape
            or count <= 0 or count > workspace.rollout_rows
        ):
            raise ValueError("workspace rollout inputs must fit its fixed row, pulse, and state shapes")
        padded = workspace.rollout_rows
        packed_actions = np.zeros((padded, workspace.maximum_pulses, 5), np.float32)
        packed_mask = np.zeros((padded, workspace.maximum_pulses), bool)
        packed_path = np.full((padded, workspace.maximum_pulses, 2), -1, np.int32)
        packed_slots = np.full(padded, -1, np.int32)
        packed_initial = np.zeros((padded,) + workspace.state_shape, np.float32)
        packed_actions[:count] = actions
        packed_mask[:count] = action_mask
        packed_path[:count] = linearization_path
        packed_slots[:count] = anchor_slots
        packed_initial[:count] = initial_current_sdf
        return count, (
            jnp.asarray(packed_initial), jnp.asarray(packed_actions), jnp.asarray(packed_mask),
            jnp.asarray(packed_path), jnp.asarray(packed_slots),
        )

    def rollout_workspace(
        self,
        actions: np.ndarray,
        action_mask: np.ndarray,
        linearization_path: np.ndarray,
        anchor_slots: np.ndarray,
        initial_current_sdf: np.ndarray,
        trust_threshold: float | None = None,
    ) -> QuickRolloutBatch:
        """Fetch complete traces; ``-1`` explicitly disables the full trust screen for diagnostics."""
        import jax.numpy as jnp
        from laser_ablation.planning.jax_bank.rollout import screening_mask

        if self._workspace_executor is None or self._workspace_arrays is None or self.workspace is None:
            raise RuntimeError("a recentered workspace must be installed before segment rollout")
        workspace = self.workspace
        count, packed = self._workspace_inputs(
            actions, action_mask, linearization_path, anchor_slots, initial_current_sdf,
        )
        threshold = (
            workspace.linearization_cosine_min
            if trust_threshold is None else float(trust_threshold)
        )
        if not -1.0 <= threshold <= 1.0:
            raise ValueError("workspace trust threshold must lie in [-1, 1]")
        values = self._workspace_executor(
            *packed, self._workspace_arrays,
            jnp.asarray(threshold, dtype=jnp.float32),
        )
        (
            states, remaining_voxels, overcut_voxels, remaining, overcut,
            healthy_overcut, clearance, pulse_count, energy, flags, cosine,
            contact_displacement, contact_has_mismatch, contact_entry_mismatch,
            contact_interval_mismatch, contact_guard_failure,
        ) = values
        self.last_workspace_cosine = np.asarray(cosine[:count])
        self.last_workspace_contact_displacement = np.asarray(contact_displacement[:count])
        self.last_workspace_contact_has_mismatch = np.asarray(contact_has_mismatch[:count])
        self.last_workspace_contact_entry_mismatch = np.asarray(contact_entry_mismatch[:count])
        self.last_workspace_contact_interval_mismatch = np.asarray(contact_interval_mismatch[:count])
        self.last_workspace_contact_guard_failure = np.asarray(contact_guard_failure[:count])
        self.workspace_full_trace_rows_fetched += count
        flags = np.asarray(flags[:count])
        arrays = [np.asarray(values[:count]) for values in (
            states, remaining_voxels, overcut_voxels, remaining, overcut,
            healthy_overcut, clearance, pulse_count, energy,
        )]
        flag_arrays = {name: flags[..., index] for index, name in enumerate(WORKSPACE_FLAG_NAMES)}
        feasible = ~np.any(flags, axis=(1, 2))
        return QuickRolloutBatch(
            current_sdf=arrays[0], remaining_voxels=arrays[1], overcut_voxels=arrays[2],
            remaining_fraction=arrays[3], overcut_fraction=arrays[4],
            healthy_overcut_fraction=arrays[5], clearance_mm=arrays[6],
            pulse_count=arrays[7], energy_j=arrays[8], flags=flag_arrays,
            constraint_feasible=feasible,
            accepted=screening_mask(arrays[3][:, -1], arrays[4][:, -1], ~feasible),
        )

    def rollout_workspace_summary(
        self,
        actions: np.ndarray,
        origin_actions: np.ndarray,
        action_mask: np.ndarray,
        linearization_path: np.ndarray,
        anchor_slots: np.ndarray,
        initial_current_sdf: np.ndarray,
        task: StaticTaskTensors,
    ) -> WorkspaceRolloutSummary:
        """Score one padded microbatch on device and return terminal scalars only."""
        import jax.numpy as jnp

        if self._workspace_summary_executor is None or self._workspace_arrays is None or self.workspace is None:
            raise RuntimeError("a recentered workspace must be installed before segment rollout")
        if task.fingerprint != self.workspace.task_fingerprint:
            raise ValueError("workspace summary task and installed workspace disagree")
        if origin_actions.shape != actions.shape:
            raise ValueError("workspace origins must align with sampled actions")
        count, packed = self._workspace_inputs(
            actions, action_mask, linearization_path, anchor_slots, initial_current_sdf,
        )
        workspace = self.workspace
        packed_origins = np.zeros((workspace.rollout_rows, workspace.maximum_pulses, 5), np.float32)
        packed_origins[:count] = origin_actions
        (
            remaining, overcut, healthy_overcut, clearance, pulse_count, energy, flag_values,
        ) = self._workspace_summary_executor(
            *packed, self._workspace_arrays,
            jnp.asarray(workspace.linearization_cosine_min, dtype=jnp.float32),
        )
        feasible = ~jnp.any(flag_values, axis=1)
        accepted = feasible
        costs, components = self.cost_compiled(
            packed[1][:, None], jnp.asarray(packed_origins), packed[2], remaining[:, None],
            healthy_overcut[:, None], jnp.asarray(task.lower_bounds), jnp.asarray(task.upper_bounds),
            (self.config.lambda_remaining, self.config.lambda_overcut, self.config.lambda_anchor),
        )
        self.workspace_summary_rows_evaluated += count
        arrays = [np.asarray(values[:count]) for values in (
            remaining, overcut, healthy_overcut, clearance, pulse_count, energy,
        )]
        flags = np.asarray(flag_values[:count])
        return WorkspaceRolloutSummary(
            remaining_fraction=arrays[0], overcut_fraction=arrays[1],
            healthy_overcut_fraction=arrays[2], clearance_mm=arrays[3],
            pulse_count=arrays[4], energy_j=arrays[5],
            flags={name: flags[:, index] for index, name in enumerate(WORKSPACE_FLAG_NAMES)},
            constraint_feasible=np.asarray(feasible[:count]), accepted=np.asarray(accepted[:count]),
            mppi_cost=np.asarray(costs[:count, 0]),
            cost_components=np.asarray(components[:count, 0]),
        )

    def sample(
        self,
        seed: int,
        iteration: int,
        tail_ids: tuple[str, ...],
        active_indices: np.ndarray,
        nominal: np.ndarray,
        mask: np.ndarray,
        perturb_mask: np.ndarray,
        task: StaticTaskTensors,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Derive order-independent anchor keys and sample correlated actions."""
        import jax
        import jax.numpy as jnp

        root = jax.random.fold_in(jax.random.PRNGKey(seed), iteration)
        keys = np.stack([
            np.asarray(jax.random.fold_in(
                root, int(tail_ids[index][:8], 16) & 0x7FFFFFFF
            ))
            for index in active_indices
        ])
        sampled, perturbations = self.sample_compiled(
            jnp.asarray(keys), jnp.asarray(nominal), jnp.asarray(mask),
            jnp.asarray(perturb_mask), jnp.asarray(task.lower_bounds),
            jnp.asarray(task.upper_bounds), samples_per_anchor=self.config.samples_per_anchor,
            beta=self.config.beta, kappa=jnp.asarray(self.config.kappa),
        )
        return np.asarray(sampled), np.asarray(perturbations)

    def screen(
        self,
        samples: np.ndarray,
        mask: np.ndarray,
        tail_ids: tuple[str, ...],
        active_indices: np.ndarray,
        task: StaticTaskTensors,
        iteration: int,
    ):
        """Terminal-screen flattened sampled rows without retaining SDF traces."""
        anchors, sample_count, pulse_count, _ = samples.shape
        flat_mask = np.broadcast_to(mask[:, None], (anchors, sample_count, pulse_count))
        del tail_ids, active_indices, iteration
        return rollout_terminal_in_batches(
            task, task.initial_current_sdf, samples.reshape(-1, pulse_count, 5),
            flat_mask.reshape(-1, pulse_count), self.config.rollout_batch_size,
            self.devices,
        )

    def cost_and_weights(
        self,
        samples: np.ndarray,
        origin: np.ndarray,
        mask: np.ndarray,
        rollout: object,
        task: StaticTaskTensors,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute paper costs and masked path-integral weights."""
        import jax.numpy as jnp

        anchors, samples_per_anchor = samples.shape[:2]
        remaining = rollout.remaining_fraction.reshape(anchors, samples_per_anchor)
        overcut = rollout.healthy_overcut_fraction.reshape(anchors, samples_per_anchor)
        costs, components = self.cost_compiled(
            jnp.asarray(samples), jnp.asarray(origin), jnp.asarray(mask),
            jnp.asarray(remaining), jnp.asarray(overcut),
            jnp.asarray(task.lower_bounds), jnp.asarray(task.upper_bounds),
            (self.config.lambda_remaining, self.config.lambda_overcut, self.config.lambda_anchor),
        )
        weights = self.weight_compiled(
            costs,
            jnp.asarray(rollout.constraint_feasible.reshape(anchors, samples_per_anchor)),
            temperature=self.config.temperature,
        )
        return np.asarray(costs), np.asarray(components), np.asarray(weights)

    def update(
        self,
        nominal: np.ndarray,
        perturbations: np.ndarray,
        weights: np.ndarray,
        mask: np.ndarray,
        task: StaticTaskTensors,
    ) -> np.ndarray:
        """Update optimizer centers while keeping them outside candidate selection."""
        import jax.numpy as jnp

        updated = self.update_compiled(
            jnp.asarray(nominal), jnp.asarray(perturbations), jnp.asarray(weights),
            jnp.asarray(mask), jnp.asarray(task.lower_bounds), jnp.asarray(task.upper_bounds),
        )
        return np.asarray(updated)

    def stream_linearized_segment(
        self,
        seed: int,
        stage: int,
        beam: SegmentBeam,
        starts: np.ndarray,
        stops: np.ndarray,
        task: StaticTaskTensors,
        library: LinearizationLibrary,
        initial_current_sdf: np.ndarray,
        capacity_records: int,
        microbatch_size: int,
    ) -> tuple[SegmentBeam | None, tuple[FeasibleRepairTrajectory, ...], dict[str, object]]:
        """Delegate one bounded local-linear segment branch to its fixed-record engine."""
        return _stream_linearized_segment(
            self, seed, stage, beam, starts, stops, task, library, initial_current_sdf,
            capacity_records, microbatch_size,
        )
