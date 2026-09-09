# Interpretation summary

The runbook now separates repository preparation from certified laser operation. The repository maintainer completes software, OCT records, registration, and robot state/safe-pose work with the laser and Raspberry Pi unpowered. The certified operator then performs the familiar physical safety actions, records five live laser values, and runs one preselected command after remote configuration validation.

The present boundaries are the live laser interface and two missing laser-free qualification commands. Raspberry Pi stopped-state semantics, an independent cutoff result, and a measured duty-to-energy table do not exist in this checkout. The mounted-folder adapter has no OCT-only entry point, and the hardware dry run cannot execute an inert MPPI treatment pose and return. Physical execution remains intentionally disabled until these paths and values are qualified. A completely unattended repository-blind operator handoff is therefore unsupported.

Human instructions and conclusions belong under each handoff or run's `human_readables/` directory. Raw device replies, OCT and power measurements, configuration identities, motion records, and terminal status belong under `machine_readables/`. Acceptance comes from `machine_readables/workflow_status.json` and its supporting records, never from console output or screenshots.

`0-OPERATOR_RUNBOOK.md` contains the short four-phase procedure. `OPERATOR_HANDOFF_REFERENCE.md` contains the exact device schemas, minimal manual config fields, and code-readiness map.

The current physical coordinator accepts scientific experiments 2 and 3. Scientific experiments 1 and 4 in `Experiment_Plan.md` have no physical entry point. All physical results remain prospective until a certified run produces complete evidence.
