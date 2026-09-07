# Experiment runbook interpretation summary

The operator can run the nominal simulation and the processed-OCT UR5e dry run from the current checkout. The OCT-geometry simulation is only preparable, and physical phantom resection is not executable because the hardware closed loop is absent.

The most important operational distinction is that a processed, registered OCT `.npz` file is the current software boundary. Raw OCT acquisition, segmentation, and registration remain external. A successful dry run proves only that the task was designated, the MPPI method identity was resolved, and RTDE returned robot state; it does not prove calibrated targeting or cutting readiness.

For a collaborator handoff, first preserve a machine audit, then recover historical OCT and laser components in isolation. Qualify them through the runbook's ladder rather than placing unknown modules directly on the active Python path. The full workflow must fail closed after any uncertain pulse or missing/stale rescan and must preserve one execution receipt and one fresh OCT observation for each confirmed physical pulse.

The delivered ALO-RATS controller is the sole algorithmic authority. It requires a fresh registered observation after every confirmed pulse and schedules plan repair every ten confirmed pulses. `see-plan-cut` is used only to locate the original hardware arrangement, calibration assets, device interfaces, and familiar operator interaction points.

Primary operator document: [`EXPERIMENT_OPERATOR_RUNBOOK.md`](EXPERIMENT_OPERATOR_RUNBOOK.md).
