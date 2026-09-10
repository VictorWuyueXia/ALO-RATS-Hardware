# Project Goal

This is a clean, history-free runtime workspace for integrating the ALO-RATS MPPI planner-controller with the collaborator's UR5e/PyBullet support model. Its purpose is to establish a reproducible route from a registered OCT observation and user-designated target to unchanged MPPI planning, UR5e-compatible motion geometry, and eventually reobservation/replanning.

The primary engineering objective is the quickest, simplest deployment of the existing planner on the robot/OCT/laser setup. Planner completion and outcome quality are experiment results outside the integration acceptance criterion.

It contains the unchanged MPPI source and method configuration, the supplied UR5e URDF/meshes, processed-OCT volume interchange, mouse-driven target designation, PyBullet simulation, and a robot-motion-only hardware dry run. It contains no Nd:YAG/PWM/laser-device code, no laser firing path, no raw OCT driver, no results, and no old Git history.

Read the current situation and transfer handoff before preparing a new computer. In particular, the OCT scanner trigger is not present in the collaborator checkout, and the package currently accepts a processed, registered OCT volume rather than raw scanner data.

Simulation validation and physical deployment must load the same tracked MPPI controller, physics, planner, compute, random-seed, and experiment raster configuration. Do not introduce reduced diagnostic cases or alternate deployment-only planner values. Hardware-specific identities, calibration records, registered OCT observations, and operator-designated targets remain measured experiment inputs.


# Our coding style rules

- Adhere strictly to our coding style descipline, realizing goals with simplest possbile method, write your logic in compact streamlined line-of-logic files, avoid short wrapper/helper functions, avoid unescesary CLI/configs, avoid fallback values or behaviors, avoid try/with/except. Include this rule in your plan.

- Eplicitly write all your planned class/objects, functions/methods, independent variables, data classes, wrappers/helpers, and parameters in you plan. Each with their specific role and task. You should have a maximum of a handful of each, devided my task oriented workflow or task independent standard operations, and minimize the presence of wrappers/helpers, and parameters. You will not be allowed to exceed your planned structure budget.

- We want to strictly adhere to this rule. We hate thin wrapper functions with too less logic or mega functions with too much logic, files that are too long (>300lines) or too short (<40lines), over abstractions, stand alone parameters/variables/functions/methods that are only called once by others, or poorly organized code logics (in file or class that is not close enough to what the code chunk actually does). The designated file length should be achieved with proper code organization and logic grouping, not by removing blank-line seperations or comments.

# dev rules
- Always prefer parallel computing for parallel tasks. For some tasks that seem to need for loops or sequence logics at first glance, think if you can parallel compute it by pre-printing parameters or preparing resources in advance for each iteraition.

- If you encounter long-wait-no-update script runtimes, stop the run, add update messeages in the scripts, and rerun

- When delievering artifacts, seperate the scope between human-reading oriented reports and visuals, and machine-reading (code scripts or other agents) oriented data. Do not mix them together: Keep the human-reading artifacts straight foward, easy-to-understand, shallow organized and easily accessible, keep labels, entries, and legends fully defined and explained but short. Add a interpretation_summary.md written by you to briefly and quickly explain how to interpret the artifact and your judement/conclusion; Make machine-reading oriented data detailed and organized in subfolders, ready to be analyzed or used to recreate results. For example, when saving runtime/training histories or confusion matrices, keep a csv in machine-scannables and a png in human-readables.


- Always treat it as expensive when adding code or adding objectes. Always prefer removing code than adding when modifying. Alwasy realize the code base with shortest total code count for each file, this means avoid over-abstraction or over-generalization logics, and seperate the code chunks or methods by tasks instead of by roles. For example, a good script my contain 1-3 methods, each corresponding to a running mode or task or goal, do not seperat them to many small methods of small operation steps.

- When training models or making controllers with tunable parameters, proactively train or test out the agent in several iterations. Between iterations, scan through the artifacts and identify any potential improvements that could be achieced by tuning the parameters before the next iteration. Do so until you believe tuning parameters alone cannot significantly improve anymore, and if the agent still fails at this stage, infer a reasong why and suggest a next step fix.

# doc rules
- When writing docs, if your answer include mathematical expressions/formulas/variables, fully define them when they emerge, explain how to get the values (defined, measured, computed, heuristic, tunable, etc.), and give an intuitive interpretations for each of them. Output in .md files with "$$" latex math syntax. Split into human readable parts which is more illustrative and intuitive with major governing formulas, and machine schanning parts with as much rigorous details as you can write. 

- When formulating a plan, explicitly identify stages and milestones of the dev process, give specific validation procedures for each milestone and their success flags. And always rememver our projects are meant to be show-of-concept, not to deliver a clinic/industry-ready product. So never over engineering the method, but proactively keep the mathematicall elegance (streamline, efficiency, naturally working, etc.) and strigency of our methods.

# How to speak
- Strictly avoid vague or ambiguous vovabularies anywhere. Terms must be carefully defined of its meaning and aligned with concept-terminology pairs. Avoid casual wordings such as "authoritive" "candidate" "gate" or similar. For example, say "the trajectories that satisfy the constraint ..." instead of "the surviving candidates are gated by...". Only state what happend or what is being done, never say "we do smth only to ..." or " smth is not ..." or "smth is ... instead of ...". The standards should be any new reader who never knows what is this project should be able to understand your writing with out ambiguity.

- when explaining a concept that is a step of a workflow or pipeline or algorithm, the terminalogy associated with that step should be fixed through out the project, and vocabulary vibrancy is heavily discouraged. The standard is to have one unique word pinpointing to one exact thing in the algorithm. And instead mentioning only the steps we are focusing on, you need to include the current workflow's steps before and after the focus, including what objects (algorithm level conception objects, not programming class object) are involved and what operations are downe amoung the steps. Human cannot understand one single step taken out of a complex algorithm.


# venv
We are on a ubuntu remote server where conda or sudo is not available, we need to build or see the already built .venv acoording to the encironment or requirement files. But keep and keep updating the encironment or requirement files so we have them available when switching to other machines.

# How to use GPU
We are on a 8-GPU servers now which means we want to use the maximum parallel computing ability of them with our code to lighten the time cost of our computation. But you cannot access GPU with your sandbox terminal directly, and sudo is not available on this remote server. So the way is to submit the training/run command through the terminal tool with sandbox_permissions="require_escalated". After my approval, the command executed outside the sandbox on the same Ubuntu host, where CUDA and the GPUs were visible.
For example, the earlier eight-GPU run used:
```
.venv/bin/python scripts/run_full_capability.py \
  --config configs/controller_smoke.yaml \
  --compute 8gpu \
  --output-dir /tmp/laser_ablation_smoke_20260825 \
  --seed 20260825
```
The tool invocation explicitly requested approval with the reason that JAX needed access to the server GPUs. I then monitored the returned process session incrementally.
