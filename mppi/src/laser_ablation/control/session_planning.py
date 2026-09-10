"""Preserved global-plan and periodic-repair ordering for a controller session."""

import csv
from time import perf_counter

from laser_ablation.planning.global_3d.interface import GlobalPlanningFailure
from laser_ablation.planning.jax_bank.repair import GlobalReplanRequired, RepairRequest, RepairTrigger


PERIODIC_REPAIR_EVENT_COLUMNS = (
    "event_id", "successful_pulses", "trigger", "attempted", "repair_time_s",
    "repair_succeeded", "periodic_global_replan", "repair_artifact_directory",
    "result_trajectory_id", "status",
)


def propose(session, index):
    """A failed global search terminates without releasing the previous plan."""
    try:
        plan = session.planner.propose(
            session.state, session.observed, session.raster_generator,
            session.banks_directory / f"global_{index:03d}",
        )
    except GlobalPlanningFailure:
        session.stopped_reason = "GLOBAL_PLANNING_FAILED"
        return False
    session.plan = plan
    return True


def select_action(session):
    """Stop an exhausted sequence before processing the coincident periodic event."""
    if session.prefix >= len(session.plan.actions):
        # The deployment uses the four initial raster parents throughout; do not
        # invoke the geometry-generated global-plan branch after sequence exhaustion.
        session.stopped_reason = "ACTION_SEQUENCE_EXHAUSTED"
        return None
    due = session.confirmed_pulses > 0 and session.confirmed_pulses % session.periodic_repair_pulses == 0
    session.cycle_record = {
        "periodic_repair_due": due, "periodic_repair_attempted": False,
        "periodic_repair_time_s": 0.0, "periodic_repair_succeeded": False,
        "periodic_global_replan": False, "periodic_repair_event_id": None,
    }
    if due and not repair(session):
        return None
    return session.plan.actions[session.prefix]


def repair(session):
    """Attempt one periodic repair without invoking another global plan."""
    session.repairs += 1
    request = RepairRequest(
        session.plan.trajectory_id, session.prefix, RepairTrigger.PERIODIC, None,
        int(next(session.repair_seeds)), session.repairs_directory / f"repair_{session.repairs:03d}",
    )
    started = perf_counter()
    replanned = failed = False
    try:
        session.plan = session.planner.repair(session.state, session.observed, request)
    except GlobalReplanRequired:
        # A failed local repair terminates this run; the expensive geometry-generated
        # global-plan branch remains unused after the initial four-raster plan.
        failed = True
        session.stopped_reason = "REPAIR_REQUIRES_GLOBAL_REPLAN"
    elapsed = perf_counter() - started
    event = {
        "event_id": len(session.repair_events) + 1,
        "successful_pulses": session.confirmed_pulses,
        "trigger": RepairTrigger.PERIODIC.value, "attempted": True,
        "repair_time_s": elapsed, "repair_succeeded": not replanned and not failed,
        "periodic_global_replan": replanned,
        "repair_artifact_directory": str(request.artifact_directory),
        "result_trajectory_id": None if failed else session.plan.trajectory_id,
        "status": "global_replan_skipped" if failed else "repaired",
    }
    session.repair_events.append(event)
    with session.event_path.open("a", newline="", encoding="utf-8") as stream:
        csv.DictWriter(stream, fieldnames=PERIODIC_REPAIR_EVENT_COLUMNS).writerow(event)
    session.cycle_record.update({
        "periodic_repair_attempted": True, "periodic_repair_time_s": elapsed,
        "periodic_repair_succeeded": event["repair_succeeded"],
        "periodic_global_replan": replanned, "periodic_repair_event_id": event["event_id"],
    })
    if failed:
        return False
    session.replans += int(replanned)
    session.selected_ids.append(session.plan.trajectory_id)
    session.prefix = 0
    return True
