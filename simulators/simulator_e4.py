#!/usr/bin/env python3
"""
simulator_e4.py

Synthetic event log designed to isolate E4 (avg_delay_in_window).

DESIGN RATIONALE
----------------
The base simulator has a single load channel: every delay is caused by resource
contention, and contention is a direct function of the number of concurrently
active cases. E1 (open_cases_at_time) measures exactly that quantity, so E1
dominates any other encoding by construction.

This variant breaks the coupling with two independent hidden regimes:

  1. SPEED regime  (fast / slow) -> multiplies all activity durations by W.
  2. ARRIVAL regime (low / high) -> controls the mean inter-arrival time 1/lambda.

Both switch according to their own Markov process; they are statistically
independent of each other and of the system state.

The resource pool is deliberately oversized so that queueing is negligible.
Therefore:

  remaining time      ~ W           (only the speed regime matters)
  E1 (open cases)     ~ lambda * W  (Little's Law: confounded by arrivals)
  E3 (arrivals)       ~ lambda      (uninformative for remaining time)
  E4 (avg delay)      ~ W           (measures the speed regime directly)

E1 cannot separate W from lambda and becomes a noisy proxy. E4 measures W
exactly. This is a construct-validity check, not evidence of practical utility.

Removed relative to the base simulator:
  - batch spawning in _instance_spawned (that mechanism belongs to simulator_e6)
  - Jane/Joe resource coupling (intra-case noise, would blur the E4 signal)
  - the commented-out E2/E5 load factors
"""

from enum import Enum
import numpy as np
import random
import pandas as pd
from pm4py.objects.conversion.log import converter as log_converter
import pm4py
from datetime import datetime, timedelta


# ----------------------------------------------------------------------------
# Parameters
# ----------------------------------------------------------------------------

SEED = 42

# Speed regime. CRITICAL: the mean sojourn times must be much longer than the
# window used by E4 (avg_delay_in_window, 168 h in the pipeline). If the regime
# switches faster than the window, E4 averages across several regimes and the
# signal cancels itself out. 800-1500 h keeps a whole window inside one regime.
SPEED_FACTOR_FAST = 1.0
SPEED_FACTOR_SLOW = 3.0
SPEED_MEAN_FAST_H = 1500.0
SPEED_MEAN_SLOW_H = 800.0

# Arrival regime. Independent of the speed regime.
ARRIVAL_SCALE_LOW_H = 20.0   # mean inter-arrival time in the low regime
ARRIVAL_SCALE_HIGH_H = 3.0   # mean inter-arrival time in the high regime
ARRIVAL_MEAN_LOW_H = 80.0
ARRIVAL_MEAN_HIGH_H = 80.0

# Oversized pool: peak load is lambda * W = (1/3) * 25 h ~ 8.3 busy resources.
# 25 resources keep the queueing probability negligible.
N_RESOURCES = 25

SIM_DURATION_H = 24 * 365 * 10 #neu


# ----------------------------------------------------------------------------
# Resources
# ----------------------------------------------------------------------------

class Resource:
    def __init__(self, name):
        self.name = name

    def sample_duration(self, activity, simulator):
        raise NotImplementedError

    def __str__(self):
        return str(self.name)


class DefaultResource(Resource):
    """
    Homogeneous resource. The ONLY source of duration variation beyond the
    lognormal noise is the simulator's global speed regime.
    """

    def __init__(self, name):
        super().__init__(name)
        self.recent_completions = []

    def sample_duration(self, activity, simulator):
        if activity.type == ActivityTypes.DIAGNOSIS:
            base = np.random.lognormal(np.log(1), 0.15)
        elif activity.type == ActivityTypes.REPAIR:
            base = np.random.lognormal(np.log(6), 0.20)
        elif activity.type == ActivityTypes.QUALITY_CONTROL:
            base = np.random.lognormal(np.log(3), 0.15)
        else:
            raise ValueError(f"no duration defined for {activity.type}")

        # The exogenous speed regime. This is the E4 signal.
        return base * simulator.speed_factor


# ----------------------------------------------------------------------------
# Process model (unchanged from the base simulator)
# ----------------------------------------------------------------------------

class ActivityTypes(Enum):
    DIAGNOSIS = 0
    REPAIR = 1
    QUALITY_CONTROL = 2
    FINISHED = 3


class ActivityState(Enum):
    ACTIVATED = 'SCHEDULE'
    STARTED = 'START'
    COMPLETED = 'COMPLETE'

    def __str__(self):
        return str(self.value)


class Activity:
    def __init__(self, id, type, instance):
        self.id = id
        self.type = type
        self.state = ActivityState.ACTIVATED
        self.instance = instance
        self.resource = None

    def start(self, resource):
        self.state = ActivityState.STARTED
        self.resource = resource

    def complete(self):
        self.state = ActivityState.COMPLETED

    def __str__(self):
        return self.type.name


class ControlFlow:
    def first_activity_type(self):
        return ActivityTypes.DIAGNOSIS

    def next_activity_type(self, current_activity_type):
        if current_activity_type == ActivityTypes.DIAGNOSIS:
            return ActivityTypes.REPAIR
        elif current_activity_type == ActivityTypes.REPAIR:
            return ActivityTypes.QUALITY_CONTROL
        elif current_activity_type == ActivityTypes.QUALITY_CONTROL:
            return ActivityTypes.FINISHED


class ProcessInstance:
    control_flow = ControlFlow()

    def __init__(self, id):
        self.id = id
        self.current_activity_id = 0
        self.current_activity = None
        self.activities = []
        self.finished = False

    def start_instance(self):
        first_activity_type = self.control_flow.first_activity_type()
        self.current_activity = Activity(0, first_activity_type, self)
        self.activities.append(self.current_activity)

    def activate_next_activity(self):
        if not self.finished:
            next_activity_type = self.control_flow.next_activity_type(self.current_activity.type)
            self.current_activity_id += 1
            self.current_activity = Activity(self.current_activity_id, next_activity_type, self)
            self.activities.append(self.current_activity)
        if self.current_activity.type == ActivityTypes.FINISHED:
            self.finished = True

    def has_finished(self):
        return self.finished

    def __str__(self):
        return str(self.id)


class EventType(Enum):
    INSTANCE_SPAWN = 0
    ACTIVITY_ACTIVATE = 1
    ACTIVITY_START = 2
    ACTIVITY_COMPLETE = 3
    INSTANCE_COMPLETE = 4


class Event:
    def __init__(self, event_type, event_time, data):
        self.type = event_type
        self.time = event_time
        self.data = data


class Resources:
    def __init__(self, simulator):
        self.resources = [DefaultResource(f"R{i:02d}") for i in range(1, N_RESOURCES + 1)]
        self.idle_resources = self.resources.copy()
        self.working_resources = []
        self.simulator = simulator

    def eligible(self, activity, resource):
        return True

    def allocate(self, resource):
        self.working_resources.append(resource)
        self.idle_resources.remove(resource)

    def free(self, resource):
        self.idle_resources.append(resource)
        self.working_resources.remove(resource)

    def sample_duration(self, activity, resource):
        return resource.sample_duration(activity, self.simulator)


# ----------------------------------------------------------------------------
# Simulator
# ----------------------------------------------------------------------------

class ProcessSimulator:
    def __init__(self, start_time=0, logger=None):
        self.current_time = start_time
        self.max_process_instance_id = -1
        self.event_queue = []
        self.activated_activities = []
        self.logger = logger
        self.resources = Resources(self)

        # Speed regime: determines W. This is what E4 should recover.
        self.speed_slow = False
        self.speed_factor = SPEED_FACTOR_FAST
        self.next_speed_switch = self.current_time + np.random.exponential(SPEED_MEAN_FAST_H)

        # Arrival regime: determines lambda. Independent of the speed regime.
        self.arrival_high = False
        self.arrival_scale = ARRIVAL_SCALE_LOW_H
        self.next_arrival_switch = self.current_time + np.random.exponential(ARRIVAL_MEAN_LOW_H)

        self.spawn_instance()

    # -- regimes -------------------------------------------------------------

    def _update_regimes(self):
        while self.current_time >= self.next_speed_switch:
            self.speed_slow = not self.speed_slow
            if self.speed_slow:
                self.speed_factor = SPEED_FACTOR_SLOW
                self.next_speed_switch += np.random.exponential(SPEED_MEAN_SLOW_H)
            else:
                self.speed_factor = SPEED_FACTOR_FAST
                self.next_speed_switch += np.random.exponential(SPEED_MEAN_FAST_H)

        while self.current_time >= self.next_arrival_switch:
            self.arrival_high = not self.arrival_high
            if self.arrival_high:
                self.arrival_scale = ARRIVAL_SCALE_HIGH_H
                self.next_arrival_switch += np.random.exponential(ARRIVAL_MEAN_HIGH_H)
            else:
                self.arrival_scale = ARRIVAL_SCALE_LOW_H
                self.next_arrival_switch += np.random.exponential(ARRIVAL_MEAN_LOW_H)

    # -- event scheduling ----------------------------------------------------

    def spawn_instance(self):
        self.max_process_instance_id += 1
        next_instance = ProcessInstance(self.max_process_instance_id)
        next_instance_spawn_time = self.current_time + np.random.exponential(scale=self.arrival_scale)
        self.event_queue.append(Event(EventType.INSTANCE_SPAWN,
                                      next_instance_spawn_time,
                                      {'instance': next_instance}))

    def activate_activity(self, activity):
        self.event_queue.append(Event(EventType.ACTIVITY_ACTIVATE,
                                      self.current_time,
                                      {'activity': activity}))

    def start_activity(self, activity, resource):
        self.event_queue.append(Event(EventType.ACTIVITY_START,
                                      self.current_time,
                                      {'activity': activity, 'resource': resource}))

    def complete_activity(self, activity, resource, activity_completed_time):
        self.event_queue.append(Event(EventType.ACTIVITY_COMPLETE,
                                      activity_completed_time,
                                      {'activity': activity, 'resource': resource}))

    # -- event handlers ------------------------------------------------------

    def _instance_spawned(self, event):
        # No batch spawning here: that mechanism belongs to simulator_e6 and
        # would introduce an arrival burst signal that E3 could pick up.
        event.data['instance'].start_instance()
        self.activate_activity(event.data['instance'].current_activity)
        self.spawn_instance()

    def _activity_activated(self, event):
        self.activated_activities.append(event.data['activity'])

    def _activity_started(self, event):
        event.data['activity'].start(event.data['resource'])

    def _activity_completed(self, event):
        event.data['activity'].complete()
        event.data['resource'].recent_completions.append(self.current_time)
        self.resources.free(event.data['resource'])
        event.data['activity'].instance.activate_next_activity()
        if not event.data['activity'].instance.has_finished():
            self.activate_activity(event.data['activity'].instance.current_activity)

    def _instance_completed(self, event):
        pass

    def _start_activated_activities(self):
        for activity in self.activated_activities[:]:
            if not self.resources.idle_resources:
                break
            random.shuffle(self.resources.idle_resources)
            for resource in self.resources.idle_resources:
                if self.resources.eligible(activity, resource):
                    self.resources.allocate(resource)
                    self.activated_activities.remove(activity)
                    self.start_activity(activity, resource)
                    duration = self.resources.sample_duration(activity, resource)
                    self.complete_activity(activity, resource, self.current_time + duration)
                    break

    # -- logging / main loop -------------------------------------------------

    def log_event(self, event):
        if self.logger:
            self.logger.log(event)

    def finish_log(self):
        if self.logger:
            self.logger.finish()

    def simulate(self, max_time=np.inf):
        self.start_time = self.current_time
        self.max_queue_len = 0
        while len(self.event_queue):
            if max_time < self.current_time - self.start_time:
                break
            current_event = self.event_queue.pop(0)
            self.current_time = current_event.time
            self._update_regimes()

            if current_event.type == EventType.INSTANCE_SPAWN:
                self._instance_spawned(current_event)
            elif current_event.type == EventType.ACTIVITY_ACTIVATE:
                self._activity_activated(current_event)
            elif current_event.type == EventType.ACTIVITY_START:
                self._activity_started(current_event)
            elif current_event.type == EventType.ACTIVITY_COMPLETE:
                self._activity_completed(current_event)
            elif current_event.type == EventType.INSTANCE_COMPLETE:
                self._instance_completed(current_event)

            self._start_activated_activities()
            self.max_queue_len = max(self.max_queue_len, len(self.activated_activities))
            self.event_queue.sort(key=lambda event: event.time)
            self.log_event(current_event)
        self.finish_log()
        return


# ----------------------------------------------------------------------------
# Loggers
# ----------------------------------------------------------------------------

class XESLifeCycleLogger:
    def __init__(self, out_file='event_log.xes'):
        self.data = []
        self.start_time = datetime(2020, 1, 1)
        self.out_file = out_file

    def log(self, event):
        if event.type in [EventType.ACTIVITY_ACTIVATE, EventType.ACTIVITY_START, EventType.ACTIVITY_COMPLETE]:
            activity = event.data['activity']
            instance = activity.instance
            resource = event.data['resource'] if event.type != EventType.ACTIVITY_ACTIVATE else None
            timestamp = self.start_time + timedelta(hours=event.time)
            self.data.append([str(instance), str(activity), activity.state, timestamp, str(resource)])

    def finish(self):
        self.df = pd.DataFrame(self.data, columns=['case:concept:name', 'concept:name',
                                                   'lifecycle:transition', 'time:timestamp',
                                                   'org:resource'])
        log = log_converter.apply(self.df, variant=log_converter.Variants.TO_EVENT_LOG)
        pm4py.write_xes(log, self.out_file)


if __name__ == '__main__':
    random.seed(SEED)
    np.random.seed(SEED)

    simulator = ProcessSimulator(logger=XESLifeCycleLogger('data/synthetic_e4.xes'))
    simulator.simulate(SIM_DURATION_H)

    # Sanity check: if this is much above 0, the pool is too small and the
    # speed regime is leaking into E1 via queueing. Increase N_RESOURCES.
    print(f"cases spawned:        {simulator.max_process_instance_id + 1}")
    print(f"max activities queued: {simulator.max_queue_len}")