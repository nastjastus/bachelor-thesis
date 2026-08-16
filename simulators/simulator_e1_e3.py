#!/usr/bin/env python3

from enum import Enum
import numpy as np
import random
import pandas as pd
from pm4py.objects.conversion.log import converter as log_converter
import pm4py
from datetime import datetime, timedelta


class Resource:
    def __init__(self, name):
        self.name = name

    def sample_duration(self, activity, simulator):
        raise NotImplementedError
    
    def __str__(self):
        return str(self.name)


class DefaultResource(Resource):
    def __init__(self, name):
        super().__init__(name)
        self.recent_completions = []

    def sample_duration(self, activity, simulator):
        if activity.type == ActivityTypes.DIAGNOSIS:
            base = np.random.lognormal(np.log(1), 0.2/1)
        elif activity.type == ActivityTypes.REPAIR:
            base = np.random.lognormal(np.log(6), 3/6)
        elif activity.type == ActivityTypes.QUALITY_CONTROL:
            base = np.random.lognormal(np.log(3), 0.5/3)

        # Fatigue effect: how many cases handled in the last 24 h?
        recent = sum(1 for t in self.recent_completions
                     if simulator.current_time - t < 24)
        load_factor = 1 + 0.2 * recent

        return base * load_factor  

class CoffeeTrinkerResource(DefaultResource):
    """
    This resource sometimes makes a coffee during a task:
        GMM durations during tasks
    """
    def sample_duration(self, activity, simulator):
        duration = super().sample_duration(activity, simulator)
        if np.random.choice([True, False], size=1, p=[0.2, 0.8])[0]:
            coffee_duration = np.random.lognormal(np.log(10/60), 5/60/(10/60))
            duration += coffee_duration
        return duration


class EarlyBirdResource(DefaultResource):
    """
    This resource is considerable faster in the morning than in the evening
        Two different normal distributions
    """
    def sample_duration(self, activity, simulator):
        duration = super().sample_duration(activity, simulator)
        if simulator.current_time % 24 < 13:
            duration *= np.random.uniform(0.6, 0.8)
        else:
            duration *= np.random.uniform(1.2, 1.4)
        return duration


class JoeResource(CoffeeTrinkerResource):
    """
    This resource takes longer when Jane has conducted a previous activity
    """
    def sample_duration(self, activity, simulator):
        duration = super().sample_duration(activity, simulator)
        jane_has_done_something = any(isinstance(a.resource, JaneResource) for a in activity.instance.activities)
        if jane_has_done_something:
            offset = np.random.lognormal(np.log(2), 0.1/2)
            duration *= offset
        return duration

class JaneResource(EarlyBirdResource):
    """
    This resource takes longer when Joe has conducted a previous activity
    """
    def sample_duration(self, activity, simulator):
        duration = super().sample_duration(activity, simulator)
        joe_has_done_something = any(isinstance(a.resource, JoeResource) for a in activity.instance.activities)
        if joe_has_done_something:
            offset = np.random.lognormal(np.log(3), 0.1/3)
            duration *= offset
        return duration


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
    def __init__(self):
        pass

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
        self.resources = [DefaultResource(1),
                          JaneResource("Jane"),
                          JoeResource("Joe"),
                          EarlyBirdResource("Clark"),
                          CoffeeTrinkerResource("Karsten")]
        self.idle_resources = self.resources.copy()
        self.working_resources = []
        self.simulator = simulator

    def eligible(self, activity, resource):
        # TODO
        return True

    def allocate(self, resource):
        self.working_resources.append(resource)
        self.idle_resources.remove(resource)

    def free(self, resource):
        self.idle_resources.append(resource)
        self.working_resources.remove(resource)

    def sample_duration(self, activity, resource):
        return resource.sample_duration(activity, self.simulator)


class ProcessSimulator:
    def __init__(self, start_time = 0, logger = None):
        self.current_time = start_time
        self.max_process_instance_id = -1
        self.event_queue = []
        self.activated_activities = []
        self.logger = logger
        self.resources = Resources(self)
        
        self.spawn_instance()
    

    def spawn_instance(self):
        self.max_process_instance_id += 1
        next_instance = ProcessInstance(self.max_process_instance_id)
       # next_instance_spawn_time = self.current_time + np.random.exponential(scale = 24)
       # tuned so that more cases spawn, to create a system overload
        hour_of_week = self.current_time % (24 * 7)
        day_of_week = int(hour_of_week / 24)

        if day_of_week in [0, 1]:      # Monday, Tuesday: many cases
            scale = 2
        elif day_of_week in [2, 3]:    # Wednesday, Thursday: normal
            scale = 8
        else:                           # Friday-Sunday: few cases
            scale = 16

        next_instance_spawn_time = self.current_time + np.random.exponential(scale=scale)
        next_instance_spawn_event = Event(EventType.INSTANCE_SPAWN,
                                          next_instance_spawn_time,
                                          {'instance' : next_instance})
        self.event_queue.append(next_instance_spawn_event)

    def activate_activity(self, activity):
        activate_event = Event(EventType.ACTIVITY_ACTIVATE,
                                self.current_time,
                                {'activity' : activity})
        self.event_queue.append(activate_event)

    def start_activity(self, activity, resource):
        start_activity = Event(EventType.ACTIVITY_START,
                               self.current_time,
                               {'activity' : activity, 'resource' : resource})
        self.event_queue.append(start_activity)

    def complete_activity(self, activity, resource, activity_completed_time):
        activity_completed = Event(EventType.ACTIVITY_COMPLETE,
                                activity_completed_time,
                                {'activity' : activity, 'resource' : resource})
        self.event_queue.append(activity_completed)

    def _instance_spawned(self, event):
        # activate first activity
        event.data['instance'].start_instance()
        self.activate_activity(event.data['instance'].current_activity)
        # set next instance spawn
        self.spawn_instance()

    def _activity_activated(self, event):
        self.activated_activities.append(event.data['activity'])

    def _activity_started(self, event):
        event.data['activity'].start(event.data['resource'])

    def _activity_completed(self, event):
        event.data['activity'].complete()
        event.data['resource'].recent_completions.append(self.current_time)  # for E2
        self.resources.free(event.data['resource'])
        event.data['activity'].instance.activate_next_activity()
        if not event.data['activity'].instance.has_finished():
            self.activate_activity(event.data['activity'].instance.current_activity)

    def _instance_completed(self, event):
        pass

    def _start_activated_activities(self):
        restart = False
        for activity in self.activated_activities:
            random.shuffle(self.resources.idle_resources)
            for resource in self.resources.idle_resources:
                if self.resources.eligible(activity, resource):
                    self.resources.allocate(resource)
                    self.activated_activities.remove(activity)
                    # set activity start
                    self.start_activity(activity, resource)

                    # set activity finish
                    duration = self.resources.sample_duration(activity, resource)
                    self.complete_activity(activity, resource, self.current_time + duration)
                    break


    def log_event(self, event):
        if self.logger:
            self.logger.log(event)

    def finish_log(self):
        if self.logger:
            self.logger.finish()

    def simulate(self, max_time = np.inf):
        self.start_time = self.current_time
        while len(self.event_queue):
            if max_time < self.current_time - self.start_time:
                break
            current_event = self.event_queue.pop(0)
            self.current_time = current_event.time

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
            self.event_queue.sort(key = lambda event : event.time)
            self.log_event(current_event)
        self.finish_log()
        return


class PrintLogger:
    def log(self, event):
        activity = event.data['activity'] if event.type not in [EventType.INSTANCE_COMPLETE, EventType.INSTANCE_SPAWN] else None
        instance = event.data['instance'] if event.type in [EventType.INSTANCE_COMPLETE, EventType.INSTANCE_SPAWN] else event.data['activity'].instance
        resource = event.data['resource'] if event.type in [EventType.ACTIVITY_START] else None
        print(event.time, instance, event.type, activity, resource)

    def finish(self):
        pass

class PandasLogger:
    def __init__(self, out_file):
        self.data = []
        self.start_time = datetime(2020, 1, 1)
        self.out_file = out_file

    def log(self, event):
        if event.type == EventType.ACTIVITY_START:
            activity = event.data['activity']
            resource = event.data['resource']
            timestamp = self.start_time + timedelta(hours = event.time)
            self.data.append([str(activity.instance), str(activity), timestamp, str(resource)])

    def finish(self):
        self.df = pd.DataFrame(self.data, columns = ['case:concept:name', 'concept:name', 'time:timestamp', 'org:resource'])


class XESLogger(PandasLogger):
    def finish(self):
        super().finish()
        # Convert DataFrame to Event Log
        log = log_converter.apply(self.df, variant=log_converter.Variants.TO_EVENT_LOG)
        # Export to XES
        pm4py.write_xes(log, self.out_file)


class XESLifeCycleLogger:
    def __init__(self, out_file='event_log.xes'):
        self.data = []
        self.start_time = datetime(2020, 1, 1)
        self.out_file = out_file

    def log(self, event):
        if event.type in [EventType.ACTIVITY_ACTIVATE, EventType.ACTIVITY_START, EventType.ACTIVITY_COMPLETE]:
            activity = event.data['activity']
            instance = event.data['activity'].instance
            resource = event.data['resource'] if event.type != EventType.ACTIVITY_ACTIVATE else None
            timestamp = self.start_time + timedelta(hours = event.time)
            self.data.append([str(instance), str(activity), activity.state, timestamp, str(resource)])

    def finish(self):
        self.df = pd.DataFrame(self.data, columns = ['case:concept:name', 'concept:name', 'lifecycle:transition', 'time:timestamp', 'org:resource'])
        # Convert DataFrame to Event Log
        log = log_converter.apply(self.df, variant=log_converter.Variants.TO_EVENT_LOG)
        # Export to XES
        pm4py.write_xes(log, self.out_file)

class CSVLogger(PandasLogger):
    def __init__(self, out_file='event_log.csv'):
        super().__init__(out_file)

    def finish(self):
        super().finish()
        self.df.to_csv(self.out_file, index=False)

if __name__ == '__main__':
    #simulator = ProcessSimulator(logger=CSVLogger('test_log.csv'))
    simulator = ProcessSimulator(logger=XESLifeCycleLogger('data/synthetic_e1_e3.xes'))
    simulator.simulate(24*365*5/2)
