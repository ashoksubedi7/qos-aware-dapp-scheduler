import importlib
from types import SimpleNamespace

import numpy as np
import pytest


class FakeAgent:
    def __init__(self, events, loss=0.25):
        self.events = events
        self.loss = loss
        self.training_steps = 0
        self.epsilon = 1.0
        self.replay = []

    def remember(self, state, action, reward, next_state, done):
        self.events.append("remember")
        self.replay.append((
            tuple(state),
            action,
            reward,
            tuple(next_state),
            done,
        ))

    def train_step(self):
        self.events.append("train")
        if self.loss is None:
            return None
        self.training_steps += 1
        return self.loss

    def decay_epsilon(self):
        self.events.append("decay")
        self.epsilon *= 0.9


class FakeRecorder:
    def __init__(self, events):
        self.events = events
        self.transitions = []
        self.updates = []

    def record_transition(self, **values):
        self.events.append("transition")
        self.transitions.append(values)

    def record_update(self, **values):
        self.events.append("update")
        self.updates.append(values)


def make_scheduler(recording, loss=0.25):
    module = importlib.import_module("assured_scheduler")

    classes = [
        obj
        for obj in vars(module).values()
        if isinstance(obj, type)
        and "_train_transition" in vars(obj)
    ]
    assert len(classes) == 1

    scheduler = classes[0].__new__(classes[0])
    events = []

    scheduler.config = SimpleNamespace(training_mode=True)
    scheduler.learning_enabled = True
    scheduler.agent = FakeAgent(events, loss=loss)

    scheduler.successful_update_callback = (
        lambda scheduler, step, value:
        events.append("controller")
    )

    recorder = FakeRecorder(events) if recording else None
    scheduler.set_training_recorder(recorder)

    return scheduler, recorder, events


def perform_transition(scheduler):
    return scheduler._train_transition(
        state=np.array([0.2, 0.4], dtype=np.float32),
        action=3,
        reward=0.5,
        next_state=np.array([0.3, 0.5], dtype=np.float32),
    )


def test_recording_preserves_original_training_behavior():
    original, _, original_events = make_scheduler(False)
    recorded, recorder, recorded_events = make_scheduler(True)

    assert perform_transition(original) == 0.25
    assert perform_transition(recorded) == 0.25

    assert original.agent.replay == recorded.agent.replay
    assert original.agent.training_steps == recorded.agent.training_steps
    assert original.agent.epsilon == recorded.agent.epsilon

    assert original_events == [
        "remember", "train", "decay", "controller",
    ]

    assert recorded_events == [
        "remember", "transition", "train",
        "decay", "controller", "update",
    ]

    assert len(recorder.transitions) == 1
    assert len(recorder.updates) == 1
    assert recorder.transitions[0]["action"] == 3
    assert recorder.updates[0]["training_step"] == 1
    assert recorder.updates[0]["epsilon"] == 0.9


def test_warmup_transition_is_recorded_without_update():
    scheduler, recorder, events = make_scheduler(
        True, loss=None,
    )

    assert perform_transition(scheduler) is None

    assert len(recorder.transitions) == 1
    assert recorder.updates == []
    assert scheduler.agent.training_steps == 0
    assert scheduler.agent.epsilon == 1.0
    assert events == ["remember", "transition", "train"]


def test_recording_failure_stops_training():
    scheduler, recorder, events = make_scheduler(True)

    def fail(**values):
        raise OSError("training-data write failed")

    recorder.record_transition = fail

    with pytest.raises(OSError, match="write failed"):
        perform_transition(scheduler)

    assert "train" not in events
    assert scheduler.agent.training_steps == 0
    assert recorder.updates == []


def test_invalid_recorder_is_rejected():
    scheduler, _, _ = make_scheduler(False)

    with pytest.raises(TypeError, match="training recorder"):
        scheduler.set_training_recorder(object())
