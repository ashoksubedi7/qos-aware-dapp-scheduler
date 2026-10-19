from types import SimpleNamespace

import pytest

from assurance.candidate_action_assessment import (
    assess_candidate_actions,
    assess_proposed_action,
)
from assurance.scheduling_opportunities import SchedulingClock


class FakeRR:
    def __init__(self, packet=None, bearer=True):
        self.schType = "Rr"
        self.ttiByms = 4.0
        self.mimomd = "SU"
        self.nlayers = 1
        self.ind_u = 0
        self.sLoad = 0.003
        self.sbFrNum = 2
        self.nrbUEmax = 8

        app = []
        buf = []

        if packet is not None:
            (buf if bearer else app).append(packet)

        flow = SimpleNamespace(
            appBuff=SimpleNamespace(pckts=app),
            deadlineEvaluatedPacketIds=set(),
        )

        self.ues = {
            "ue1": SimpleNamespace(
                id="ue1",
                BWPs=20,
                prbs=0,
                pendingTB=[],
                packetFlows=[flow],
                bearers=[
                    SimpleNamespace(
                        buffer=SimpleNamespace(pckts=buf)
                    )
                ],
            )
        }

        self.calls = []

    def setMod(self, ue, prbs):
        self.calls.append((ue, prbs))
        return (prbs * 80, "QPSK", 2, 1)


def packet(deadline=0.75, arrival=0.0):
    return SimpleNamespace(
        secNum=1,
        tIn=arrival,
        deadline=deadline,
        scheduled_at=None,
        size=50,
    )


def evaluate(
    scheduler,
    now=0.1,
    interval=1.0,
    *,
    control_first=False,
    diagnostics=False,
):
    return assess_candidate_actions(
        urllc_slice=SimpleNamespace(
            schedulerDL=scheduler
        ),
        now_ms=now,
        control_interval_ms=interval,
        clock=SchedulingClock(0.0, 0.25),
        total_reference_prbs=32,
        num_ref_factors=(1, 4, 1),
        control_precedes_current_mac_event=control_first,
        include_diagnostics=diagnostics,
    )


def test_all_22_actions_assessed():
    results = evaluate(FakeRR(packet()))
    assert len(results) == 22
    assert [r.action for r in results] == list(range(22))


def test_zero_urllc_is_rejected_off_boundary():
    results = evaluate(FakeRR(packet()))
    assert results[15].urllc_prbs == 0
    assert results[15].verdict == "REJECT"
    assert results[15].reason == (
        "zero_urllc_budget_before_known_deadline"
    )


def test_positive_allocation_does_not_prove_service():
    results = evaluate(FakeRR(packet()))
    assert results[6].urllc_prbs > 0
    assert results[6].verdict == "UNKNOWN"


def test_current_event_order_causes_abstention():
    results = evaluate(
        FakeRR(packet()),
        now=0.0,
    )
    assert results[15].verdict == "UNKNOWN"
    assert results[15].reason == (
        "same_time_event_order_not_established"
    )


def test_application_only_does_not_create_fake_service():
    results = evaluate(
        FakeRR(packet(), bearer=False),
        diagnostics=True,
    )
    assert results[6].potential_payload_bytes == 0
    assert results[6].verdict == "UNKNOWN"


def test_no_known_demand_is_only_potentially_admissible():
    results = evaluate(FakeRR())
    assert all(
        r.verdict == "POTENTIALLY_ADMISSIBLE"
        for r in results
    )


def test_expired_packet_is_not_treated_as_rescuable():
    results = evaluate(
        FakeRR(packet(deadline=0.05))
    )
    assert results[15].verdict == "UNKNOWN"
    assert results[15].reason == (
        "previously_missed_deadline_irrecoverable"
    )


def test_no_future_clock_event_is_reported():
    results = evaluate(
        FakeRR(packet(deadline=0.15))
    )
    assert results[6].verdict == "UNKNOWN"
    assert results[6].reason == (
        "no_future_clock_event_for_known_packet"
    )


def test_deadline_at_next_control_is_not_rejected():
    results = evaluate(
        FakeRR(packet(deadline=1.1))
    )
    assert results[15].verdict == (
        "POTENTIALLY_ADMISSIBLE"
    )


def test_no_scheduler_mutation():
    scheduler = FakeRR(packet())
    ue = scheduler.ues["ue1"]
    before = (
        scheduler.nrbUEmax,
        ue.prbs,
        len(ue.bearers[0].buffer.pckts),
    )

    evaluate(scheduler)

    after = (
        scheduler.nrbUEmax,
        ue.prbs,
        len(ue.bearers[0].buffer.pckts),
    )
    assert before == after


def test_mismatched_tti_rejected():
    scheduler = FakeRR(packet())
    scheduler.ttiByms = 2.0
    with pytest.raises(ValueError):
        evaluate(scheduler)


def test_invalid_control_interval_rejected():
    with pytest.raises(ValueError):
        evaluate(FakeRR(), interval=0.0)



def test_measured_control_first_order_enables_boundary_rejection():
    results = evaluate(
        FakeRR(packet()),
        now=0.0,
        control_first=True,
    )

    assert results[15].verdict == "REJECT"
    assert results[15].reason == (
        "zero_urllc_budget_before_known_deadline"
    )
    assert results[6].verdict == "UNKNOWN"


def test_fast_mode_skips_expensive_radio_projection():
    scheduler = FakeRR(packet())

    results = evaluate(
        scheduler,
        now=0.0,
        control_first=True,
    )

    assert len(results) == 22
    assert scheduler.calls == []
    assert all(
        item.potential_payload_bytes is None
        and item.first_visit_status is None
        for item in results
    )


def test_detailed_projection_remains_available_on_request():
    scheduler = FakeRR(packet())

    results = evaluate(
        scheduler,
        diagnostics=True,
    )

    assert scheduler.calls
    assert results[6].potential_payload_bytes > 0
    assert results[6].first_visit_status is not None



def proposed(
    scheduler,
    action,
    *,
    now=0.1,
    control_first=False,
):
    return assess_proposed_action(
        urllc_slice=SimpleNamespace(
            schedulerDL=scheduler
        ),
        proposed_action=action,
        now_ms=now,
        control_interval_ms=1.0,
        clock=SchedulingClock(0.0, 0.25),
        total_reference_prbs=32,
        num_ref_factors=(1, 4, 1),
        control_precedes_current_mac_event=control_first,
    )


@pytest.mark.parametrize(
    "now,control_first",
    [
        (0.1, False),
        (0.0, False),
        (0.0, True),
    ],
)
@pytest.mark.parametrize("bearer", [True, False])
def test_single_action_matches_full_catalog(
    now, control_first, bearer
):
    scheduler = FakeRR(
        packet(), bearer=bearer
    )

    catalog = evaluate(
        scheduler,
        now=now,
        control_first=control_first,
    )

    assert len(catalog) == 22

    for action, expected in enumerate(catalog):
        actual = proposed(
            scheduler,
            action,
            now=now,
            control_first=control_first,
        )

        assert actual == expected


@pytest.mark.parametrize(
    "invalid",
    [-1, 22, True, 1.5, None],
)
def test_single_action_rejects_invalid_input(invalid):
    with pytest.raises(ValueError):
        proposed(
            FakeRR(packet()),
            invalid,
        )


def test_single_action_uses_one_prb_conversion(monkeypatch):
    import assurance.candidate_action_assessment as module

    original = module.percentage_action_to_numerology_prbs
    calls = []

    def counted(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(
        module,
        "percentage_action_to_numerology_prbs",
        counted,
    )

    result = proposed(
        FakeRR(packet()),
        15,
        now=0.0,
        control_first=True,
    )

    assert len(calls) == 1
    assert result.action == 15
    assert result.verdict == "REJECT"


def test_single_action_skips_radio_diagnostics():
    scheduler = FakeRR(packet())

    result = proposed(
        scheduler,
        17,
        now=0.0,
        control_first=True,
    )

    assert result.action == 17
    assert result.verdict == "UNKNOWN"
    assert result.potential_payload_bytes is None
    assert result.first_visit_status is None
    assert scheduler.calls == []


def test_zero_budget_without_future_mac_opportunity():
    result = evaluate(
        FakeRR(packet(deadline=0.15))
    )[15]

    assert result.verdict == "UNKNOWN"


def test_zero_budget_with_application_only_packet():
    result = evaluate(
        FakeRR(packet(), bearer=False)
    )[15]

    assert result.verdict == "UNKNOWN"
