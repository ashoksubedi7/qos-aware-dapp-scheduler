from types import SimpleNamespace

from Results import (
    prepare_end_of_run_packet_accounting,
)


class Packet:
    def __init__(
        self,
        sec_num,
    ):
        self.secNum = sec_num


class Buffer:
    def __init__(
        self,
        packets=None,
    ):
        self.pckts = (
            list(
                packets
                or []
            )
        )


def make_flow(
    assured=True,
):
    return SimpleNamespace(
        generatedPacketIds=set(),
        completedPacketIds=set(),
        droppedPacketIds=set(),
        lostPackets=0,
        appBuff=Buffer(),
        experiment_config=(
            object()
            if assured
            else None
        ),
    )


def make_ue():
    return SimpleNamespace(
        id="ue1",
        pendingPckts={},
        pendingTB=[],
        bearers=[
            SimpleNamespace(
                buffer=Buffer()
            )
        ],
    )


def make_scheduler(
    ue,
):
    return SimpleNamespace(
        ues={
            ue.id: ue
        },
        queue=SimpleNamespace(
            res=[]
        ),
    )


def test_assured_residual_is_not_counted_as_drop():
    flow = make_flow(
        assured=True
    )

    ue = make_ue()

    scheduler = (
        make_scheduler(
            ue
        )
    )

    flow.generatedPacketIds.add(
        7
    )

    flow.appBuff.pckts.append(
        Packet(
            7
        )
    )

    accounting = (
        prepare_end_of_run_packet_accounting(
            flow,
            ue,
            scheduler,
        )
    )

    assert (
        accounting.generated
        == 1
    )

    assert (
        accounting.delivered
        == 0
    )

    assert (
        accounting.dropped
        == 0
    )

    assert (
        accounting.residual
        == 1
    )

    assert (
        flow.lostPackets
        == 0
    )


def test_assured_true_drop_remains_drop():
    flow = make_flow(
        assured=True
    )

    ue = make_ue()

    scheduler = (
        make_scheduler(
            ue
        )
    )

    flow.generatedPacketIds.add(
        7
    )

    flow.droppedPacketIds.add(
        7
    )

    flow.lostPackets = 1

    accounting = (
        prepare_end_of_run_packet_accounting(
            flow,
            ue,
            scheduler,
        )
    )

    assert (
        accounting.dropped
        == 1
    )

    assert (
        accounting.residual
        == 0
    )

    assert (
        flow.lostPackets
        == 1
    )


def test_assured_completed_packet_is_not_residual():
    flow = make_flow(
        assured=True
    )

    ue = make_ue()

    scheduler = (
        make_scheduler(
            ue
        )
    )

    flow.generatedPacketIds.add(
        7
    )

    flow.completedPacketIds.add(
        7
    )

    accounting = (
        prepare_end_of_run_packet_accounting(
            flow,
            ue,
            scheduler,
        )
    )

    assert (
        accounting.delivered
        == 1
    )

    assert (
        accounting.dropped
        == 0
    )

    assert (
        accounting.residual
        == 0
    )


def test_legacy_run_preserves_pending_as_lost():
    flow = make_flow(
        assured=False
    )

    ue = make_ue()

    scheduler = (
        make_scheduler(
            ue
        )
    )

    ue.pendingPckts[
        7
    ] = 1

    result = (
        prepare_end_of_run_packet_accounting(
            flow,
            ue,
            scheduler,
        )
    )

    assert result is None

    assert (
        flow.lostPackets
        == 1
    )


def test_legacy_fragment_duplicate_is_counted_once():
    flow = make_flow(
        assured=False
    )

    ue = make_ue()

    scheduler = (
        make_scheduler(
            ue
        )
    )

    ue.pendingPckts[
        7
    ] = 1

    ue.bearers[
        0
    ].buffer.pckts.append(
        Packet(
            7
        )
    )

    prepare_end_of_run_packet_accounting(
        flow,
        ue,
        scheduler,
    )

    assert (
        flow.lostPackets
        == 1
    )
