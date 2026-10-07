from SchedulerUtils import (
    getTbs,
)


def test_get_tbs_uses_explicit_table():
    table = [
        24,
        32,
        40,
        48,
        56,
        64,
        72,
    ]

    result = getTbs(
        Ninfo=50,
        r=0.5,
        tbs_table=table,
    )

    assert result == 48
