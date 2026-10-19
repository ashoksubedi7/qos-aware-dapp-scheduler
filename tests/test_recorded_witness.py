import hashlib
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "evidence/witnesses/m1-update50-seed202-t1"


def test_saved_witness_hashes():
    for line in (EVIDENCE / "SHA256SUMS").read_text().splitlines():
        expected, filename = line.split()

        actual = hashlib.sha256(
            (EVIDENCE / filename).read_bytes()
        ).hexdigest()

        assert actual == expected, filename


def test_recorded_packet_counterfactual():
    result = subprocess.run(
        [
            sys.executable,
            "scripts/assurance/verify_witness.py",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
    )

    assert result.returncode == 0, (
        result.stdout + result.stderr
    )
    assert (
        "Verified packet-specific misses avoided: 3"
        in result.stdout
    )
