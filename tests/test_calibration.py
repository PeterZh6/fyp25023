"""Tests for calibration pipeline."""

import json
import os
import tempfile

import pytest

from calibration.compute_distributions import compute_distributions
from calibration.compute_success_rates import compute_success_rates


@pytest.fixture
def mock_labels_dir(tmp_path):
    """Create a temp directory with mock silver label JSONs."""
    mock_data = {
        "binary": "mock_binary",
        "total": 3,
        "summary": {"easy": 1, "medium": 1, "medium_conflict": 0, "medium_partial": 1, "hard": 1},
        "sites": [
            {
                "address": "00001000",
                "type": "jump",
                "type_source": "ghidra",
                "ghidra_type": "jump",
                "angr_type": "jump",
                "type_agree": True,
                "function": "func_a",
                "label": "easy",
                "label_detail": "easy",
                "ghidra_targets": ["00001004", "00001008"],
                "angr_targets": ["00001004", "00001008"],
                "overlap_ratio": 1.0,
                "ghidra_found": True,
                "angr_found": True,
            },
            {
                "address": "00002000",
                "type": "call",
                "type_source": "ghidra",
                "ghidra_type": "call",
                "angr_type": "call",
                "type_agree": True,
                "function": "func_b",
                "label": "medium",
                "label_detail": "medium_partial",
                "ghidra_targets": ["00002004"],
                "angr_targets": [],
                "overlap_ratio": 0.0,
                "ghidra_found": True,
                "angr_found": False,
            },
            {
                "address": "00003000",
                "type": "jump",
                "type_source": "ghidra",
                "ghidra_type": "jump",
                "angr_type": "jump",
                "type_agree": False,
                "function": "func_c",
                "label": "hard",
                "label_detail": "hard",
                "ghidra_targets": [],
                "angr_targets": [],
                "overlap_ratio": None,
                "ghidra_found": True,
                "angr_found": True,
            },
        ],
    }

    fpath = tmp_path / "mock_binary_labels.json"
    with open(fpath, "w") as f:
        json.dump(mock_data, f)
    return str(tmp_path)


def test_compute_distributions_uniform(mock_labels_dir):
    result = compute_distributions(mock_labels_dir)
    pb = result["per_binary"]["mock_binary"]

    assert pb["total_sites"] == 3
    assert abs(pb["easy"] - 1 / 3) < 0.01
    assert abs(pb["medium"] - 1 / 3) < 0.01
    assert abs(pb["hard"] - 1 / 3) < 0.01
    assert pb["counts"] == {"easy": 1, "medium": 1, "hard": 1}


def test_compute_distributions_mixed(mock_labels_dir):
    result = compute_distributions(mock_labels_dir)
    mixed = result["mixed_all"]
    assert mixed["total_sites"] == 3


def test_compute_success_rates_l1(mock_labels_dir):
    result = compute_success_rates(mock_labels_dir)
    rates = result["per_binary"]["mock_binary"]

    assert rates["L1"]["easy"] == 1.0
    assert rates["L1"]["medium"] == 1.0
    assert rates["L1"]["hard"] == 0.0


def test_compute_success_rates_l2(mock_labels_dir):
    result = compute_success_rates(mock_labels_dir)
    rates = result["per_binary"]["mock_binary"]

    assert rates["L2"]["easy"] == 1.0
    assert rates["L2"]["medium"] == 0.0
    assert rates["L2"]["hard"] is None


def test_compute_success_rates_with_overrides(mock_labels_dir):
    result = compute_success_rates(
        mock_labels_dir,
        l2_hard_estimate=0.15,
        l3_easy=1.0,
        l3_medium=0.95,
        l3_hard=0.7,
    )
    rates = result["per_binary"]["mock_binary"]

    assert rates["L2"]["hard"] == 0.15
    assert rates["L3"]["easy"] == 1.0
    assert rates["L3"]["medium"] == 0.95
    assert rates["L3"]["hard"] == 0.7


def test_medium_breakdown(mock_labels_dir):
    result = compute_success_rates(mock_labels_dir)
    bkdn = result["medium_breakdown"]["mock_binary"]

    assert bkdn["medium_partial"] == 1
    assert bkdn["medium_conflict"] == 0
    assert bkdn["ghidra_only"] == 1
    assert bkdn["angr_only"] == 0


def test_real_data_gcc_distribution():
    """Spot-check gcc against known values (requires real data)."""
    labels_dir = "data/silver_labels"
    if not os.path.exists(labels_dir):
        pytest.skip("Real data not available")

    result = compute_distributions(labels_dir)
    gcc = result["per_binary"].get("gcc_base.arm32-gcc81-O3")
    if gcc is None:
        pytest.skip("gcc label file not found")

    assert abs(gcc["easy"] - 0.5115) < 0.005
    assert abs(gcc["medium"] - 0.2132) < 0.005
    assert abs(gcc["hard"] - 0.2753) < 0.005
