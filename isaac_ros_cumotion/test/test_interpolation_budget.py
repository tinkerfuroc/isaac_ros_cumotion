# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the interpolation_steps sizing rule (pure python, no GPU/ROS).

The module is loaded by file path so the test does not import the
``isaac_ros_cumotion`` package ``__init__`` (which pulls in cuRobo / torch).
"""
import importlib.util
import os

import pytest

_MODULE_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    'isaac_ros_cumotion', 'interpolation_budget.py')
_spec = importlib.util.spec_from_file_location('interpolation_budget', _MODULE_PATH)
interpolation_budget = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(interpolation_budget)
check_interpolation_steps = interpolation_budget.check_interpolation_steps
required_interpolation_steps = interpolation_budget.required_interpolation_steps


def test_production_config_needs_372_rows():
    # planner.yml: num_trajopt_time_steps 32, maximum_trajectory_dt 0.30,
    # interpolation_dt 0.025 -> ceil(31 * 0.30 / 0.025) = 372.
    assert required_interpolation_steps(32, 0.30, 0.025) == 372


def test_curobo_default_and_production_value_pass():
    assert check_interpolation_steps(5000, 32, 0.30, 0.025) == 372
    assert check_interpolation_steps(1536, 32, 0.30, 0.025) == 372  # production value
    assert check_interpolation_steps(1024, 32, 0.30, 0.025) == 372
    assert check_interpolation_steps(372, 32, 0.30, 0.025) == 372


def test_undersized_buffer_is_rejected_with_actionable_message():
    with pytest.raises(ValueError) as excinfo:
        check_interpolation_steps(371, 32, 0.30, 0.025)
    msg = str(excinfo.value)
    assert 'interpolation_steps=371' in msg
    assert '372' in msg
    assert 'maximum_trajectory_dt' in msg


def test_requirement_scales_with_horizon_and_dt():
    # Longer horizon or larger max dt needs more rows; finer interpolation too.
    assert required_interpolation_steps(48, 0.30, 0.025) == 564
    assert required_interpolation_steps(32, 0.15, 0.025) == 186
    assert required_interpolation_steps(32, 0.30, 0.0125) == 744


def test_invalid_inputs():
    with pytest.raises(ValueError):
        required_interpolation_steps(1, 0.30, 0.025)
    with pytest.raises(ValueError):
        required_interpolation_steps(32, 0.30, 0.0)
