import math
import pytest
import json

import numpy as np

import auto_calibrator
import utils_auto_calibrator

@pytest.fixture
def calibration_data():
    # Load test message
    with open("data/calibration_msg.json") as f:
        data = json.load(f)["data"]
    return data

class TestAutoCalibrator:
    """Test construction of rotations and calculation of camera
    pointing."""

    def test_calculate_calibration_error(self, calibration_data):
        """Test calculation of calibration error"""
        rho_0, tau_0, rho_epsilon, tau_epsilon = auto_calibrator.AutoCalibrator._calculate_calibration_error(calibration_data)

        assert rho_0 == -47.617992362840106
        assert tau_0 == 1.500321537077791
        assert rho_epsilon == -1.449921875
        assert tau_epsilon == -3.032962962962963

    def test_calculate_pointing_error(self):
        """Tested implicitly."""
        pass

    def test_minimize(self):
        """TODO: Complete"""
        pass

