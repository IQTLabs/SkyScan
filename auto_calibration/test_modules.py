import pytest
import json

import auto_calibrator


@pytest.fixture
def calibration_data():
    # Load test message
    with open("data/calibration_msg.json") as f:
        msg = json.load(f)["data"]
    return msg


@pytest.fixture
def additional_data():
    # Load test message
    with open("data/additional_info_msg.json") as f:
        msg = json.load(f)["data"]
    return msg


class TestAutoCalibrator:
    """Test construction of rotations and calculation of camera
    pointing."""

    def test_calculate_calibration_error(self, calibration_data):
        """Test calculation of calibration error"""

        rho_0, tau_0, rho_epsilon, tau_epsilon = auto_calibrator.AutoCalibrator._calculate_calibration_error(
            calibration_data)

        assert rho_0 == -47.617992362840106
        assert tau_0 == 1.500321537077791
        assert rho_epsilon == -1.449921875
        assert tau_epsilon == -3.032962962962963

    def test_calculate_pointing_error(self):
        """Tested implicitly."""
        pass

    def test_minimize(self, additional_data):
        """Test Broyden, Fletcher, Goldfarb and Shanno minimization algorithm for pointing error method"""

        rho_0 = -47.617992362840106
        tau_0 = 1.500321537077791
        rho_epsilon = -1.449921875
        tau_epsilon = -3.032962962962963

        alpha, beta, gamma = auto_calibrator.AutoCalibrator._minimize(auto_calibrator.AutoCalibrator, additional_data, rho_0, tau_0, rho_epsilon, tau_epsilon)

        assert alpha == 90.81332044751359
        assert beta == 51.76150246750204
        assert gamma == -20.249553096392578

