import json
import math
import pytest

import auto_calibrator


# Set expected test results
RHO_EPSILON_EXPECTED = -1.449921875
TAU_EPSILON_EXPECTED = -3.032962962962963
ALPHA_EXPECTED = 96.22945929035237
BETA_EXPECTED = 31.55893394983606
GAMMA_EXPECTED = 1.5230141040882903
MIN_ZOOM_EXPECTED = 0
MAX_ZOOM_EXPECTED = 9999

# Set precision of angle [deg] differences
PRECISION = 1.0e-5


@pytest.fixture
def calibrator():
    """Construct a calibrator."""
    calibrator = auto_calibrator.AutoCalibrator(
        pointing_error_topic="skyscan/pointing_error/json",
        calibration_topic="skyscan/calibration/json",
        config_topic="skyscan/config/json",
        heartbeat_interval=10.0,
        min_zoom=MIN_ZOOM_EXPECTED,
        max_zoom=MAX_ZOOM_EXPECTED,
        min_horizontal_fov=6.7,
        max_horizontal_fov=61.8,
        min_vertical_fov=3.8,
        max_vertical_fov=37.2,
        mqtt_ip="mqtt",
        use_mqtt=False,
    )
    return calibrator


@pytest.fixture
def config_msg():
    """Load mock config message."""
    with open("data/config_msg_module.json") as f:
        msg = json.load(f)
    return msg


@pytest.fixture
def calibration_msg():
    """Load mock calibration message."""
    with open("data/pointing_error_msg.json") as f:
        msg = json.load(f)
    return msg


@pytest.fixture
def calibration_data(calibration_msg):
    """Load calibration data from message."""
    return calibration_msg["data"]


@pytest.fixture
def additional_info_msg():
    """WIP... to be deleted."""
    with open("data/additional_info_msg.json") as f:
        msg = json.load(f)
    return msg


@pytest.fixture
def additional_data(additional_info_msg):
    """WIP... to be deleted."""
    return additional_info_msg["data"]


class TestAutoCalibrator:
    """Test construction of rotations and calculation of camera
    pointing.
    """

    def test_calculate_calibration_error(self, calibrator, calibration_data):
        """Test calculation of calibration error."""

        (
            rho_epsilon,
            tau_epsilon,
        ) = calibrator._calculate_calibration_error(calibration_data)

        assert rho_epsilon == RHO_EPSILON_EXPECTED
        assert tau_epsilon == TAU_EPSILON_EXPECTED

    def test_calculate_pointing_error(self):
        """Tested implicitly."""
        pass

    def test_minimize_pointing_error(self, calibrator, additional_data):
        """Test pointing error minimization."""

        rho_epsilon = RHO_EPSILON_EXPECTED
        tau_epsilon = TAU_EPSILON_EXPECTED

        alpha, beta, gamma = calibrator._minimize_pointing_error(
            additional_data, rho_epsilon, tau_epsilon
        )

        assert math.fabs(alpha - ALPHA_EXPECTED) < PRECISION
        assert math.fabs(beta - BETA_EXPECTED) < PRECISION
        assert math.fabs(gamma - GAMMA_EXPECTED) < PRECISION

    def test_config_callback(self, calibrator, config_msg):
        """Test config callback updates values, or not."""

        _client = None
        _userdata = None
        calibrator._config_callback(_client, _userdata, config_msg)

        # Assert changes for values in config msg
        assert calibrator.min_horizontal_fov == 1.0
        assert calibrator.max_horizontal_fov == 1.0
        assert calibrator.min_vertical_fov == 1.0
        assert calibrator.max_vertical_fov == 1.0
        assert calibrator.horizontal_pixels == 1.0
        assert calibrator.vertical_pixels == 1.0
        assert calibrator.alpha == 1.0
        assert calibrator.beta == 1.0
        assert calibrator.gamma == 1.0

        # Assert no changes for values not in config msg
        assert calibrator.min_zoom == MIN_ZOOM_EXPECTED
        assert calibrator.max_zoom == MAX_ZOOM_EXPECTED

    def test_calibration_callback(self, calibrator, additional_info_msg):
        """Test calibration callback reads message, calculates alpha,
        beta, and gamma correctly, and updates those values.
        """
        _client = None
        _userdata = None
        calibrator._pointing_error_callback(_client, _userdata, additional_info_msg)

        assert math.fabs(calibrator.alpha - ALPHA_EXPECTED) < PRECISION
        assert math.fabs(calibrator.beta - BETA_EXPECTED) < PRECISION
        assert math.fabs(calibrator.gamma - GAMMA_EXPECTED) < PRECISION
