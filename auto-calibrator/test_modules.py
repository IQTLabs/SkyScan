import json
import math
import pytest

import auto_calibrator


# Set expected test results
RHO_EPSILON_EXPECTED = 0.272939225747946
TAU_EPSILON_EXPECTED = 0.9167572668664233
ALPHA_EXPECTED = 0.28128766932337035
BETA_EXPECTED = -0.8980084867517584
GAMMA_EXPECTED = 0.8502402424547709
MIN_ZOOM_EXPECTED = 0
MAX_ZOOM_EXPECTED = 9999

# Set precision of angle [deg] differences
PRECISION = 1.0e-5


@pytest.fixture
def calibrator():
    """Construct a calibrator."""
    calibrator = auto_calibrator.AutoCalibrator(
        mqtt_ip="mqtt",
        config_topic="skyscan/config/json",
        pointing_error_topic="skyscan/pointing_error/json",
        calibration_topic="skyscan/calibration/json",
        heartbeat_interval=10.0,
        min_zoom=MIN_ZOOM_EXPECTED,
        max_zoom=MAX_ZOOM_EXPECTED,
        fit_horizontal_fov_max=48.31584827530176,
        fit_horizontal_fov_scale=0.001627997881937721,
        fit_horizontal_fov_min=3.803123285538903,
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
def pointing_error_msg():
    """Load mock calibration message."""
    with open("data/pointing_error_msg.json") as f:
        msg = json.load(f)
    return msg


class TestAutoCalibrator:
    """Test construction of rotations and calculation of camera
    pointing.
    """

    def test_calculate_calibration_error(self, calibrator, pointing_error_msg):
        """Test calculation of calibration error."""

        (
            rho_epsilon,
            tau_epsilon,
        ) = calibrator._calculate_calibration_error(pointing_error_msg)

        assert rho_epsilon == RHO_EPSILON_EXPECTED
        assert tau_epsilon == TAU_EPSILON_EXPECTED

    def test_calculate_pointing_error(self):
        """Tested implicitly."""
        pass

    def test_minimize_pointing_error(self, calibrator, pointing_error_msg):
        """Test pointing error minimization."""

        rho_epsilon = RHO_EPSILON_EXPECTED
        tau_epsilon = TAU_EPSILON_EXPECTED

        alpha, beta, gamma = calibrator._minimize_pointing_error(
            pointing_error_msg, rho_epsilon, tau_epsilon
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
        assert calibrator.horizontal_pixels == 1.0
        assert calibrator.vertical_pixels == 1.0
        assert calibrator.alpha == 1.0
        assert calibrator.beta == 1.0
        assert calibrator.gamma == 1.0

        # Assert no changes for values not in config msg
        assert calibrator.min_zoom == MIN_ZOOM_EXPECTED
        assert calibrator.max_zoom == MAX_ZOOM_EXPECTED

    def test_calibration_callback(self, calibrator, pointing_error_msg):
        """Test calibration callback reads message, calculates alpha,
        beta, and gamma correctly, and updates those values.
        """
        _client = None
        _userdata = None
        calibrator._pointing_error_callback(_client, _userdata, pointing_error_msg)
        # Calibrator saves alpha beta gamma by averaging previous value. In this case 0.0
        assert math.fabs(calibrator.alpha - ((ALPHA_EXPECTED + 0.0) / 2)) < PRECISION
        assert math.fabs(calibrator.beta - ((BETA_EXPECTED + 0.0) / 2)) < PRECISION
        assert math.fabs(calibrator.gamma - ((GAMMA_EXPECTED + 0.0) / 2)) < PRECISION
