import json
import math
import pytest

import auto_calibrator


# Set expected test results
RHO_EPSILON_EXPECTED = 0.8232437187424534
TAU_EPSILON_EXPECTED = 0.29720679135508243
ALPHA_EXPECTED = -0.05741640058288248
BETA_EXPECTED = -0.2086519072401008
GAMMA_EXPECTED = 0.5212586272873916
CALLBACK_ALPHA_EXPECTED = -0.0009024826257139674
CALLBACK_BETA_EXPECTED = -0.755986887017733
CALLBACK_GAMMA_EXPECTED = 0.18989302171916092
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
        min_horizontal_fov_fit=3.803123285538903,
        max_horizontal_fov_fit=48.31584827530176,
        scale_horizontal_fov_fit=0.001627997881937721,
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
def pointing_error_msg_1():
    """Load mock calibration message."""
    with open("data/pointing_error_msg_1.json") as f:
        msg = json.load(f)
    return msg


@pytest.fixture
def pointing_error_msg_2():
    """Load mock calibration message."""
    with open("data/pointing_error_msg_2.json") as f:
        msg = json.load(f)
    return msg


@pytest.fixture
def pointing_error_msg_3():
    """Load mock calibration message."""
    with open("data/pointing_error_msg_3.json") as f:
        msg = json.load(f)
    return msg


class TestAutoCalibrator:
    """Test construction of rotations and calculation of camera
    pointing.
    """

    def test_calculate_calibration_error(self, calibrator, pointing_error_msg_1):
        """Test calculation of calibration error."""

        (
            rho_epsilon,
            tau_epsilon,
        ) = calibrator._calculate_calibration_error(pointing_error_msg_1)

        assert rho_epsilon == RHO_EPSILON_EXPECTED
        assert tau_epsilon == TAU_EPSILON_EXPECTED

    def test_calculate_pointing_error(self):
        """Tested implicitly."""
        pass

    def test_minimize_pointing_error(self, calibrator, pointing_error_msg_1):
        """Test pointing error minimization."""

        # _minimize_pointing_error method is designed to work with minimum data
        calibrator.rho_epsilon_list = [RHO_EPSILON_EXPECTED] * 8
        calibrator.tau_epsilon_list = [TAU_EPSILON_EXPECTED] * 8
        calibrator.data_list = [pointing_error_msg_1] * 8

        alpha, beta, gamma = calibrator._minimize_pointing_error()

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

    def test_pointing_error_callback(self, calibrator, pointing_error_msg_1, pointing_error_msg_2, pointing_error_msg_3):
        """Test calibration callback reads messages, calculates alpha,
        beta, and gamma correctly, and updates those values.
        """
        _client = None
        _userdata = None

        # pointing_error_callback requires a minimum amount of data before processing.
        # while loop ensures calibrator processes before moving on
        while calibrator.alpha == 0.0 and calibrator.beta == 0.0 and calibrator.gamma == 0.0:
            calibrator._pointing_error_callback(_client, _userdata, pointing_error_msg_1)
            calibrator._pointing_error_callback(_client, _userdata, pointing_error_msg_2)
            calibrator._pointing_error_callback(_client, _userdata, pointing_error_msg_3)

        assert math.fabs(calibrator.alpha - CALLBACK_ALPHA_EXPECTED) < PRECISION
        assert math.fabs(calibrator.beta - CALLBACK_BETA_EXPECTED) < PRECISION
        assert math.fabs(calibrator.gamma - CALLBACK_GAMMA_EXPECTED) < PRECISION
