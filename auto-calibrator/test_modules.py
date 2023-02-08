import json

import pytest

import auto_calibrator

# Set expected test results
rho_epsilon_expected = -1.449921875
tau_epsilon_expected = -3.032962962962963
alpha_expected = 96.22945929035237
beta_expected = 31.55893394983606
gamma_expected = 1.5230141040882903
min_zoom_expected = 0
max_zoom_expected = 9999


@pytest.fixture
def calibrator():
    """Construct a calibrator."""
    calibrator = auto_calibrator.AutoCalibrator(
        config_topic="skyscan/config/json",
        calibration_topic="skyscan/calibration/json",
        min_zoom=min_zoom_expected,
        max_zoom=max_zoom_expected,
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
    with open("data/config_msg.json") as f:
        msg = json.load(f)
    return msg


@pytest.fixture
def calibration_msg():
    """Load mock calibration message."""
    with open("data/calibration_msg.json") as f:
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
    pointing."""

    def test_calculate_calibration_error(self, calibrator, calibration_data):
        """Test calculation of calibration error."""

        (
            rho_epsilon,
            tau_epsilon,
        ) = calibrator._calculate_calibration_error(calibration_data)

        assert rho_epsilon == rho_epsilon_expected
        assert tau_epsilon == tau_epsilon_expected

    def test_calculate_pointing_error(self):
        """Tested implicitly."""
        pass

    def test_minimize_pointing_error(self, calibrator, additional_data):
        """Test pointing error minimization."""

        rho_epsilon = rho_epsilon_expected
        tau_epsilon = tau_epsilon_expected

        alpha, beta, gamma = calibrator._minimize_pointing_error(
            additional_data, rho_epsilon, tau_epsilon
        )

        assert alpha == alpha_expected
        assert beta == beta_expected
        assert gamma == gamma_expected

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
        assert calibrator.min_zoom == min_zoom_expected
        assert calibrator.max_zoom == max_zoom_expected

    def test_calibration_callback(self, calibrator, additional_info_msg):
        """Test calibration callback reads message, calculates alpha, beta, and gamma correctly, and updates those values."""

        _client = None
        _userdata = None
        calibrator._calibration_callback(_client, _userdata, additional_info_msg)

        assert calibrator.alpha == alpha_expected
        assert calibrator.beta == beta_expected
        assert calibrator.gamma == gamma_expected
