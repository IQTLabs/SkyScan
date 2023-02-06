import os

import pytest
import json

import auto_calibrator

@pytest.fixture
def calibrator():
    calibrator = auto_calibrator.AutoCalibrator(
        env_variable=os.environ.get("ENV_VARIABLE"),
        calibration_topic=str(os.environ.get("CALIBRATION_TOPIC")),
        config_topic=os.environ.get("CONFIG_TOPIC"),
        mqtt_ip=os.environ.get("MQTT_IP"),
        debug=True
    )
    return calibrator

@pytest.fixture
def config_msg():
    # Load test message
    with open("data/config_msg.json") as f:
        msg = json.load(f)
    return msg

@pytest.fixture
def calibration_msg():
    # Load test message
    with open("data/calibration_msg.json") as f:
        msg = json.load(f)
    return msg

@pytest.fixture
def calibration_data(calibration_msg):
    # Load data
    return calibration_msg["data"]

@pytest.fixture
def additional_info_msg():
    # Load test message
    with open("data/additional_info_msg.json") as f:
        msg = json.load(f)
    return msg

@pytest.fixture
def additional_data(additional_info_msg):
    # Load data
    return additional_info_msg["data"]


class TestAutoCalibrator:
    """Test construction of rotations and calculation of camera
    pointing."""

    def test_calculate_calibration_error(self, calibrator, calibration_data):
        """Test calculation of calibration error"""

        rho_0, tau_0, rho_epsilon, tau_epsilon = calibrator._calculate_calibration_error(
            calibration_data)

        assert rho_0 == -47.617992362840106
        assert tau_0 == 1.500321537077791
        assert rho_epsilon == -1.449921875
        assert tau_epsilon == -3.032962962962963

    def test_calculate_pointing_error(self):
        """Tested implicitly."""
        pass

    def test_minimize(self, calibrator, additional_data):
        """Test Broyden, Fletcher, Goldfarb and Shanno minimization algorithm for pointing error method"""

        rho_0 = -47.617992362840106
        tau_0 = 1.500321537077791
        rho_epsilon = -1.449921875
        tau_epsilon = -3.032962962962963

        alpha, beta, gamma = calibrator._minimize(additional_data, rho_0, tau_0, rho_epsilon, tau_epsilon)

        assert alpha == 96.22945929035237
        assert beta == 31.55893394983606
        assert gamma == 1.5230141040882903

    def test_config_callback(self, calibrator, config_msg):
        """Test config callback updates values"""

        _client = None
        _userdata = None
        calibrator._config_callback(_client, _userdata, config_msg)

        assert calibrator.alpha == 1.0
        assert calibrator.beta == 1.0
        assert calibrator.gamma == 1.0

    def test_calibration_callback(self, calibrator, additional_info_msg):
        """Test calibration callback updates values"""

        _client = None
        _userdata = None
        calibrator._calibration_callback(_client, _userdata, additional_info_msg)

        assert calibrator.alpha == 96.22945929035237
        assert calibrator.beta == 31.55893394983606
        assert calibrator.gamma == 1.5230141040882903

