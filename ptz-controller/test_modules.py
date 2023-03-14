import json
import math
import os

import numpy as np
import pytest
import quaternion
import ptz_controller
import ptz_utilities


PRECISION = 1.0e-12
RELATIVE_DIFFERENCE = 2.0  # %
ANGULAR_DIFFERENCE = 1.0  # [deg]

# Aligns ENz with XYZ
LAMBDA_T = 270.0  # [deg]
VARPHI_T = 90.0  # [deg]
H_T = 0.0  # [m]

# Places aircraft south of tripod
LAMBDA_A = 270.0  # [deg]
VARPHI_A = 89.99  # [deg]
H_A = 1000.0  # [m]
AIR_SPEED = 100.0  # [m/s]

HEARTBEAT_INTERVAL = 10.0
UPDATE_INTERVAL = 0.10
CAPTURE_INTERVAL = 2.0
LEAD_TIME = 0.0
PAN_GAIN = 0.2
PAN_RATE_MIN = 1.0
PAN_RATE_MAX = 100.0
TILT_GAIN = 0.2
TILT_RATE_MIN = 1.0
TILT_RATE_MAX = 100.0
JPEG_RESOLUTION = "1920x1080"
JPEG_COMPRESSION = 5


def qnorm(q):
    """Compute the quaternion norm."""
    return math.sqrt((q * q.conjugate()).w)


def R_pole():
    """Compute the semi-minor axis of the geoid."""
    f = 1.0 / ptz_utilities.F_INV
    N_pole = ptz_utilities.R_OPLUS / math.sqrt(1.0 - f * (2.0 - f))
    return (1.0 - f) ** 2 * N_pole


@pytest.fixture
def controller():
    """Construct a controller."""
    controller = ptz_controller.PtzController(
        camera_ip=os.getenv("CAMERA_IP"),
        camera_user=os.getenv("CAMERA_USER"),
        camera_password=os.getenv("CAMERA_PASSWORD"),
        mqtt_ip=os.getenv("MQTT_IP"),
        config_topic=os.getenv("CONFIG_TOPIC"),
        calibration_topic=os.getenv("CALIBRATION_TOPIC"),
        flight_topic=os.getenv("FLIGHT_TOPIC"),
        capture_topic=os.getenv("CAPTURE_TOPIC"),
        logger_topic=os.getenv("LOGGER_TOPIC"),
        heartbeat_interval=HEARTBEAT_INTERVAL,
        update_interval=UPDATE_INTERVAL,
        capture_interval=CAPTURE_INTERVAL,
        lead_time=LEAD_TIME,
        pan_gain=PAN_GAIN,
        pan_rate_min=PAN_RATE_MIN,
        pan_rate_max=PAN_RATE_MAX,
        tilt_gain=TILT_GAIN,
        tilt_rate_min=TILT_RATE_MIN,
        tilt_rate_max=TILT_RATE_MAX,
        jpeg_resolution=JPEG_RESOLUTION,
        jpeg_compression=JPEG_COMPRESSION,
        use_mqtt=False,
        use_camera=False,
        include_age=False,
    )
    return controller


@pytest.fixture
def config_msg():
    """Populate a config message."""
    with open("data/config_msg.json", "r") as f:
        msg = json.load(f)
    return msg


@pytest.fixture
def calibration_msg_0s():
    """Populate a calibration message with all 0 deg angles."""
    with open("data/calibration_msg_0s.json", "r") as f:
        msg = json.load(f)
    return msg


@pytest.fixture
def calibration_msg_90s():
    """Populate a calibration message with all 90 deg angles."""
    with open("data/calibration_msg_90s.json", "r") as f:
        msg = json.load(f)
    return msg


@pytest.fixture
def flight_msg():
    """Populate a flight message with velocity along the line of
    sight, using the calculation noted below.

    # ENz at the tripod aligns with XYZ, and the tripod remains stationary
    r_XYZ_t = ptz_utilities.compute_r_XYZ(LAMBDA_T, VARPHI_T, H_T)
    r_XYZ_a = ptz_utilities.compute_r_XYZ(LAMBDA_A, VARPHI_A, H_A)
    r_XYZ_a_t = r_XYZ_a - r_XYZ_t
    v_ENz_T_a = AIR_SPEED * r_XYZ_a_t / ptz_utilities.norm(r_XYZ_a_t)

    # ENz directly below the aircraft miss-aligns with XYZ slightly
    E_XYZ_to_ENz, _, _, _ = ptz_utilities.compute_E_XYZ_to_ENz(LAMBDA_A, VARPHI_A)
    v_ENz_A_a = np.matmul(E_XYZ_to_ENz, v_ENz_T_a)
    """
    with open("data/flight_msg.json", "r") as f:
        msg = json.load(f)
    return msg


class TestPtzController:
    """Test message callbacks and update method."""

    def test_config_callback(self, controller, config_msg):

        # Align ENz with XYZ
        _client = None
        _userdata = None
        controller._config_callback(_client, _userdata, config_msg)

        # Assign expected values
        r_XYZ_t_exp = np.array([0.0, 0.0, R_pole()])
        E_XYZ_to_ENz_exp = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        e_E_XYZ_exp = np.array([1.0, 0.0, 0.0])
        e_N_XYZ_exp = np.array([0.0, 1.0, 0.0])
        e_z_XYZ_exp = np.array([0.0, 0.0, 1.0])

        # Y component of r_XYZ_t precision lower due to precision of
        # cos(90) and magnitude of R_oplus
        assert controller.lambda_t == LAMBDA_T
        assert controller.varphi_t == VARPHI_T
        assert controller.h_t == H_T
        assert np.linalg.norm(controller.r_XYZ_t - r_XYZ_t_exp) < 400 * PRECISION
        assert np.linalg.norm(controller.E_XYZ_to_ENz - E_XYZ_to_ENz_exp) < PRECISION
        assert np.linalg.norm(controller.e_E_XYZ - e_E_XYZ_exp) < PRECISION
        assert np.linalg.norm(controller.e_N_XYZ - e_N_XYZ_exp) < PRECISION
        assert np.linalg.norm(controller.e_z_XYZ - e_z_XYZ_exp) < PRECISION

    def test_calibration_callback(self, controller, config_msg, calibration_msg_90s):

        # Align ENz with XYZ
        _client = None
        _userdata = None
        controller._config_callback(_client, _userdata, config_msg)

        # Use 90 degree rotations
        controller._calibration_callback(_client, _userdata, calibration_msg_90s)

        # Assign expected values by performing some mental rotation
        # gymnastics
        alpha = 90.0
        q_alpha_exp = ptz_utilities.as_rotation_quaternion(
            alpha, np.array([0.0, 0.0, -1.0])
        )  # About -w or -Z
        beta = 90.0
        q_beta_exp = ptz_utilities.as_rotation_quaternion(
            beta, np.array([0.0, -1.0, 0.0])
        )  # About u_alpha or -Y
        gamma = 90.0
        q_gamma_exp = ptz_utilities.as_rotation_quaternion(
            gamma, np.array([0.0, 0.0, 1.0])
        )  # About v_beta_alpha or Z
        E_XYZ_to_uvw_exp = np.array(
            [[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]]  # [X, Z, -Y]
        )

        assert qnorm(controller.q_alpha - q_alpha_exp) < PRECISION
        assert qnorm(controller.q_beta - q_beta_exp) < PRECISION
        assert qnorm(controller.q_gamma - q_gamma_exp) < PRECISION
        assert np.linalg.norm(controller.E_XYZ_to_uvw - E_XYZ_to_uvw_exp) < PRECISION

    def test_compute_pan_rate_index(self, controller):
        assert controller._compute_pan_rate_index(PAN_RATE_MIN / 2.0) == -100
        assert controller._compute_pan_rate_index((PAN_RATE_MAX + PAN_RATE_MIN) / 2.0) == 0
        assert controller._compute_pan_rate_index(PAN_RATE_MAX * 2.0) == +100

    def test_compute_tilt_rate_index(self, controller):
        assert controller._compute_tilt_rate_index(TILT_RATE_MIN / 2.0) == -100
        assert controller._compute_tilt_rate_index((TILT_RATE_MAX + TILT_RATE_MIN) / 2.0) == 0
        assert controller._compute_tilt_rate_index(TILT_RATE_MAX * 2.0) == +100

    def test_flight_callback(
        self, controller, config_msg, calibration_msg_0s, flight_msg
    ):

        # Align ENz with XYZ
        _client = None
        _userdata = None
        controller._config_callback(_client, _userdata, config_msg)

        # Use 0 degree rotations so uvw aligns with XYZ
        controller._calibration_callback(_client, _userdata, calibration_msg_0s)

        # Align velocity along the line of sight
        controller._flight_callback(_client, _userdata, flight_msg)

        # Compute expected values
        r_uvw_t = ptz_utilities.compute_r_XYZ(LAMBDA_T, VARPHI_T, H_T)
        r_uvw_a = ptz_utilities.compute_r_XYZ(LAMBDA_A, VARPHI_A, H_A)
        r_uvw_a_t = r_uvw_a - r_uvw_t
        # Expect +/-180.0
        rho_a_exp = math.fabs(math.degrees(math.atan2(r_uvw_a_t[0], r_uvw_a_t[1])))
        tau_a_exp = math.degrees(
            math.atan2(r_uvw_a_t[2], math.sqrt(r_uvw_a_t[0] ** 2 + r_uvw_a_t[1] ** 2))
        )
        delta_rho_dot_c_exp = PAN_GAIN * rho_a_exp  # Since rho_c = 0.0
        delta_tau_dot_c_exp = TILT_GAIN * tau_a_exp  # Since tau_c = 0.0
        r_rst_a_0_t = np.array([0.0, ptz_utilities.norm(r_uvw_a_t), 0.0])
        v_rst_a_0_t = np.array([0.0, AIR_SPEED, 0.0])

        assert math.fabs(controller.rho_a) == rho_a_exp
        assert controller.tau_a == tau_a_exp
        assert controller.delta_rho_dot_c == delta_rho_dot_c_exp
        assert controller.delta_tau_dot_c == delta_tau_dot_c_exp
        assert np.linalg.norm(controller.r_rst_a_0_t - r_rst_a_0_t) < PRECISION
        # Magnitude of velocity difference less than 0.02% of velocity magnitude
        assert (
            100
            * np.linalg.norm(controller.v_rst_a_0_t - v_rst_a_0_t)
            / np.linalg.norm(v_rst_a_0_t)
            < RELATIVE_DIFFERENCE / 100
        )


class TestPtzUtilities:
    """Test construction of directions, a corresponding direction
    cosine matrix, and quaternions."""

    # Rotate east through Y, -X, and -Y
    @pytest.mark.parametrize(
        "o_lambda, e_E_XYZ_exp",
        [
            (0.0, np.array([0.0, 1.0, 0.0])),
            (90.0, np.array([-1.0, 0.0, 0.0])),
            (180.0, np.array([0.0, -1.0, 0.0])),
        ],
    )
    def test_compute_e_E_XYZ(self, o_lambda, e_E_XYZ_exp):
        e_E_XYZ_act = ptz_utilities.compute_e_E_XYZ(o_lambda)
        assert np.linalg.norm(e_E_XYZ_act - e_E_XYZ_exp) < PRECISION

    # Rotate north through Z, Z, and between -X and Z
    @pytest.mark.parametrize(
        "o_lambda, o_varphi, e_N_XYZ_exp",
        [
            (0.0, 0.0, np.array([0.0, 0.0, 1.0])),
            (90.0, 0.0, np.array([0.0, 0.0, 1.0])),
            (
                0.0,
                45.0,
                np.array([-1.0 / math.sqrt(2.0), 0.0, 1.0 / math.sqrt(2.0)]),
            ),
        ],
    )
    def test_compute_e_N_XYZ(self, o_lambda, o_varphi, e_N_XYZ_exp):
        e_N_XYZ_act = ptz_utilities.compute_e_N_XYZ(o_lambda, o_varphi)
        assert np.linalg.norm(e_N_XYZ_act - e_N_XYZ_exp) < PRECISION

    # Rotate zenith through X, Y, and Z
    @pytest.mark.parametrize(
        "o_lambda, o_varphi, e_z_XYZ_exp",
        [
            (0.0, 0.0, np.array([1.0, 0.0, 0.0])),
            (90.0, 0.0, np.array([0.0, 1.0, 0.0])),
            (0.0, 90.0, np.array([0.0, 0.0, 1.0])),
        ],
    )
    def test_compute_e_z_XYZ(self, o_lambda, o_varphi, e_z_XYZ_exp):
        e_z_XYZ_act = ptz_utilities.compute_e_z_XYZ(o_lambda, o_varphi)
        assert np.linalg.norm(e_z_XYZ_act - e_z_XYZ_exp) < PRECISION

    # Rotate zenith to between X, Y, and Z
    @pytest.mark.parametrize(
        "o_lambda, o_varphi, E_exp",
        [
            (
                45.0,
                45.0,
                np.array(
                    [
                        [-1.0 / math.sqrt(2.0), 1.0 / math.sqrt(2), 0.0],
                        [-0.5, -0.5, 1.0 / math.sqrt(2)],
                        [0.5, 0.5, 1.0 / math.sqrt(2)],
                    ]
                ),
            ),
        ],
    )
    def test_compute_E_XYZ_to_ENz(self, o_lambda, o_varphi, E_exp):
        E_act, _, _, _ = ptz_utilities.compute_E_XYZ_to_ENz(o_lambda, o_varphi)
        assert np.linalg.norm(E_act - E_exp) < PRECISION

    # Compute two positions at the equator, and one at the pole
    @pytest.mark.parametrize(
        "o_lambda, o_varphi, o_h, r_XYZ_exp",
        [
            (0.0, 0.0, 0.0, np.array([ptz_utilities.R_OPLUS, 0.0, 0.0])),
            (90.0, 0.0, 0.0, np.array([0.0, ptz_utilities.R_OPLUS, 0.0])),
            (0.0, 90.0, 0.0, np.array([0.0, 0.0, R_pole()])),
        ],
    )
    def test_compute_r_XYZ(self, o_lambda, o_varphi, o_h, r_XYZ_exp):
        r_XYZ_act = ptz_utilities.compute_r_XYZ(o_lambda, o_varphi, o_h)
        # Decrease precision to accommodate R_OPLUS [ft]
        assert np.linalg.norm(r_XYZ_act - r_XYZ_exp) < 10000 * PRECISION

    # Construct quaternions from a numpy.ndarray
    @pytest.mark.parametrize(
        "s, v, q_exp",
        [
            (0.0, np.array([1.0, 2.0, 3.0]), np.quaternion(0.0, 1.0, 2.0, 3.0)),
        ],
    )
    def test_as_quaternion(self, s, v, q_exp):
        q_act = ptz_utilities.as_quaternion(s, v)
        assert np.equal(q_act, q_exp).any()

    # Construct rotation quaternions from numpy.ndarrays
    @pytest.mark.parametrize(
        "s, v, r_exp",
        [
            (0.0, np.array([1.0, 2.0, 3.0]), np.quaternion(1.0, 0.0, 0.0, 0.0)),
            (180.0, np.array([1.0, 2.0, 3.0]), np.quaternion(0.0, 1.0, 2.0, 3.0)),
        ],
    )
    def test_as_rotation_quaternion(self, s, v, r_exp):
        r_act = ptz_utilities.as_rotation_quaternion(s, v)
        assert qnorm(r_act - r_exp) < PRECISION

    # Get the vector part of a vector quaternion
    @pytest.mark.parametrize(
        "q, v_exp",
        [
            (np.quaternion(0.0, 1.0, 2.0, 3.0), np.array([1.0, 2.0, 3.0])),
        ],
    )
    def test_as_vector(self, q, v_exp):
        v_act = ptz_utilities.as_vector(q)
        assert np.equal(v_act, v_exp).any()

    # Compute the cross product of two vectors
    def test_cross(self):
        u = np.array([2.0, 3.0, 4.0])
        v = np.array([3.0, 4.0, 5.0])
        w_exp = np.array([-1, 2, -1])
        w_act = ptz_utilities.cross(u, v)
        assert np.equal(w_act, w_exp).any()

        # Test using external package
        w_npq = np.cross(u, v)
        assert np.equal(w_npq, w_exp).any()

    # Compute the Euclidean norm of a vector
    def test_norm(self):
        v = np.array([3.0, 4.0, 5.0])
        n_exp = math.sqrt(50)
        n_act = ptz_utilities.norm(v)
        assert n_exp == n_act

    # Compute camera rotations aligning ENz with XYZ and using only 90
    # deg rotations
    def test_compute_camera_rotations(self):

        # Align ENz with XYZ
        o_lambda = 270.0
        o_varphi = 90.0
        e_E_XYZ = ptz_utilities.compute_e_E_XYZ(o_lambda)  # [1, 0, 0] or X
        e_N_XYZ = ptz_utilities.compute_e_N_XYZ(o_lambda, o_varphi)  # [0, 1, 0] or Y
        e_z_XYZ = ptz_utilities.compute_e_z_XYZ(o_lambda, o_varphi)  # [0, 0, 1] or Z

        # Only use 90 degree rotations
        alpha = 90.0
        beta = 90.0
        gamma = 90.0
        rho = 90.0
        tau = 90.0

        # Perform some mental rotation gymnastics
        q_alpha_exp = ptz_utilities.as_rotation_quaternion(
            alpha, np.array([0.0, 0.0, -1.0])
        )  # About -w or -Z
        q_beta_exp = ptz_utilities.as_rotation_quaternion(
            beta, np.array([0.0, -1.0, 0.0])
        )  # About u_alpha or -Y
        q_gamma_exp = ptz_utilities.as_rotation_quaternion(
            gamma, np.array([0.0, 0.0, 1.0])
        )  # About v_beta_alpha or Z
        E_XYZ_to_uvw_exp = np.array(
            [[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]]  # [X, Z, -Y]
        )
        q_rho_exp = ptz_utilities.as_rotation_quaternion(
            rho, np.array([0.0, 1.0, 0.0])
        )  # About -t_gamma_beta_alpha or Y
        q_tau_exp = ptz_utilities.as_rotation_quaternion(
            tau, np.array([0.0, 0.0, -1.0])
        )  # About r_rho_gamma_beta_alpha or -Z
        E_XYZ_to_rst_exp = np.array(
            [[0.0, 0.0, -1.0], [0.0, -1.0, 0.0], [-1.0, 0.0, 0.0]]  # [-Z, -N, -E]
        )

        # Compute and compare the camera rotations
        (
            q_alpha_act,
            q_beta_act,
            q_gamma_act,
            E_XYZ_to_uvw_act,
            q_rho_act,
            q_tau_act,
            E_XYZ_to_rst_act,
        ) = ptz_utilities.compute_camera_rotations(
            e_E_XYZ, e_N_XYZ, e_z_XYZ, alpha, beta, gamma, rho, tau
        )
        assert qnorm(q_alpha_act - q_alpha_exp) < PRECISION
        assert qnorm(q_beta_act - q_beta_exp) < PRECISION
        assert qnorm(q_gamma_act - q_gamma_exp) < PRECISION
        assert np.linalg.norm(E_XYZ_to_uvw_act - E_XYZ_to_uvw_exp) < PRECISION
        assert qnorm(q_rho_act - q_rho_exp) < PRECISION
        assert qnorm(q_tau_act - q_tau_exp) < PRECISION
        assert np.linalg.norm(E_XYZ_to_rst_act - E_XYZ_to_rst_exp) < PRECISION

    # Compute the great-circle distance between two points on a sphere
    # separated by a quarter circumference
    @pytest.mark.parametrize(
        "lambda_1, varphi_1, lambda_2, varphi_2, d_exp",
        [
            (0.0, 0.0, 90.0, 0.0, math.pi * ptz_utilities.R_OPLUS / 2.0),
            (0.0, 0.0, 0.0, 90.0, math.pi * ptz_utilities.R_OPLUS / 2.0),
        ],
    )
    def test_great_circle_distance(self, lambda_1, varphi_1, lambda_2, varphi_2, d_exp):
        d_act = ptz_utilities.compute_great_circle_distance(
            lambda_1, varphi_1, lambda_2, varphi_2
        )
        assert math.fabs((d_act - d_exp) / d_exp) < PRECISION
