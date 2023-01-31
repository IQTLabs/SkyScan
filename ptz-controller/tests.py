import math
from pathlib import Path
import pytest

import numpy as np
import pandas as pd
import quaternion

import ptz_controller
import ptz_utilities

PRECISION = 1e-12
RELATIVE_DIFFERENCE = 2  # %
ANGULAR_DIFFERENCE = 1  # [deg]


def qnorm(q):
    """Compute the quaternion norm."""
    return math.sqrt((q * q.conjugate()).w)


class TestPtzController:
    """Test construction of rotations and calculation of camera
    pointing."""

    def test_compute_tripod_position(self):
        """Tested implicitly."""
        pass

    def test_compute_topocentric_dcm(self):
        """Tested implicitly."""
        pass

    def test_compute_camera_rotations(self):
        """Test construction of rotations."""
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

        # Perform some mental rotation gymnastics (hint: use paper)
        q_alpha_exp = ptz_utilities.as_rotation_quaternion(
            90.0, np.array([0.0, 0.0, -1.0])
        )  # About -w or -Z
        q_beta_exp = ptz_utilities.as_rotation_quaternion(
            90.0, np.array([0.0, -1.0, 0.0])
        )  # About u_alpha or -Y
        q_gamma_exp = ptz_utilities.as_rotation_quaternion(
            90.0, np.array([0.0, 0.0, 1.0])
        )  # About v_beta_alpha or Z
        E_XYZ_to_uvw_exp = np.array(
            [[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]]  # [X, Z, -Y]
        )
        q_rho_exp = ptz_utilities.as_rotation_quaternion(
            90.0, np.array([0.0, 1.0, 0.0])
        )  # About -t_gamma_beta_alpha or Y
        q_tau_exp = ptz_utilities.as_rotation_quaternion(
            90.0, np.array([0.0, 0.0, -1.0])
        )  # About r_rho_gamma_beta_alpha or -Z
        E_XYZ_to_rst_exp = np.array(
            [[0.0, 0.0, -1.0], [0.0, -1.0, 0.0], [-1.0, 0.0, 0.0]]  # [-Z, -N, -E]
        )

        controller = ptz_controller.PtzController(
            config_topic="skyscan/config/json",
            calibration_topic="skyscan/calibration/json",
            flight_topic="skyscan/flight/json",
            lead_time="0.25",
            update_interval="0.10",
            gain_pan="0.2",
            gain_tilt="0.2",
            mqtt_ip="127.0.0.1",
            test=True,
        )
        _client = None
        _userdata = None
        msg = {}
        msg["data"] = {}
        msg["data"]["tripod_latitude"] = 90.0
        msg["data"]["tripod_longitude"] = 270.0
        msg["data"]["tripod_altitude"] = 0.0
        controller._config_callback(_client, _userdata, msg)
        controller._compute_tripod_position()
        controller._compute_topocentric_dcm()
        (
            q_alpha_act,
            q_beta_act,
            q_gamma_act,
            E_XYZ_to_uvw_act,
            q_rho_act,
            q_tau_act,
            E_XYZ_to_rst_act,
        ) = controller._compute_camera_rotations(alpha, beta, gamma, rho, tau)

        assert qnorm(q_alpha_act - q_alpha_exp) < PRECISION
        assert qnorm(q_beta_act - q_beta_exp) < PRECISION
        assert qnorm(q_gamma_act - q_gamma_exp) < PRECISION
        assert np.linalg.norm(E_XYZ_to_uvw_act - E_XYZ_to_uvw_exp) < PRECISION
        assert qnorm(q_rho_act - q_rho_exp) < PRECISION
        assert qnorm(q_tau_act - q_tau_exp) < PRECISION
        assert np.linalg.norm(E_XYZ_to_rst_act - E_XYZ_to_rst_exp) < PRECISION

    def test_assign_pointing(self):
        """TODO: Complete"""
        pass

    def test_update_pointing(self):
        """TODO: Complete"""
        pass


def R_pole():
    """Compute the semi-minor axis of the geoid"""
    f = 1.0 / ptz_utilities.F_INV
    N_pole = ptz_utilities.R_OPLUS / math.sqrt(1.0 - f * (2.0 - f))
    return (1 - f) ** 2 * N_pole


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

    # Compute the great-circle distance between two points on a sphere
    # separated by a quarter circumference
    @pytest.mark.parametrize(
        "varphi_1, lambda_1, varphi_2, lambda_2, d_exp",
        [
            (0.0, 0.0, 0.0, 90.0, math.pi * ptz_utilities.R_OPLUS / 2.0),
            (0.0, 0.0, 90.0, 0.0, math.pi * ptz_utilities.R_OPLUS / 2.0),
        ],
    )
    def test_great_circle_distance(self, varphi_1, lambda_1, varphi_2, lambda_2, d_exp):
        d_act = ptz_utilities.compute_great_circle_distance(
            varphi_1, lambda_1, varphi_2, lambda_2
        )
        assert math.fabs((d_act - d_exp) / d_exp) < PRECISION
