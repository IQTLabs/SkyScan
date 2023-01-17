import math
from pathlib import Path
import pytest

import numpy as np
import pandas as pd
import quaternion as qn

import camera
import utils

PRECISION = 1e-12
RELATIVE_DIFFERENCE = 2  # %
ANGULAR_DIFFERENCE = 1  # [deg]

class TestCameraModule:
    """Test construction of rotations and calculation of camera
    pointing."""

    def test_compute_rotations(self):
        """Test construction of rotations."""
        # Align ENz with XYZ
        o_lambda = 270.0
        o_varphi = 90.0
        e_E_XYZ = utils.compute_e_E_XYZ(o_lambda)  # [1, 0, 0] or X
        e_N_XYZ = utils.compute_e_N_XYZ(o_lambda, o_varphi)  # [0, 1, 0] or Y
        e_z_XYZ = utils.compute_e_z_XYZ(o_lambda, o_varphi)  # [0, 0, 1] or Z

        # Only use 90 degree rotations
        alpha = 90.0
        beta = 90.0
        gamma = 90.0
        rho = 90.0
        tau = 90.0

        # Perform some mental rotation gymnastics (hint: use paper)
        q_alpha_exp = utils.as_rotation_quaternion(
            90.0, [0.0, 0.0, -1.0]
        )  # About -w or -Z
        q_beta_exp = utils.as_rotation_quaternion(
            90.0, [0.0, -1.0, 0.0]
        )  # About u_alpha or -Y
        q_gamma_exp = utils.as_rotation_quaternion(
            90.0, [0.0, 0.0, 1.0]
        )  # About v_beta_alpha or Z
        E_XYZ_to_uvw_exp = np.array(
            [[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]]  # [X, Z, -Y]
        )
        q_rho_exp = utils.as_rotation_quaternion(
            90.0, [0.0, 1.0, 0.0]
        )  # About -t_gamma_beta_alpha or Y
        q_tau_exp = utils.as_rotation_quaternion(
            90.0, [0.0, 0.0, -1.0]
        )  # About r_rho_gamma_beta_alpha or -Z
        E_XYZ_to_rst_exp = np.array(
            [[0.0, 0.0, -1.0], [0.0, -1.0, 0.0], [-1.0, 0.0, 0.0]]  # [-Z, -N, -E]
        )

        (
            q_alpha_act,
            q_beta_act,
            q_gamma_act,
            E_XYZ_to_uvw_act,
            q_rho_act,
            q_tau_act,
            E_XYZ_to_rst_act,
        ) = camera.compute_rotations(
            e_E_XYZ, e_N_XYZ, e_z_XYZ, alpha, beta, gamma, rho, tau
        )

        def qnorm(q):
            """Compute the quaternion norm."""
            return math.sqrt((q * q.conjugate()).w)

        assert qnorm(q_alpha_act - q_alpha_exp) < PRECISION
        assert qnorm(q_beta_act - q_beta_exp) < PRECISION
        assert qnorm(q_gamma_act - q_gamma_exp) < PRECISION
        assert np.linalg.norm(E_XYZ_to_uvw_act - E_XYZ_to_uvw_exp) < PRECISION
        assert qnorm(q_rho_act - q_rho_exp) < PRECISION
        assert qnorm(q_tau_act - q_tau_exp) < PRECISION
        assert np.linalg.norm(E_XYZ_to_rst_act - E_XYZ_to_rst_exp) < PRECISION

    def test_calculateCameraPositionB(self):
        data = pd.read_csv("data/A19A08-processed-track.csv")

        camera.camera_latitude = 38.0  # [deg]
        camera.camera_longitude = -77.0  # [deg]
        camera.camera_altitude = 86.46  # [m]
        camera.camera_lead = 0.0  # [s]

        for index in range(0, data.shape[0]):
            camera.currentPlane = data.iloc[index, :].to_dict()

            # Convert to specified units of measure
            # currentPlane["lat"]  # [deg]
            # currentPlane["lon"]  # [deg]
            # currentPlane["latLonTime"]
            camera.currentPlane["altitude"] *= 0.3048  # [ft] * [m/ft] = [m]
            # currentPlane["altitudeTime"]
            # currentPlane["track"]  # [deg]
            camera.currentPlane["groundSpeed"] *= (
                6076.12 / 3600 * 0.3048
            )  # [nm/h] * [ft/nm] / [s/h] * [m/ft] = [m/s]
            camera.currentPlane["verticalRate"] *= 0.3048  # [ft/s] * [m/ft] = [m/s]
            # currentPlane["icao24"]
            # currentPlane["type"]

            camera.calculateCameraPositionA()

            distance3dA = camera.distance3d
            distance2dA = camera.distance2d
            bearingA = camera.bearing
            elevationA = camera.elevation
            angularVelocityHorizontalA = camera.angularVelocityHorizontal
            angularVelocityVerticalA = camera.angularVelocityVertical
            cameraPanA = camera.cameraPan
            cameraTiltA = camera.cameraTilt

            camera.calculateCameraPositionB()

            distance3dB = camera.distance3d
            distance2dB = camera.distance2d
            bearingB = camera.bearing
            elevationB = camera.elevation
            angularVelocityHorizontalB = camera.angularVelocityHorizontal
            angularVelocityVerticalB = camera.angularVelocityVertical
            cameraPanB = camera.cameraPan
            cameraTiltB = camera.cameraTilt

            def thetaDifference(d_thetaA, d_thetaB):
                r_thetaA = math.radians(d_thetaA)
                r_thetaB = math.radians(d_thetaB)
                thetaD = math.acos(math.cos(r_thetaB - r_thetaA))
                return math.degrees(thetaD)

            distance3dD = 100 * abs((distance3dB - distance3dA) / distance3dA)
            distance2dD = 100 * abs((distance2dB - distance2dA) / distance2dA)
            bearingD = thetaDifference(bearingB, bearingA)
            elevationD = thetaDifference(elevationB, elevationA)
            angularVelocityHorizontalD = 100 * abs(
                (angularVelocityHorizontalB - angularVelocityHorizontalA)
                / angularVelocityHorizontalA
            )
            angularVelocityVerticalD = 100 * abs(
                (angularVelocityVerticalB - angularVelocityVerticalA)
                / angularVelocityVerticalA
            )
            cameraPanD = thetaDifference(cameraPanB, cameraPanA)
            cameraTiltD = thetaDifference(cameraTiltB, cameraTiltA)

            assert distance3dD < RELATIVE_DIFFERENCE
            assert distance2dD < RELATIVE_DIFFERENCE
            # Computation of bearing differs substantially
            assert bearingD < 60 * ANGULAR_DIFFERENCE
            assert elevationD < ANGULAR_DIFFERENCE
            # Computation of angular velocity differs very substantially
            assert angularVelocityHorizontalD < 100 * RELATIVE_DIFFERENCE
            assert angularVelocityVerticalD < 100 * RELATIVE_DIFFERENCE
            # Pan and tilt agree within a reasonable difference
            assert cameraPanD < ANGULAR_DIFFERENCE
            assert cameraTiltD < ANGULAR_DIFFERENCE


def R_pole():
    """Compute the semi-minor axis of the geoid"""
    f = 1.0 / utils.F_INV
    N_pole = utils.R_OPLUS / math.sqrt(1.0 - f * (2.0 - f))
    return (1 - f) ** 2 * N_pole


class TestUtilsModule:
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
        e_E_XYZ_act = utils.compute_e_E_XYZ(o_lambda)
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
        e_N_XYZ_act = utils.compute_e_N_XYZ(o_lambda, o_varphi)
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
        e_z_XYZ_act = utils.compute_e_z_XYZ(o_lambda, o_varphi)
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
    def test_compute_E(self, o_lambda, o_varphi, E_exp):
        E_act, _, _, _ = utils.compute_E(o_lambda, o_varphi)
        assert np.linalg.norm(E_act - E_exp) < PRECISION

    # Compute two positions at the equator, and one at the pole
    @pytest.mark.parametrize(
        "o_lambda, o_varphi, o_h, r_XYZ_exp",
        [
            (0.0, 0.0, 0.0, np.array([utils.R_OPLUS, 0.0, 0.0])),
            (90.0, 0.0, 0.0, np.array([0.0, utils.R_OPLUS, 0.0])),
            (0.0, 90.0, 0.0, np.array([0.0, 0.0, R_pole()])),
        ],
    )
    def test_compute_r_XYZ(self, o_lambda, o_varphi, o_h, r_XYZ_exp):
        r_XYZ_act = utils.compute_r_XYZ(o_lambda, o_varphi, o_h)
        # Decrease precision to accommodate R_OPLUS [ft]
        assert np.linalg.norm(r_XYZ_act - r_XYZ_exp) < 10000 * PRECISION

    # Construct quaternions from a list or numpy.ndarray
    @pytest.mark.parametrize(
        "s, v, q_exp",
        [
            (0.0, [1.0, 2.0, 3.0], np.quaternion(0.0, 1.0, 2.0, 3.0)),
            (0.0, np.array([1.0, 2.0, 3.0]), np.quaternion(0.0, 1.0, 2.0, 3.0)),
        ],
    )
    def test_as_quaternion(self, s, v, q_exp):
        q_act = utils.as_quaternion(s, v)
        assert q_act.equal(q_exp)

    # Raise an exception when constructing quaternions without floats,
    # or three dimensional vectors
    @pytest.mark.parametrize(
        "s, v, q_exp",
        [
            (0, [1.0, 2.0, 3.0], np.quaternion(0.0, 1.0, 2.0, 3.0)),
            (0.0, [1, 2.0, 3.0], np.quaternion(0.0, 1.0, 2.0, 3.0)),
            (0.0, [1.0, 2.0], np.quaternion(0.0, 1.0, 2.0, 3.0)),
        ],
    )
    def test_as_quaternion_with_exception(self, s, v, q_exp):
        with pytest.raises(Exception):
            q_act = utils.as_quaternion(s, v)

    # Construct rotation quaternions from a list or numpy.ndarray
    @pytest.mark.parametrize(
        "s, v, r_exp",
        [
            (0.0, [1.0, 2.0, 3.0], np.quaternion(0.0, 1.0, 2.0, 3.0)),
            (0.0, np.array([1.0, 2.0, 3.0]), np.quaternion(0.0, 1.0, 2.0, 3.0)),
        ],
    )
    def test_as_rotation_quaternion(self, s, v, r_exp):
        r_act = utils.as_rotation_quaternion(s, v)
        assert r_act.equal(r_exp)

    # Raise an exception when constructing rotation quaternions
    # without floats, or three dimensional vectors
    @pytest.mark.parametrize(
        "s, v, r_exp",
        [
            (0, [1.0, 2.0, 3.0], np.quaternion(0.0, 1.0, 2.0, 3.0)),
            (0.0, [1, 2.0, 3.0], np.quaternion(0.0, 1.0, 2.0, 3.0)),
            (0.0, [1.0, 2.0], np.quaternion(0.0, 1.0, 2.0, 3.0)),
        ],
    )
    def test_as_rotation_quaternion(self, s, v, r_exp):
        with pytest.raises(Exception):
            r_act = utils.as_rotation_quaternion(s, v)

    # Get the vector part of a vector quaternion
    @pytest.mark.parametrize(
        "q, v_exp",
        [
            (np.quaternion(0.0, 1.0, 2.0, 3.0), np.array([1.0, 2.0, 3.0])),
        ],
    )
    def test_as_vector(self, q, v_exp):
        v_act = utils.as_vector(q)
        assert np.linalg.norm(v_act - v_act) < PRECISION

    # Raise an exception when getting the vector part of a quaternion
    # that is not a vector quaternion
    @pytest.mark.parametrize(
        "q, v_exp",
        [
            (np.quaternion(1.1e-12, 1.0, 2.0, 3.0), np.array([1.0, 2.0, 3.0])),
        ],
    )
    def test_as_vector(self, q, v_exp):
        with pytest.raises(Exception):
            v_act = utils.as_vector(q)

    # Compute the great-circle distance between two points on a sphere
    @pytest.mark.parametrize(
        "varphi_1, lambda_1, varphi_2, lambda_2, d_exp",
        [
            (0.0, 0.0, 0.0, 90.0, math.pi * utils.R_OPLUS / 2.0),
            (0.0, 0.0, 90.0, 0.0, math.pi * utils.R_OPLUS / 2.0),
        ],
    )
    def test_great_circle_distance(self, varphi_1, lambda_1, varphi_2, lambda_2, d_exp):
        d_act = utils.compute_great_circle_distance(varphi_1, lambda_1, varphi_2, lambda_2)
        assert math.fabs((d_act - d_exp) / d_exp) < PRECISION
