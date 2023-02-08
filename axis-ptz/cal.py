import utils

import math

def p_current(msg):
    rho_0 = msg["rho"]
    tau_0 = msg["tau"]
    # Use YOLO or equiv to find pointing error
    rho_epsilon
    tau_epsilon
    return rho_0, tau_0, rho_epsilon, tau_epsilon


def p_epsilon(alpha, beta, gamma, ...  # Independent vars
              msg, rho_0, tau_0, rho_epsilon, tau_epsion):  # Parameters
    
    # Compute position of the aircraft
    a_varphi = msg["lat"]  # [deg]
    a_lambda = msg["lon"]  # [deg]
    a_h = msg["altitude"]  # [m]
    r_XYZ_a = utils.compute_r_XYZ(a_lambda, a_varphi, a_h)

    # Compute position of the tripod
    t_varphi = camera_latitude  # [deg]
    t_lambda = camera_longitude  # [deg]
    t_h = camera_altitude  # [m]
    r_XYZ_t = utils.compute_r_XYZ(t_lambda, t_varphi, t_h)

    # Compute orthogonal transformation matrix from geocentric to
    # topocentric coordinates, and corresponding unit vectors
    # system of the tripod
    E_XYZ_to_ENz, e_E_XYZ, e_N_XYZ, e_z_XYZ = utils.compute_E(t_lambda, t_varphi)

    # Compute the rotations from the XYZ coordinate system to the uvw
    # (camera housing fixed) coordinate system
    alpha = 0.0  # [deg]
    beta = 0.0  # [deg]
    gamma = 0.0  # [deg]
    _, _, _, E_XYZ_to_uvw, _, _, _ = compute_rotations(
        e_E_XYZ, e_N_XYZ, e_z_XYZ, alpha, beta, gamma, 0.0, 0.0
    )

    # Compute position in the uvw coordinate system of the aircraft
    # relative to the tripod
    r_uvw_a_t = np.matmul(E_XYZ_to_uvw, r_XYZ_a - r_XYZ_t)

    # Compute pan and tilt to point the camera at the aircraft given
    # the updated values of alpha, beta, and gamma
    rho = math.degrees(math.atan2(r_uvw_a_t[0], r_uvw_a_t[1]))  # [deg]
    tau = math.degrees(
        math.atan2(r_uvw_a_t[2], utils.norm(r_uvw_a_t[0:2]))
    )  # [deg]

    # Return the pointing error to be minimized
    return math.sqrt((rho_0 + rho_epsilon - rho)**2 + (tau_0 + tau_epsilon - tau)**2)

def minimize():
    bfgs(p_epsilon)
