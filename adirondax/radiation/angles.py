import numpy as np

# Pure functions for the discrete ordinates (angular grid) of the radiation field
#
# Each direction n = (sin(zeta) cos(psi), cos(zeta), sin(zeta) sin(psi)) is
# written in the (x, y, out-of-plane) basis of the 2D mesh, which in
# cylindrical geometry is the local (R, z, phi) basis of each cell. Both
# geometries are mirror symmetric through the mesh plane (z -> -z in cartesian,
# phi -> -phi in axisymmetry), which maps psi -> -psi, so only psi in (0, pi)
# is evolved and each direction stands for itself and its mirror image.


def get_angles(num_polar, num_azimuthal):
    """
    Build the angular grid: a Gauss-Legendre grid in mu_y = cos(zeta) times
    a uniform midpoint grid in psi in (0, pi). Both sizes must be even, so
    that no direction lies in the mesh plane or along an axis.

    The quadrature integrates the zeroth, first and second moments exactly:
    sum(w) = 1, sum(w mu) = 0 and sum(w mu_x^2) = sum(w mu_y^2) = 1/3.

    Returns
    -------
    angles: dict
      'mu_x', 'mu_y', 'mu_z'  direction cosines along x (or R), y (or z) and
                              out of the plane, shape (N,)
      'w'                     quadrature weights, normalized to sum to 1
      'reflect_x'             index of the direction mirrored in x (mu_x -> -mu_x)
      'reflect_y'             index of the direction mirrored in y (mu_y -> -mu_y)
      'sin_zeta'              sin(zeta) of each direction
      'psi_face_lo', 'psi_face_hi'  coefficients of the psi-flux at the lower
                              and upper psi-face of each direction, which turn
                              the local basis as a ray crosses an axisymmetric
                              mesh (see 'angular_redistribution' in rt2d.py)
      'd_psi'                 width of each psi bin
      'num_polar', 'num_azimuthal'
    """

    if num_polar < 2 or num_polar % 2 != 0:
        raise ValueError(f"'num_polar' must be even and >= 2: {num_polar}")
    if num_azimuthal < 2 or num_azimuthal % 2 != 0:
        raise ValueError(f"'num_azimuthal' must be even and >= 2: {num_azimuthal}")

    mu_y, w_polar = np.polynomial.legendre.leggauss(num_polar)
    sin_zeta = np.sqrt(1.0 - mu_y**2)

    d_psi = np.pi / num_azimuthal
    psi = d_psi * (np.arange(num_azimuthal) + 0.5)

    # the midpoint sums of cos(psi)^2 and sin(psi)^2 over (0, pi) are exactly
    # num_azimuthal/2, so the second moments are exact
    mu_y_2d = np.repeat(mu_y, num_azimuthal)
    sin_zeta_2d = np.repeat(sin_zeta, num_azimuthal)
    mu_x = sin_zeta_2d * np.tile(np.cos(psi), num_polar)
    mu_z = sin_zeta_2d * np.tile(np.sin(psi), num_polar)
    w = np.repeat(0.5 * w_polar, num_azimuthal) / num_azimuthal

    # the psi-flux coefficient at the upper face of bin k is the running sum of
    # cos(psi) d_psi, a discrete sin(psi) that vanishes at psi = 0 and psi = pi
    # (by symmetry), so the redistribution conserves the energy exactly and
    # balances the radial flux of an isotropic field to round-off
    face_hi = d_psi * np.cumsum(np.cos(psi))
    face_hi[-1] = 0.0
    face_lo = np.concatenate(([0.0], face_hi[:-1]))

    # mirror images: psi -> pi - psi flips mu_x, zeta -> pi - zeta flips mu_y
    j, k = np.divmod(np.arange(num_polar * num_azimuthal), num_azimuthal)
    reflect_x = j * num_azimuthal + (num_azimuthal - 1 - k)
    reflect_y = (num_polar - 1 - j) * num_azimuthal + k

    return {
        "mu_x": mu_x,
        "mu_y": mu_y_2d,
        "mu_z": mu_z,
        "w": w,
        "reflect_x": reflect_x,
        "reflect_y": reflect_y,
        "sin_zeta": sin_zeta_2d,
        "psi_face_lo": np.tile(face_lo, num_polar),
        "psi_face_hi": np.tile(face_hi, num_polar),
        "d_psi": d_psi,
        "num_polar": num_polar,
        "num_azimuthal": num_azimuthal,
    }
