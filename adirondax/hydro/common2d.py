import jax.numpy as jnp

# Pure functions for 2D hydrodynamics

_SWAP_XY = {
    "vx": "vy",
    "vy": "vx",
    "Bx": "By",
    "By": "Bx",
    "momx": "momy",
    "momy": "momx",
}


def get_curl(Az, dx, dy):
    """
    Calculate the discrete curl
    """

    bx = (Az - jnp.roll(Az, 1, axis=1)) / dy  # = d Az / d y
    by = -(Az - jnp.roll(Az, 1, axis=0)) / dx  # =-d Az / d x

    return bx, by


def get_div(bx, by, dx, dy):
    """
    Calculate the discrete divergence
    """

    div_B = (bx - jnp.roll(bx, 1, axis=0)) / dx + (by - jnp.roll(by, 1, axis=1)) / dy

    return div_B


def get_avg(bx, by):
    """
    Calculate the volume-averaged magnetic field
    """

    Bx = 0.5 * (bx + jnp.roll(bx, 1, axis=0))
    By = 0.5 * (by + jnp.roll(by, 1, axis=1))

    return Bx, By


def get_gradient(f, dx, dy):
    """Calculate the gradients of a field"""

    # (right - left) / (2*dx)
    f_dx = (jnp.roll(f, -1, axis=0) - jnp.roll(f, 1, axis=0)) / (2.0 * dx)
    f_dy = (jnp.roll(f, -1, axis=1) - jnp.roll(f, 1, axis=1)) / (2.0 * dy)

    return f_dx, f_dy


def get_gradients(W, dx, dy, use_slope_limiting):
    """Calculate the (optionally slope-limited) x and y gradients of a dict of fields"""

    W_d = {name: get_gradient(f, dx, dy) for name, f in W.items()}
    if use_slope_limiting:
        W_d = {name: slope_limit(W[name], *W_d[name], dx, dy) for name in W}

    return {k: d[0] for k, d in W_d.items()}, {k: d[1] for k, d in W_d.items()}


def slope_limit(f, f_dx, f_dy, dx, dy):
    """
    Apply slope limiter to slopes
    """
    denom = (f_dx + 1.0e-8 * (f_dx == 0)) * dx
    f_dx_new = (
        f_dx
        * jnp.maximum(
            0.0,
            jnp.minimum(1.0, (f - jnp.roll(f, 1, axis=0)) / denom),
        )
        * jnp.maximum(
            0.0,
            jnp.minimum(1.0, -(f - jnp.roll(f, -1, axis=0)) / denom),
        )
    )
    denom = (f_dy + 1.0e-8 * (f_dy == 0)) * dy
    f_dy_new = (
        f_dy
        * jnp.maximum(
            0.0,
            jnp.minimum(1.0, (f - jnp.roll(f, 1, axis=1)) / denom),
        )
        * jnp.maximum(
            0.0,
            jnp.minimum(1.0, -(f - jnp.roll(f, -1, axis=1)) / denom),
        )
    )

    return f_dx_new, f_dy_new


def extrapolate_to_face(f, f_dx, f_dy, dx, dy):
    """
    Extrapolate the field from cell centers to the faces using gradients

    Each pair holds the two states that meet at the face above the cell (i+1/2
    or j+1/2), in the usual Riemann-solver convention: '_L' is extrapolated
    forwards from the cell below the face and '_R' backwards from the cell
    above it.
    """

    f_XL = f + f_dx * dx / 2.0
    f_XR = jnp.roll(f - f_dx * dx / 2.0, -1, axis=0)  # right/up roll

    f_YL = f + f_dy * dy / 2.0
    f_YR = jnp.roll(f - f_dy * dy / 2.0, -1, axis=1)

    return f_XL, f_XR, f_YL, f_YR


def face_states(W, W_dx, W_dy, dx, dy):
    """Extrapolate a dict of fields to the faces, as the (XL, XR, YL, YR) states"""

    faces = {
        name: extrapolate_to_face(W[name], W_dx[name], W_dy[name], dx, dy) for name in W
    }

    return tuple({name: f[i] for name, f in faces.items()} for i in range(4))


def swap_xy(W):
    """Swap the x and y components of a dict of fields or fluxes"""

    return {_SWAP_XY.get(name, name): f for name, f in W.items()}


def apply_fluxes(F, flux_F_X, flux_F_Y, area_x, area_y, dt):
    """
    Apply fluxes to conserved variables

    The fluxes are weighted by the area of the face they act on.
    """
    AX = area_x * flux_F_X
    AY = area_y * flux_F_Y

    F_new = (
        F + dt * (-AX + jnp.roll(AX, 1, axis=0)) + dt * (-AY + jnp.roll(AY, 1, axis=1))
    )

    return F_new
