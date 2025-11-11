import jax.numpy as jnp

# Pure functions for gravity calculations


def calculate_gravitational_potential(rho, k_sq, G, rho_bar):
    V_hat = -jnp.fft.fftn(4.0 * jnp.pi * G * (rho - rho_bar)) / (k_sq + (k_sq == 0))
    V = jnp.real(jnp.fft.ifftn(V_hat))
    return V


def get_acceleration(V, kx, ky, dx, dy, bc_x_is_reflective, bc_y_is_reflective):
    if not bc_x_is_reflective or not bc_y_is_reflective:
        V_hat = jnp.fft.fftn(V)
    if bc_x_is_reflective:
        # 2nd order finite difference
        ax = -(jnp.roll(V, -1, axis=0) - jnp.roll(V, 1, axis=0)) / (2.0 * dx)
        # one-sided 2nd order difference at boundary
        ax = ax.at[0, :].set(-(-3.0 * V[0, :] + 4.0 * V[1, :] - V[2, :]) / (2.0 * dx))
        ax = ax.at[-1, :].set(
            -(3.0 * V[-1, :] - 4.0 * V[-2, :] + V[-3, :]) / (2.0 * dx)
        )
    else:
        # periodic
        ax = -jnp.real(jnp.fft.ifftn(1.0j * kx * V_hat))
    if bc_y_is_reflective:
        # 2nd order finite difference
        ay = -(jnp.roll(V, -1, axis=1) - jnp.roll(V, 1, axis=1)) / (2.0 * dy)
        # one-sided 2nd order difference at boundary
        ay = ay.at[:, 0].set(-(-3.0 * V[:, 0] + 4.0 * V[:, 1] - V[:, 2]) / (2.0 * dy))
        ay = ay.at[:, -1].set(
            -(3.0 * V[:, -1] - 4.0 * V[:, -2] + V[:, -3]) / (2.0 * dy)
        )
    else:
        # periodic
        ay = -jnp.real(jnp.fft.ifftn(1.0j * ky * V_hat))

    return ax, ay
