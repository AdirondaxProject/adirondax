import jax.numpy as jnp

# Pure functions for gravity calculations


def calculate_gravitational_potential(rho, k_sq, G, rho_bar):
    V_hat = -jnp.fft.fftn(4.0 * jnp.pi * G * (rho - rho_bar)) / (k_sq + (k_sq == 0))
    V = jnp.real(jnp.fft.ifftn(V_hat))
    return V


def get_acceleration(V, kx, ky):
    V_hat = jnp.fft.fftn(V)
    ax = -jnp.real(jnp.fft.ifftn(1.0j * kx * V_hat))
    ay = -jnp.real(jnp.fft.ifftn(1.0j * ky * V_hat))
    return ax, ay
