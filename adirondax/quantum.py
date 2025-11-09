import jax.numpy as jnp

# Pure functions for quantum simulation


def quantum_kick(psi, V, m_per_hbar, dt):
    psi_new = jnp.exp(-1.0j * m_per_hbar * dt * V) * psi
    return psi_new


def quantum_drift(psi, k_sq, m_per_hbar, dt):
    psi_hat = jnp.fft.fftn(psi)
    psi_hat = jnp.exp(dt * (-1.0j * k_sq / m_per_hbar / 2.0)) * psi_hat
    psi_new = jnp.fft.ifftn(psi_hat)
    return psi_new


def quantum_timestep(m_per_hbar, dx, dy):
    dl = jnp.minimum(dx, dy)
    dt = (m_per_hbar / 6.0) * (dl * dl)
    return dt
