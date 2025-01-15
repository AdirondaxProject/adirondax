import jax
import jax.numpy as jnp

# Pure functions for quantum simulation


@jax.jit
def quantum_kick(psi, V, dt):
    return jnp.exp(-1.0j * dt * V) * psi


@jax.jit
def quantum_drift(psi, kSq, dt):
    psihat = jnp.fft.fftn(psi)
    psihat = jnp.exp(dt * (-1.0j * kSq / 2.0)) * psihat
    return jnp.fft.ifftn(psihat)
