import jax
import jax.numpy as jnp

# Pure functions for gravity calculations


@jax.jit
def calculate_gravitational_potential(rho, G, kSq):
    rho_bar = jnp.mean(rho)
    Vhat = -jnp.fft.fftn(4.0 * jnp.pi * G * (rho - rho_bar)) / (kSq + (kSq == 0))
    return jnp.real(jnp.fft.ifftn(Vhat))
