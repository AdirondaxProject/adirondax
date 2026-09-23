import jax
import jax.numpy as jnp

# Pure functions for coupling the simulation domain to external circuits
#
# The domain and the circuits meet at ports. Each port carries a current into
# the domain and reports a voltage back out. Solved via Sherman-Morrison-Woodbury.


def coupled_step(load_step, circuit_step, currents):
    """
    Advance a domain and its external circuits by one step.

    Parameters
    ----------
    load_step: callable
      currents -> (state, voltages). Advances the domain one timestep with the
      given port currents imposed, and returns the new state along with the
      voltage measured at each port. Must be differentiable.
    circuit_step: callable
      voltages -> (circuit_state, currents). Advances the circuits one timestep
      with the given port voltages imposed, and returns the new circuit state
      along with the current delivered to each port. Must be differentiable.
    currents: jax.Array
      Port currents from the previous step, shape (num_ports,).

    Returns
    -------
    state: the domain state.
    circuit_state: the circuit state.
    currents: the port currents, shape (num_ports,).
    voltages: the port voltages, shape (num_ports,).
    """

    num_ports = currents.shape[0]

    def load_voltages(i):
        return load_step(i)[1]

    def circuit_currents(v):
        return circuit_step(v)[1]

    # the split solves: the domain driven by the incoming currents, then the
    # circuits driven by the port voltages that result
    voltages_split = load_voltages(currents)
    currents_split = circuit_currents(voltages_split)

    # how each side responds to the other: Z is the domain's port impedance
    # over this step, Y the circuits' port admittance
    Z = jax.jacfwd(load_voltages)(currents)
    Y = jax.jacfwd(circuit_currents)(voltages_split)

    # solve the two responses against each other
    delta = jnp.linalg.solve(jnp.eye(num_ports) - Y @ Z, currents_split - currents)
    currents = currents + delta

    # take the step both sides now agree on
    state, voltages = load_step(currents)
    circuit_state, currents = circuit_step(voltages)

    return state, circuit_state, currents, voltages
