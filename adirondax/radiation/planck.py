import jax.numpy as jnp

# Pure functions for blackbody emission integrated over frequency groups

# Bernoulli expansion of int_0^x t^3/(e^t-1) dt, as (power, coefficient)
_SERIES = (
    (3, 1.0 / 3.0),
    (4, -1.0 / 8.0),
    (5, 1.0 / 60.0),
    (7, -1.984126984126984e-04),
    (9, 3.6743092298647855e-06),
    (11, -7.515632515632516e-08),
    (13, 1.6059043836821615e-09),
    (15, -3.522793425791662e-11),
    (17, 7.872080312167458e-13),
    (19, -1.784042261222412e-14),
    (21, 4.088600979179926e-16),
    (23, -9.455950863295921e-18),
    (25, 2.203601131344092e-19),
    (27, -5.1683202540046385e-21),
)
_X_SWITCH = 2.0
_NUM_EXP_TERMS = 20
# beyond this the emission above x, ~ x^3 e^{-x}, underflows in double precision
_X_MAX = 700.0


def _planck_fractions(x):
    """
    The fractions of the blackbody energy density emitted below and above
    x = h nu / (k_B T), each to full relative precision, even where it is tiny
    """

    x = jnp.asarray(x)
    norm = 15.0 / jnp.pi**4

    # small x: the series converges for x < 2 pi. Clamping each branch to its
    # own range keeps the unused branch (and its gradient) finite.
    is_low = x < _X_SWITCH
    xs = jnp.where(is_low, x, _X_SWITCH)
    low = norm * sum(c * xs**p for p, c in _SERIES)

    # large x: int_x^inf t^3/(e^t-1) dt = sum_k e^{-kx} (x^3/k + 3x^2/k^2 + 6x/k^3 + 6/k^4)
    xl = jnp.where(is_low, _X_SWITCH, x)
    tail = 0.0
    for k in range(1, _NUM_EXP_TERMS + 1):
        tail = tail + jnp.exp(-k * xl) * (
            xl**3 / k + 3.0 * xl**2 / k**2 + 6.0 * xl / k**3 + 6.0 / k**4
        )
    high = norm * tail

    below = jnp.where(is_low, low, 1.0 - high)
    above = jnp.where(is_low, 1.0 - low, high)
    return below, above


def planck_fraction(x):
    """
    Fraction of the blackbody energy density emitted below x = h nu / (k_B T):
    (15/pi^4) int_0^x t^3/(e^t - 1) dt, which rises from 0 at x=0 to 1 as x->inf.
    """

    return _planck_fractions(x)[0]


def group_emission(T, group_edges, a_r):
    """
    Thermal emission epsilon_f = (1/4pi) int B_nu(T) d nu of each frequency
    group, which sums to a_r T^4 / (4 pi) over all groups.

    Parameters
    ----------
    T: jax.Array
      Gas temperature.
    group_edges: sequence
      The interior group boundaries h nu / k_B, in temperature units; the
      groups are [0, e_0), [e_0, e_1), ..., [e_last, inf). Empty for grey.
    a_r: float
      Radiation constant.

    Returns
    -------
    eps: jax.Array of shape (num_groups,) + T.shape
    """

    T = jnp.asarray(T)
    if len(group_edges) == 0:
        return (a_r * T**4 / (4.0 * jnp.pi))[None]

    edges = jnp.asarray(group_edges, dtype=T.dtype).reshape((-1,) + (1,) * T.ndim)
    # x = edges / T, capped at _X_MAX, which keeps it (and its derivative)
    # finite as T -> 0, where all the emission is in the lowest group
    x = edges / jnp.maximum(T, edges / _X_MAX)
    below, above = _planck_fractions(x)
    zero = jnp.zeros_like(below[:1])
    one = jnp.ones_like(below[:1])
    below = jnp.concatenate((zero, below, one), axis=0)
    above = jnp.concatenate((one, above, zero), axis=0)

    # a group is the difference of the fractions below its edges, or, where
    # most of the emission lies below it, of those above them, which would
    # otherwise cancel in the Wien tail
    frac = jnp.where(below[:-1] > 0.5, above[:-1] - above[1:], below[1:] - below[:-1])

    return (a_r * T**4 / (4.0 * jnp.pi)) * frac
