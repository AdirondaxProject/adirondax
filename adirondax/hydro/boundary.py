import jax.numpy as jnp
from jax import lax

# Pure functions for ghost cells at non-periodic boundaries

_NORMAL = ({"vx", "Bx", "bx"}, {"vy", "By", "by"})
_OUT_OF_PLANE = {"vz", "Bz", "bz"}
_FACE_AXIS = {"bx": 0, "by": 1}
_MIRRORED = ("reflective", "wall", "axis")


def _parity(name, axis, bc):
    """(lo, hi) signs with which a cell-centered field is mirrored into the ghosts"""

    if name in _NORMAL[axis]:
        return -1.0, -1.0
    if name in _OUT_OF_PLANE and bc == "axis":
        return -1.0, 1.0
    return 1.0, 1.0


def add_ghost_cells(W, axis, bc, ghost=None):
    """
    Pad each field with one ghost cell on each side of the given axis

    'outflow' copies the edge value outwards. 'reflective', 'wall' and 'axis'
    mirror the fields, oddly for the components normal to the boundary, and
    zero the face-centered normal magnetic field. 'axis' also mirrors the
    out-of-plane components oddly through R=0. 'driven' takes the (lo, hi)
    ghost cells of each field from ghost.
    """

    if bc not in ("outflow", "driven") + _MIRRORED:
        raise ValueError(f"unknown boundary condition: '{bc}'")

    padded = {}
    for name, f in W.items():
        if bc == "driven":
            lo, hi = ghost[name]
        else:
            lo = lax.index_in_dim(f, 0, axis)
            hi = lax.index_in_dim(f, -1, axis)
        if bc in _MIRRORED:
            if _FACE_AXIS.get(name) == axis:
                lo, hi = jnp.zeros_like(lo), jnp.zeros_like(hi)
            else:
                s_lo, s_hi = _parity(name, axis, bc)
                lo, hi = s_lo * lo, s_hi * hi
        padded[name] = jnp.concatenate((lo, f, hi), axis=axis)

    return padded


def set_ghost_gradients(W_d, axis, bc):
    """
    Set the gradients normal to the boundary in the ghost cells: flat for
    'outflow' and 'driven', otherwise mirrored with the opposite parity to the
    field.
    """

    fixed = {}
    for name, f_d in W_d.items():
        if bc in _MIRRORED:
            s_lo, s_hi = _parity(name, axis, bc)
            lo = -s_lo * lax.index_in_dim(f_d, 1, axis, keepdims=False)
            hi = -s_hi * lax.index_in_dim(f_d, -2, axis, keepdims=False)
        else:
            lo = hi = 0.0
        if axis == 0:
            fixed[name] = f_d.at[0, :].set(lo).at[-1, :].set(hi)
        else:
            fixed[name] = f_d.at[:, 0].set(lo).at[:, -1].set(hi)

    return fixed


def strip_ghosts(f, axis):
    """Drop the ghost cell on each side of the given axis"""

    return f[1:-1, :] if axis == 0 else f[:, 1:-1]
