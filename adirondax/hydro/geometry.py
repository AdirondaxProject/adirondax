import jax.numpy as jnp

# Pure functions for 2D mesh geometry (finite-volume metric factors)


def get_geometry(geometry, box_size, resolution, num_ghost_x=0, r_min=0.0):
    """
    Build the finite-volume metric factors for a 2D mesh.

    Parameters
    ----------
    geometry: str
      'cartesian' for (x,y), or 'cylindrical' for axisymmetric (R,z).
    box_size: list
      Domain extent per dimension. For 'cylindrical' this is (L_R, L_z), and
      the domain is R in [0, L_R], z in [0, L_z].
    resolution: list
      Number of cells per dimension.
    num_ghost_x: int
      Number of ghost cells added on each side of the x/R axis.
    r_min: float
      Inner radius of the domain.

    Returns
    -------
    geom: dict
      'vol'       cell volume
      'area_x'    area of the face at i+1/2
      'area_y'    area of the face at j+1/2
      'd_area_x'  area_x - (area of the face at i-1/2); drives the geometric
                  pressure source term, and is zero in cartesian geometry
      'r'         volume-centroid radius (None in cartesian geometry)
      'r_face_x'  radius of the face at i+1/2 (None in cartesian geometry)
      'dx', 'dy'  cell widths
      'is_cylindrical'  bool
    """

    Lx, Ly = box_size[0], box_size[1]
    nx, ny = resolution[0], resolution[1]
    dx = Lx / nx
    dy = Ly / ny

    if geometry == "cartesian":
        return {
            "vol": dx * dy,
            "area_x": dy,
            "area_y": dx,
            "d_area_x": 0.0,
            "r": None,
            "r_face_x": None,
            "dx": dx,
            "dy": dy,
            "is_cylindrical": False,
        }

    if geometry != "cylindrical":
        raise ValueError(f"unknown mesh geometry: {geometry}")

    # Cell edges, extended by the ghost cells. Taking the absolute value
    # mirrors the metric across the axis, so a ghost cell spanning [-dx, 0]
    # gets the same volume as the first interior cell.
    g = num_ghost_x
    edges = r_min + dx * (jnp.arange(nx + 1 + 2 * g) - g)
    if r_min == 0.0:
        edges = jnp.abs(edges)
    Rm = edges[:-1]  # R_{i-1/2}
    Rp = edges[1:]  # R_{i+1/2}

    dR2 = jnp.abs(Rp**2 - Rm**2)
    dR3 = jnp.abs(Rp**3 - Rm**3)

    # vol = 2*pi * int R dR dz, area_x = 2*pi*R*dz, area_y = 2*pi * int R dR
    vol = (jnp.pi * dR2 * dy)[:, None]
    area_x = (2.0 * jnp.pi * Rp * dy)[:, None]
    area_x_lo = (2.0 * jnp.pi * Rm * dy)[:, None]
    area_y = (jnp.pi * dR2)[:, None]

    # volume-centroid radius; equals (2/3)*dx in the cell touching the axis,
    # so terms carrying a 1/R never divide by zero
    r = ((2.0 / 3.0) * dR3 / dR2)[:, None]

    return {
        "vol": vol,
        "area_x": area_x,
        "area_y": area_y,
        "d_area_x": area_x - area_x_lo,
        "r": r,
        "r_face_x": Rp[:, None],
        "dx": dx,
        "dy": dy,
        "is_cylindrical": True,
    }
