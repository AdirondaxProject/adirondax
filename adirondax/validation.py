import itertools

BOUNDARY_CONDITIONS = ["periodic", "reflective", "axis", "outflow", "wall", "driven"]
MAGNETIC_BOUNDARY_CONDITIONS = ["periodic", "outflow", "axis", "wall", "driven"]


def validate_params(params):
    """
    Check that the simulation parameters are consistent and supported.

    Raises ValueError for invalid settings, and NotImplementedError for valid
    settings that are not yet supported.
    """

    mesh = params["mesh"]
    physics = params["physics"]
    hydro = params["hydro"]
    resolution = mesh["resolution"]
    bc_x, bc_y = mesh["boundary_condition"][:2]
    is_cylindrical = mesh["geometry"] == "cylindrical"

    if len(resolution) != len(mesh["box_size"]):
        raise ValueError("'resolution' and 'box_size' must have same shape")

    if len(resolution) == 3:
        raise NotImplementedError("3D is not yet implemented.")

    if any(n % 2 != 0 for n in resolution):
        raise ValueError(f"'resolution' must be even in every dimension: {resolution}")

    if mesh["type"] not in ["eulerian", "lagrangian", "ale"]:
        raise ValueError("mesh 'type' must be 'eulerian', 'lagrangian' or 'ale'")

    if mesh["geometry"] not in ["cartesian", "cylindrical"]:
        raise ValueError("mesh 'geometry' must be 'cartesian' or 'cylindrical'")

    for bc in (bc_x, bc_y):
        if bc not in BOUNDARY_CONDITIONS:
            raise ValueError(f"unknown boundary condition: '{bc}'")

    if physics["magnetic"]:
        for bc in (bc_x, bc_y):
            if bc not in MAGNETIC_BOUNDARY_CONDITIONS:
                raise NotImplementedError(
                    f"'{bc}' boundaries are not yet implemented for magnetic=True "
                    f"(use one of {MAGNETIC_BOUNDARY_CONDITIONS})"
                )
        for name in ["gravity", "external_potential"]:
            if physics[name]:
                raise NotImplementedError(
                    f"'{name}' is not yet implemented for magnetic=True"
                )

    if bc_y == "axis":
        raise ValueError("the 'axis' boundary condition only applies to dimension 0")

    if is_cylindrical:
        if mesh["origin"][0] < 0.0:
            raise ValueError("cylindrical geometry requires origin[0] >= 0")
        if mesh["origin"][0] == 0.0 and bc_x != "axis":
            raise ValueError(
                "a cylindrical domain reaching R=0 requires "
                "boundary_condition[0] == 'axis'"
            )
        if mesh["origin"][0] > 0.0 and bc_x == "axis":
            raise ValueError("the 'axis' boundary condition requires origin[0] == 0")
        for name in ["gravity", "quantum"]:
            if physics[name]:
                raise NotImplementedError(
                    f"'{name}' is not yet implemented for cylindrical geometry."
                )
    elif bc_x == "axis":
        raise ValueError("the 'axis' boundary condition requires cylindrical geometry")

    if physics["rotation"]:
        if not physics["hydro"]:
            raise ValueError("'rotation' requires hydro")
        if not (is_cylindrical or physics["magnetic"]):
            raise ValueError(
                "'rotation' requires cylindrical geometry or magnetic=True"
            )

    for name, size in params["units"].items():
        if not size > 0.0:
            raise ValueError(f"the code unit of '{name}' must be positive: {size}")

    for name, value in params["constants"].items():
        if value is not None and not value > 0.0:
            raise ValueError(f"the constant '{name}' must be positive or null: {value}")

    if not params["quantum"]["m_eV"] > 0.0:
        raise ValueError("the axion mass 'm_eV' must be positive")

    if not hydro["eos"]["mean_molecular_weight"] > 0.0:
        raise ValueError("'mean_molecular_weight' must be positive")

    if physics["radiation"]:
        validate_radiation_params(params["radiation"])
        if not physics["hydro"]:
            raise ValueError("'radiation' requires hydro")

    if hydro["riemann_solver"] not in ["llf", "hlld", "hllc"]:
        raise ValueError("riemann solver does not exist")

    if hydro["riemann_solver"] == "hlld" and not physics["magnetic"]:
        raise ValueError("'hlld' riemann solver only exists for magnetic=True")

    if hydro["riemann_solver"] == "hllc" and physics["magnetic"]:
        raise ValueError("'hllc' riemann solver only exists for magnetic=False")

    if bc_x != "periodic" or bc_y != "periodic":
        if physics["quantum"]:
            raise NotImplementedError(
                "Quantum only implemented for periodic boundary conditions."
            )
        if physics["gravity"]:
            raise NotImplementedError(
                "Gravity only implemented for periodic boundary conditions."
            )

    if (
        params["output"]["save"]
        and params["time"]["num_timesteps"] > 0
        and params["time"]["num_timesteps"] % params["output"]["num_checkpoints"] != 0
    ):
        raise ValueError("'num_checkpoints' must divide 'num_timesteps'")


def validate_radiation_params(rad):
    """Check that the radiation parameters are consistent"""

    for name in ("num_polar", "num_azimuthal"):
        n = rad[name]
        if n < 2 or n % 2 != 0:
            raise ValueError(f"radiation '{name}' must be even and >= 2: {n}")

    edges = rad["group_edges"]
    if any(e <= 0.0 for e in edges) or any(
        e1 >= e2 for e1, e2 in itertools.pairwise(edges)
    ):
        raise ValueError(
            f"radiation 'group_edges' must be positive and increasing: {edges}"
        )

    num_groups = len(edges) + 1
    for name in ("absorption", "planck"):
        kappa = rad["opacity"][name]
        if isinstance(kappa, (list, tuple)) and len(kappa) != num_groups:
            raise ValueError(
                f"radiation opacity '{name}' needs one value per frequency group "
                f"({num_groups}): {kappa}"
            )

    if rad["solver"] not in ["gmres", "jacobi"]:
        raise ValueError(f"unknown radiation solver: '{rad['solver']}'")
