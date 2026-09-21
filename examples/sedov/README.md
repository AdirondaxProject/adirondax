# Sedov-Taylor blast wave (cylindrical)

A point explosion in a uniform medium, run on an axisymmetric cylindrical
`(R,z)` mesh with the `axis` boundary condition at `R=0`.

The initial condition is spherically symmetric, so the solution must stay
spherically symmetric even though the mesh is not. The blast expanding as a
circle in the `(R,z)` plane is a direct check of two things: that the geometric
pressure source term in the radial momentum equation is discretized
consistently with the radius-weighted face areas, and that no artifact is
generated at the axis.

Run it with:

```console
python sedov.py
```
