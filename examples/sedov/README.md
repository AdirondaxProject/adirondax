# Sedov-Taylor blast wave

Simulate the Sedov-Taylor blast wave in cylindrical coordinates.

Philip Mocz (2026)

A point explosion in a uniform medium, run on a 2D cylindrical mesh.

The initial condition is spherically symmetric, so the solution must stay spherically symmetric.

The script plots the final density of every cell of the `(R,z)` mesh against its
spherical radius. A spherically symmetric solution collapses onto a single
curve, which is compared against the analytic Sedov-Taylor similarity solution.

Run it with:

```console
python sedov.py
```
