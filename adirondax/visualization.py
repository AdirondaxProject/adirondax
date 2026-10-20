import os

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

from .radiation.angles import get_angles


def plot_sim(state, checkpoint_dir, i, params):
    """Plot the simulation state."""

    dynamic_range = params["output"]["plot_dynamic_range"]

    # process distributed data
    if params["physics"]["quantum"]:
        rho_bar_dm = jnp.mean(jnp.abs(state["psi"]) ** 2)
        rho_proj_dm = jnp.log10(jnp.abs(state["psi"]) ** 2).T
    if params["physics"]["hydro"]:
        rho_gas = state["rho"]
        if params["mesh"]["geometry"] == "cylindrical":
            # reflect about the symmetry axis at R=0
            rho_gas = jnp.concatenate((jnp.flip(rho_gas, axis=0), rho_gas), axis=0)
        rho_bar_gas = jnp.mean(state["rho"])
        rho_proj_gas = jnp.log10(rho_gas).T
    if params["physics"]["radiation"]:
        rad = params["radiation"]
        if rad["method"] == "diffusion":
            E_rad = jnp.sum(state["Er"], axis=0)
        else:
            w = get_angles(rad["num_polar"], rad["num_azimuthal"])["w"]
            E_rad = 4.0 * jnp.pi * jnp.einsum("n,fnij->ij", w, state["I"])
        if params["mesh"]["geometry"] == "cylindrical":
            E_rad = jnp.concatenate((jnp.flip(E_rad, axis=0), E_rad), axis=0)
        # floored, as the radiation can be zero where no light has yet reached
        E_rad = jnp.maximum(E_rad, jnp.finfo(E_rad.dtype).tiny)
        E_rad_max = jnp.max(E_rad)
        E_rad_proj = jnp.log10(E_rad).T

    # create plot on process 0
    if jax.process_index() == 0:
        if params["physics"]["quantum"]:
            plt.clf()

            # DM projection
            nx = state["psi"].shape[0]
            ny = state["psi"].shape[1]
            vmin = jnp.log10(rho_bar_dm / dynamic_range)
            vmax = jnp.log10(rho_bar_dm * dynamic_range)

            ax = plt.gca()
            ax.imshow(
                rho_proj_dm,
                cmap="inferno",
                origin="lower",
                vmin=vmin,
                vmax=vmax,
                extent=[0, nx, 0, ny],
            )
            ax.set_aspect("equal")
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

            plt.savefig(
                os.path.join(checkpoint_dir, f"dm{i:03d}.png"),
                bbox_inches="tight",
                pad_inches=0,
            )
            plt.close()

        if params["physics"]["hydro"]:
            plt.clf()

            # gas projection
            ny, nx = rho_proj_gas.shape
            vmin = jnp.log10(rho_bar_gas / dynamic_range)
            vmax = jnp.log10(rho_bar_gas * dynamic_range)

            ax = plt.gca()
            ax.imshow(
                rho_proj_gas,
                cmap="viridis",
                origin="lower",
                vmin=vmin,
                vmax=vmax,
                extent=[0, nx, 0, ny],
            )
            ax.set_aspect("equal")
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

            plt.savefig(
                os.path.join(checkpoint_dir, f"gas{i:03d}.png"),
                bbox_inches="tight",
                pad_inches=0,
            )
            plt.close()

        if params["physics"]["radiation"]:
            plt.clf()

            # radiation energy density, down from its maximum
            ny, nx = E_rad_proj.shape
            vmax = jnp.log10(E_rad_max)
            vmin = vmax - jnp.log10(dynamic_range)

            ax = plt.gca()
            ax.imshow(
                E_rad_proj,
                cmap="inferno",
                origin="lower",
                vmin=vmin,
                vmax=vmax,
                extent=[0, nx, 0, ny],
            )
            ax.set_aspect("equal")
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

            plt.savefig(
                os.path.join(checkpoint_dir, f"rad{i:03d}.png"),
                bbox_inches="tight",
                pad_inches=0,
            )
            plt.close()
