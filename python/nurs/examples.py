import numpy as np
from scipy.stats import norm
import nurs
import nurs_step_adapt as nurs_ssa
import nurs_step_direct_adapt as nurs_sda

import pandas as pd
import plotnine as pn
import numpy as np


def running_rmsse(draws, true_means, true_sds):
    """Compute running root mean square standardized error.

    Args:
        draws (np.ndarray): Array of shape (num_draws, dims) with draws.
        true_means (np.ndarray): Array of shape (dims,) with true means.
        true_sds (np.ndarray): Array of shape (dims,) with true standard deviations.

    Returns:
        np.ndarray: Array of shape (num_draws,) containing the running
                    root mean square standardized errors.
    """
    num_draws = draws.shape[0]
    running_sum = np.cumsum(draws, axis=0)
    counts = np.arange(1, num_draws + 1)[:, None]
    running_means = running_sum / counts
    sq_std_errors = ((running_means - true_means) / true_sds) ** 2
    rmss = np.sqrt(sq_std_errors.mean(axis=1))
    return rmss


def plot_running_rmsse(draws, true_means, true_sds):
    num_draws, dim = np.shape(draws)
    rmss = running_rmsse(draws, true_means, true_sds)
    n_vals = np.arange(1, num_draws + 1)
    theoretical = 1 / np.sqrt(n_vals)
    df = pd.DataFrame({"n": n_vals, "rmss": rmss, "theoretical": theoretical})
    plot = (
        pn.ggplot(df, pn.aes(x="n"))
        + pn.geom_line(pn.aes(y="theoretical"), linetype="dotted", color="red", size=1)
        + pn.geom_line(pn.aes(y="rmss"), size=0.5)
        + pn.labs(x="iteration", y="RMSSE")
        + pn.scale_x_log10()
        + pn.scale_y_log10()
        + pn.ggtitle("Learning curve")
    )
    return plot


def normal_logpdf(x):
    return -0.5 * np.sum(x**2)

def funnel_logpdf(x):
    return 0.0


def scatterplot(xs, ys, xlab="x", ylab="y", title=None):
    df = pd.DataFrame({"x": xs, "y": ys})
    plot = (
        pn.ggplot(df, pn.aes(x="x", y="y"))
        + pn.geom_point()
        + pn.labs(x=xlab, y=ylab)
        + pn.ggtitle(title)
    )
    return plot


# =================================


def example_normal_sda(
    num_draws=10_000, min_step_size=0.25, threshold=0.0, max_tree_doublings=1,
    seed=1234, max_step_doublings=1, ensemble_size=10
):
    print("FITTING NORMAL WITH DIRECTION AND STEP-SIZE ADAPTIVE NURS")
    rng = np.random.default_rng(seed)
    dim = 2
    theta_init = rng.normal(size=(ensemble_size, dim))
    draws, accepts, depths = nurs_sda.nurs_sda(
        rng, normal_logpdf, theta_init, num_draws, 
        min_step_size, max_tree_doublings, max_step_doublings, threshold
    )
    mean = np.mean(draws, axis=0)
    sd = np.std(draws, axis=0)
    print(f"     {mean=}\n       {sd=}")

    sp = scatterplot(
        draws[:, 0], draws[:, 1], "x1", "x2", title="Std Normal Scatterplot"
    )
    sp.show()

    true_mean = np.zeros(2)
    true_sd = np.ones(2)
    rp = plot_running_rmsse(draws, true_mean, true_sd)
    rp.show()


def example_normal_ssa(
    num_draws=10_000, step_size=0.75, threshold=1e-5, max_tree_doublings=10,
    seed=1234, max_step_doublings=8,
):
    print("FITTING NORMAL WITH STEP-SIZE ADAPTIVE NURS")
    rng = np.random.default_rng(seed)
    dim = 2
    theta_init = rng.normal(size=2)
    draws, accepts, depths = nurs_ssa.nurs_ssa(
        rng, normal_logpdf, theta_init, num_draws, step_size,
        max_tree_doublings, max_step_doublings, threshold
    )
    mean = np.mean(draws, axis=0)
    sd = np.std(draws, axis=0)
    print(f"     {mean=}\n       {sd=}")

    sp = scatterplot(
        draws[:, 0], draws[:, 1], "x1", "x2", title="Std Normal Scatterplot"
    )
    sp.show()

    true_mean = np.zeros(2)
    true_sd = np.ones(2)
    rp = plot_running_rmsse(draws, true_mean, true_sd)
    rp.show()

def example_normal(
    num_draws=10_000, step_size=0.2, threshold=1e-5, max_tree_doublings=10,
    seed=1234, max_step_doublings=8,
):
    print("FITTING NORMAL WITH BASE NURS")
    rng = np.random.default_rng(seed)
    dim = 2
    theta_init = rng.normal(size=2)
    draws, accepts, depths = nurs.nurs(
        rng, normal_logpdf, theta_init, num_draws, step_size,
        max_tree_doublings, threshold
    )
    mean = np.mean(draws, axis=0)
    sd = np.std(draws, axis=0)
    print(f"     {mean=}\n       {sd=}")

    sp = scatterplot(
        draws[:, 0], draws[:, 1], "x1", "x2", title="Std Normal Scatterplot"
    )
    sp.show()

    true_mean = np.zeros(2)
    true_sd = np.ones(2)
    rp = plot_running_rmsse(draws, true_mean, true_sd)
    rp.show()
    

def example_independent_rmsse():
    num_draws = 100_000
    dims = 5
    true_means = np.zeros(dims)
    true_sds = np.ones(dims)
    draws = np.random.normal(loc=true_means, scale=true_sds, size=(num_draws, dims))
    plot = plot_running_rmsse(draws, true_means, true_sds)
    plot.show()


# COMMENT OUT TESTS TO SKIP

example_normal_sda()
# example_normal_ssa()
# example_normal()
# example_independent_rmsse()
