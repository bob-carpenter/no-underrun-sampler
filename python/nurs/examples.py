import numpy as np
from scipy.stats import norm
import nurs_reference as nr
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
    theoretical = 1 / np.sqrt(n_vals)  # theoretical error curve
    
    # Create DataFrame for plotting.
    df = pd.DataFrame({
        'n': n_vals,
        'rmss': rmss,
        'theoretical': theoretical
    })
    
    # Plot: red dotted line for theoretical, blue solid for running rmss error.
    plot = (
        pn.ggplot(df, pn.aes(x='n'))
        + pn.geom_line(pn.aes(y='theoretical'), linetype='dotted', color='red', size=1)
        + pn.geom_line(pn.aes(y='rmss'), size=0.5)
        + pn.labs(x='iteration', y='RMSSE')
        + pn.scale_x_log10()
        + pn.scale_y_log10()
    )
    return plot

    
def normal_logpdf(x):
    return -0.5 * np.sum(x**2)

def scatterplot(xs, ys, xlab='x', ylab='y', title=None):
    df = pd.DataFrame({'x': xs, 'y': ys})
    plot = (pn.ggplot(df, pn.aes(x='x', y='y'))
                + pn.geom_point()
                + pn.labs(x=xlab, y=ylab)
                + pn.ggtitle(title))
    return plot


# =================================

def example_normal(num_draws=100_000, step_size=0.75, threshold = 1e-5, max_doublings=10, seed=1234):
    rng = np.random.default_rng(seed)
    dim = 2
    theta_init = rng.normal(size=2)
    draws, accepts, depths = nr.nurs(rng, normal_logpdf, theta_init, num_draws, step_size, max_doublings, threshold)
    mean = np.mean(draws, axis=0)
    sd = np.std(draws, axis=0)
    print(f"     {mean=}\n       {sd=}")

    sp = scatterplot(draws[:, 0], draws[:, 1], 'x1', 'x2', title="Std Normal Scatterplot")
    sp.show()

    true_mean = np.zeros(2)
    true_sd = np.ones(2)
    rp = plot_running_rmsse(draws, true_mean, true_sd)
    rp.show()
    
def example_running_rmsse():
    num_draws = 100_000
    dims = 5
    true_means = np.zeros(dims)
    true_sds = np.ones(dims)
    draws = np.random.normal(loc=true_means, scale=true_sds, size=(num_draws, dims))
    plot = plot_running_rmsse(draws, true_means, true_sds)
    plot.show()

# example_running_rmsse()
example_normal()    
