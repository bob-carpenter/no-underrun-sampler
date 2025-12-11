import jax
import jax.numpy as jnp
import jax.scipy as sp
from functools import partial

@jax.jit
def categorical_logit_random(logits, key):
    logits = jnp.where(logits > 1000, -jnp.inf, logits)
    logits = jnp.where(logits < -1000, -jnp.inf, logits)
    logits = jnp.nan_to_num(logits, nan=-jnp.inf)

    logits = logits - jnp.max(logits)

    probabilities = sp.special.softmax(logits)
    selected_index = jax.random.choice(key, len(logits), p=probabilities)
    return selected_index

@partial(jax.jit, static_argnames=['lp_fun', 'd', 'grid_size'])
def draw(lp_fun, theta, d, key, grid_size=1024, step_size=0.1):
    keys = jax.random.split(key, 5)

    # sample rho
    rho = jax.random.normal(keys[0], shape=(d,))
    rho /= jnp.linalg.norm(rho)

    # metropolis step
    s = (jax.random.uniform(keys[1]) - .5) * step_size
    log_accept_prob = lp_fun(theta + s * rho) - lp_fun(theta)
    accept = jax.random.bernoulli(keys[2], jnp.exp(log_accept_prob))
    theta = theta + s * rho * accept

    # build grid
    b = jax.random.choice(keys[3], grid_size)
    a = grid_size - 1 - b
    grid = jnp.linspace(-a, b, grid_size) * step_size 
    positions = theta + rho * grid[:, None]
    log_probs = jax.vmap(lp_fun, in_axes=0)(positions)
    idx = categorical_logit_random(log_probs, keys[4])
    return positions[idx], accept

@partial(jax.jit, static_argnames=['lp_fun', 'num_draws', 'grid_size'])
def NURS_jax(lp_fun, theta_initial, num_draws, key, grid_size, step_size):
    theta = theta_initial.copy()
    d = len(theta_initial)
    keys = jax.random.split(key, num_draws)

    def body_fun(theta, i):
        theta, accept = draw(lp_fun, theta, d, keys[i], grid_size=grid_size, step_size=step_size)
        return theta, (theta, accept)
    
    _, samples = jax.lax.scan(body_fun, theta, jnp.arange(num_draws))
    
    return jnp.array(samples[0]), jnp.array(samples[1])
