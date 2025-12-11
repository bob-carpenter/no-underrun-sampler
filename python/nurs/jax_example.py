import jax
import jax.numpy as jnp
import jax.scipy as sp
import time

from nurs.nurs_jax import NURS_jax

@jax.jit
def lp_fun(x):
    y = x[0]
    z = x[1:]
    return jax.numpy.sum(sp.stats.norm.logpdf(z, 0, jnp.exp(y / 2))) + sp.stats.norm.logpdf(y, 0, 3)

d = 10
grid_size = 2**14
step_size = 0.01
N = 1_000_000
theta = jnp.zeros(d+1) 

start = time.time()
samples, accepts = NURS_jax(lp_fun, theta, N, jax.random.key(0), grid_size, step_size)
print(jnp.mean(accepts))
end = time.time()

print(end - start)
