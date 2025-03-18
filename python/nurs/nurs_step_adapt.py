import numpy as np
from scipy.special import logsumexp
import tqdm

# tree is tuple(selected[0], logp[1], left[2], right[3], logp_left[4], logp_right[5])


def nurs_ssa(
    rng,
    logpdf,
    theta_init,
    num_draws,
    min_step_size,
    max_tree_doublings,
    max_step_doublings,
    threshold,
):
    dim = np.size(theta_init)
    log_threshold = np.log(threshold)

    def stopping_condition(tree, current_step_size):
        log_epsilon = log_threshold + np.log(current_step_size) + tree[1]
        return tree[4] < log_epsilon and tree[5] < log_epsilon

    def leaf(theta):
        lp = logpdf(theta)
        return (theta, lp, theta, theta, lp, lp)

    def combine_trees(tree1, tree2, direction):
        lp1 = tree1[1]
        lp2 = tree2[1]
        lp12 = logsumexp([lp1, lp2])
        update = rng.binomial(1, np.exp(lp2 - lp12))
        selected = tree2[0] if update else tree1[0]
        if direction == 1:
            return (selected, lp12, tree1[2], tree2[3], tree1[4], tree2[5])
        else:
            return (selected, lp12, tree2[2], tree1[3], tree2[4], tree1[5])

    def build_tree(depth, theta_last, rho, direction, current_step_size):
        h = current_step_size * (2 * direction - 1)
        if depth == 0:
            theta_next = theta_last + h * rho
            return leaf(theta_next)
        tree1 = build_tree(depth - 1, theta_last, rho, direction, current_step_size)
        if not tree1:
            return None
        theta_mid = tree1[3] if direction == 1 else tree1[2]
        tree2 = build_tree(depth - 1, theta_mid, rho, direction, current_step_size)
        if not tree2:
            return None
        tree = combine_trees(tree1, tree2, direction)
        if stopping_condition(tree, current_step_size):
            return None
        return tree

    def random_direction():
        u = rng.normal(size=dim)
        return u / np.linalg.norm(u)

    def metropolis(theta, rho, current_step_size):
        lp_theta = logpdf(theta)  # computed twice (also by leaf)
        s = (rng.random() - 0.5) * current_step_size
        theta_star = theta + s * rho
        lp_theta_star = logpdf(theta_star)
        accept_prob = np.min([1.0, np.exp(lp_theta_star - lp_theta)])
        accept = rng.binomial(1, accept_prob)
        return (theta_star if accept else theta), accept

    def transition(theta):
        rho = random_direction()
        directions = rng.integers(0, 2, size=max_tree_doublings)
        current_step_size = min_step_size
        for _ in range(max_step_doublings):
            theta, accept = metropolis(theta, rho, current_step_size)
            tree = leaf(theta)
            for tree_depth in range(max_tree_doublings):
                direction = directions[tree_depth]
                theta_mid = tree[3] if direction == 1 else tree[2]
                tree_next = build_tree(
                    tree_depth, theta_mid, rho, direction, current_step_size
                )
                if not tree_next:
                    break
                tree = combine_trees(tree, tree_next, direction)
                if stopping_condition(tree, current_step_size):
                    return tree[0], accept, tree_depth
            current_step_size *= 2.0
        return tree[0], accept, tree_depth

    def sample():
        draws = np.zeros((num_draws, dim))
        accepts = np.zeros(num_draws, int)
        depths = np.zeros(num_draws, int)
        draws[0] = theta_init
        for m in tqdm.tqdm(range(1, num_draws), initial=1, total=num_draws):
            draws[m, :], accepts[m], depths[m] = transition(draws[m - 1])
        return draws, accepts, depths

    return sample()
