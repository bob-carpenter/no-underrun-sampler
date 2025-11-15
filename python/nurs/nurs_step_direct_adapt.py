import numpy as np
from scipy.special import logsumexp
import tqdm

# tree is tuple(selected[0], logp[1], left[2], right[3], logp_left[4], logp_right[5])


def nurs_sda(
    rng,
    logpdf,
    theta_inits,
    num_draws,
    min_step_size,
    max_tree_doublings,
    max_step_doublings,
    threshold,
):
    dim = np.size(theta_inits[0])
    log_threshold = np.log(threshold)
    ensemble_size = np.shape(theta_inits)[0]
    
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

        # Barker:
        # update = rng.binomial(1, np.exp(lp2 - lp12))
        # Metropolis:
        update = rng.binomial(1, np.min([1.0, np.exp(lp2 - lp1)]))

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

    def metropolis(theta, rho, current_step_size):
        lp_theta = logpdf(theta)  # computed twice (also by leaf)
        s = (rng.random() - 0.5) * current_step_size
        theta_star = theta + s * rho
        lp_theta_star = logpdf(theta_star)
        accept_prob = np.min([1.0, np.exp(lp_theta_star - lp_theta)])
        accept = rng.binomial(1, accept_prob)
        return (theta_star if accept else theta), accept

    def diff_direction(theta1, theta2):
        diff = theta1 - theta2
        return diff / np.linalg.norm(diff)
    
    def transition(ensemble):
        idx, idx_comp1, idx_comp2 = rng.choice(ensemble_size, size=3, replace=False) 
        theta, theta_comp1, theta_comp2 = ensemble[idx], ensemble[idx_comp1], ensemble[idx_comp2]
        rho = diff_direction(theta_comp1, theta_comp2) # changed to side move
        directions = rng.integers(0, 2, size=max_tree_doublings)
        current_step_size = min_step_size
        for _ in range(max_step_doublings):
            theta_met, accept = metropolis(theta, rho, current_step_size)
            tree = leaf(theta_met)
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
                    theta_star = tree[0]
                    ensemble[idx] = theta_star
                    return ensemble, accept, tree_depth
            current_step_size *= 2.0
        theta_star = tree[0]
        ensemble[idx] = theta_star
        return ensemble, accept, tree_depth

    def sample():
        ensemble = theta_inits
        draws = np.zeros((num_draws, ensemble_size, dim))
        accepts = np.zeros(num_draws, int)
        depths = np.zeros(num_draws, int)
        for m in tqdm.tqdm(range(num_draws)):
            ensemble, accepts[m], depths[m] = transition(ensemble)
            draws[m, :, :] = ensemble
        draws = draws.reshape((num_draws * ensemble_size, dim))
        return draws, accepts, depths

    return sample()
