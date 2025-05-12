# pso_tuner.py
"""
Particle Swarm Optimization for hyperparameter tuning using pyswarms.
"""
import numpy as np
import pyswarms as ps

def tune_hyperparameters(objective_fn, bounds, n_particles: int = 10, iters: int = 20):
    """
    objective_fn: function that accepts an array of shape (n_particles, dims) and returns loss array.
    bounds: tuple of (lower_bounds, upper_bounds) each as numpy arrays.
    """
    options = {'c1': 0.5, 'c2': 0.3, 'w': 0.9}
    optimizer = ps.single.GlobalBestPSO(
        n_particles=n_particles,
        dimensions=len(bounds[0]),
        options=options,
        bounds=bounds
    )
    best_cost, best_pos = optimizer.optimize(objective_fn, iters=iters)
    return best_pos, best_cost