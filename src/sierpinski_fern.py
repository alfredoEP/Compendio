# !/usr/bin/env python

__author__ = "Alfredo Espinoza"

"""
sierpinski_fern.py: This script contains a framework to simulate the generation of a fractal fern leaf (Barnsley Fern).
Here we run the iterated function system multiple times with random initial configurations to simulate organic growth.
"""

import random
import matplotlib.pyplot as plt

def get_next_point(
    x: float,
    y: float,
    mutation_factor: float=0.0
) -> tuple:
    """
    Calculate the next coordinate point based on the Barnsley Fern affine transformations.
    
    Parameters:
    x: The current x coordinate.
    y: The current y coordinate.
    mutation_factor: A randomness factor to simulate organic imperfections in the leaf.

    Returns:
    tuple, the new (x, y) coordinates
    """
    
    # We use a random number to determine which transformation to apply.
    # Probability weights: 1% Stem, 85% Successive Leaflets, 7% Largest Left, 7% Largest Right
    r = random.random()
    
    # Random noise for mutation (affects the coefficients slightly)
    noise = 0.0
    if mutation_factor > 0:
        noise = random.uniform(-mutation_factor, mutation_factor)

    if r < 0.01:
        # 1. Stem generation
        next_x = 0
        next_y = 0.16 * y
    elif r < 0.86:
        # 2. Successively smaller leaflets
        next_x = (0.85 + noise) * x + (0.04 + noise) * y
        next_y = (-0.04 + noise) * x + (0.85 + noise) * y + 1.6
    elif r < 0.93:
        # 3. Largest left-hand leaflet
        next_x = 0.2 * x - 0.26 * y
        next_y = 0.23 * x + (0.22 + noise) * y + 1.6
    else:
        # 4. Largest right-hand leaflet
        next_x = -0.15 * x + 0.28 * y
        next_y = 0.26 * x + 0.24 * y + 0.44

    return next_x, next_y



def simulate_fern_growth(
    iterations: int=10000,
    start_x: float=0.0,
    start_y: float=0.0,
    mutation_factor: float=0.0
) -> list:
    """
    Run the simulation loop to generate the set of points forming the fern.

    Parameters:
    iterations: The total number of points to generate.
    start_x: The starting x coordinate (random configuration).
    start_y: The starting y coordinate (random configuration).
    mutation_factor: The degree of random variation in the growth logic.

    Returns:
    list, a list of tuples containing (x, y) coordinates
    """
    points = []
    current_x, current_y = start_x, start_y
    
    # Discard the first few points to settle into the attractor
    settling_steps = 20

    for i in range(iterations + settling_steps):
        current_x, current_y = get_next_point(current_x, current_y, mutation_factor)
        if i >= settling_steps:
            points.append((current_x, current_y))
            
    return points

def plot_fern_simulation(
    points: list,
    iterations: int,
    mutation_factor: float
) -> None:
    """
    Plot the resulting fern fractal.

    Parameters:
    points: List of (x,y) tuples.
    iterations: Number of points generated.
    mutation_factor: The mutation factor used.
    """
    x_vals = [p[0] for p in points]
    y_vals = [p[1] for p in points]

    plt.figure(figsize=(8, 12))
    # Green with slight transparency to show density
    plt.scatter(x_vals, y_vals, s=0.2, c='forestgreen', marker='.')
    
    plt.title(f'Barnsley Fern Simulation\nPoints: {iterations}, Mutation: {mutation_factor:.4f}', fontsize=14, fontweight='bold')
    plt.axis('off') # Hide axes for better aesthetic
    
    filename = './assets/fern_simulation_plot.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"✓ Plot saved as '{filename}'")

if __name__ == '__main__':
    print("=" * 70)
    print("SIERPINSKI FERN SIMULATION")
    print("=" * 70)

    # ============ CONFIGURATION ============
    ITERATIONS = 50000
    # Randomize initial configurations as requested
    START_X = random.uniform(-2.0, 2.0)
    START_Y = random.uniform(0.0, 5.0)
    # A small mutation simulates biological variety (0.0 = perfect math fern)
    MUTATION_FACTOR = random.uniform(0.0, 0.02) 
    # =======================================

    print(f"\n[Configuration] Random Initialization")
    print("-" * 70)
    print(f"Starting Coordinates: ({START_X:.4f}, {START_Y:.4f})")
    print(f"Genetic Mutation Factor: {MUTATION_FACTOR:.4f}")
    print(f"Iterations: {ITERATIONS}")

    print("\nRunning simulation...")
    fern_points = simulate_fern_growth(
        iterations=ITERATIONS,
        start_x=START_X,
        start_y=START_Y,
        mutation_factor=MUTATION_FACTOR
    )

    print(f"Generated {len(fern_points)} points.")
    
    plot_fern_simulation(fern_points, ITERATIONS, MUTATION_FACTOR)

    print("\n" + "=" * 70)
    print("Simulation complete!")
    print("=" * 70)