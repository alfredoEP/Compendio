# !/usr/bin/env python

__author__ = "Alfredo Espinoza"

"""
lightbulb_survivability.py: This script contains a framework to simulate the survavivability of lightbulbs over time.
Here we run the experiment multiple times and average their surviving results.
And then we predict using the exponential decay model.
"""

import random
import numpy as np
import matplotlib.pyplot as plt

def single_lightbulb_survivability(
    lifespan_in_days: int=1000,
    failure_probability: float=0.001
)->bool:
    """
    Run of a single lightbulb survivability test with its outcome.

    Parameters:
    lifespan_in_days: The total time the lightbulb is tested.
    failure_probability: The probability of failure at each day.

    Returns:
    bool, either it is True (survived) or False (failed)
    """
    
    survived = True
    for day in range(lifespan_in_days):
        if random.random() < failure_probability:
            survived = False
            break
    return survived

def series_of_experiments(
    quantity_of_runs: int=1000,
    lifespan_in_days: int=1000,
    failure_probability: float=0.001
)-> float:
    """
    Multiple runs and get their average.

    Parameters:
    quantity_of_runs: How many times we want to run a lightbulb survivability test.
    lifespan_in_days: The total time the lightbulb is tested.
    failure_probability: The probability of failure at each day.

    Returns:
    float, the average of the survivability
    """
    experiment_outcomes = [ 1 if single_lightbulb_survivability(
        lifespan_in_days, failure_probability) else 0 for _ in range(quantity_of_runs)  ]
    avg_experiment_outcomes = sum(experiment_outcomes) / quantity_of_runs
    return avg_experiment_outcomes

def get_theoretical_survivability(
    lifespan_in_days: int,
    failure_probability: float
) -> float:
    """
    Get the theoretical survivability using the exponential decay model.

    Parameters:
    lifespan_in_days: The total time the lightbulb is tested.
    failure_probability: The probability of failure at each day.

    Returns:
    float, the theoretical survivability
    """
    theoretical_survivability = np.exp(-failure_probability * lifespan_in_days)
    return theoretical_survivability

def fit_failure_probability(
    lifespan_in_days: int,
    observed_survivability: float
) -> float:
    """
    Fit the failure probability based on observed survivability.

    Parameters:
    lifespan_in_days: The total time the lightbulb is tested.
    observed_survivability: The observed survivability rate.

    Returns:
    float, the fitted failure probability
    """
    if observed_survivability <= 0 or observed_survivability > 1:
        raise ValueError("Observed survivability must be in the range (0, 1].")
    
    fitted_failure_probability = -np.log(observed_survivability) / lifespan_in_days
    return fitted_failure_probability

def plot_experiment_outcomes(
    time_points: list,
    experimental_results: list,
    theoretical_results: list,
    failure_probability: float,
    lifespan_in_days: int,
    fitted_failure_probability: float = None
) -> None:
    """
    Plot the survivability results along with the exponential decay model.

    Parameters:
    time_points: List of time points in days.
    experimental_results: List of experimental survival rates.
    theoretical_results: List of theoretical survival rates.
    failure_probability: The failure probability per day (original or assumed).
    lifespan_in_days: The total lifespan in days.
    fitted_failure_probability: The fitted failure probability (optional).
    """
    plt.figure(figsize=(12, 7))
    
    plt.plot(time_points, experimental_results, label='Experimental Results', marker='o', linewidth=2, markersize=6)
    plt.plot(time_points, theoretical_results, label=f'Theoretical Model (λ={failure_probability:.4f})', linestyle='--', linewidth=2)
    
    if fitted_failure_probability is not None:
        fitted_theoretical = [get_theoretical_survivability(day, fitted_failure_probability) for day in time_points]
        plt.plot(time_points, fitted_theoretical, label=f'Fitted Model (λ={fitted_failure_probability:.4f})', linestyle=':', linewidth=2)
    
    plt.xlabel('Days', fontsize=12)
    plt.ylabel('Survivability Rate', fontsize=12)
    
    title = f'Lightbulb Survivability Over Time ({lifespan_in_days} days, {len(experimental_results)} samples)'
    if fitted_failure_probability is not None:
        title += f'\nOriginal λ={failure_probability:.4f} → Fitted λ={fitted_failure_probability:.4f}'
    
    plt.title(title, fontsize=14, fontweight='bold')
    plt.legend(fontsize=11, loc='best')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('lightbulb_survivability_plot.png', dpi=300)
    print("✓ Plot saved as 'lightbulb_survivability_plot.png'")

if __name__ == '__main__':
    print("=" * 70)
    print("LIGHTBULB SURVIVABILITY SIMULATION")
    print("=" * 70)
    
    # Experiment 1: Known failure probability
    print("\n[Experiment 1] Known Failure Probability")
    print("-" * 70)
    
    experiment_settings = {
        'lifespan_in_days': 5000,
        'failure_probability': 0.001,
        'quantity_of_runs': 1000
    }
    
    experiments_outcomes = series_of_experiments(
        experiment_settings['quantity_of_runs'],
        experiment_settings['lifespan_in_days'],
        experiment_settings['failure_probability']
    )
    print(f"Average Survivability over {experiment_settings['lifespan_in_days']} days: {experiments_outcomes:.4f}")

    theoretical_survivability = get_theoretical_survivability(
        experiment_settings['lifespan_in_days'],
        experiment_settings['failure_probability']
    )
    print(f"Theoretical Survivability over {experiment_settings['lifespan_in_days']} days: {theoretical_survivability:.4f}")
    print(f"Difference: {abs(experiments_outcomes - theoretical_survivability):.4f}")

    time_points = list(range(0, experiment_settings['lifespan_in_days'] + 1, 500))
    experimental_results = [series_of_experiments(
        experiment_settings['quantity_of_runs'],
        days,
        experiment_settings['failure_probability']) for days in time_points]
    theoretical_results = [get_theoretical_survivability(
        days,
        experiment_settings['failure_probability']) for days in time_points]
    
    plot_experiment_outcomes(
        time_points=time_points,
        experimental_results=experimental_results,
        theoretical_results=theoretical_results,
        failure_probability=experiment_settings['failure_probability'],
        lifespan_in_days=experiment_settings['lifespan_in_days']
    )
    
    # Experiment 2: Unknown failure probability (random)
    print("\n[Experiment 2] Unknown Failure Probability (Random)")
    print("-" * 70)
    
    unknown_failure_probability = round(random.uniform(0.0005, 0.002), 4)
    print(f"Hidden failure probability: λ = {unknown_failure_probability:.4f}")
    
    experiment_settings_2 = {
        'lifespan_in_days': 5000,
        'failure_probability': unknown_failure_probability,
        'quantity_of_runs': 1000
    }
    
    observed_survivability = series_of_experiments(
        experiment_settings_2['quantity_of_runs'],
        experiment_settings_2['lifespan_in_days'],
        experiment_settings_2['failure_probability']
    )
    print(f"Observed Survivability: {observed_survivability:.4f}")
    
    fitted_lambda = fit_failure_probability(
        experiment_settings_2['lifespan_in_days'],
        observed_survivability
    )
    print(f"Fitted failure probability: λ = {fitted_lambda:.4f}")
    print(f"Estimation error: {abs(fitted_lambda - unknown_failure_probability):.6f}")
    
    time_points_2 = list(range(0, experiment_settings_2['lifespan_in_days'] + 1, 500))
    experimental_results_2 = [series_of_experiments(
        experiment_settings_2['quantity_of_runs'],
        days,
        experiment_settings_2['failure_probability']) for days in time_points_2]
    theoretical_results_2 = [get_theoretical_survivability(
        days,
        experiment_settings_2['failure_probability']) for days in time_points_2]
    
    plot_experiment_outcomes(
        time_points=time_points_2,
        experimental_results=experimental_results_2,
        theoretical_results=theoretical_results_2,
        failure_probability=experiment_settings_2['failure_probability'],
        lifespan_in_days=experiment_settings_2['lifespan_in_days'],
        fitted_failure_probability=fitted_lambda
    )
    plt.savefig('lightbulb_survivability_plot_fitted.png', dpi=300)
    print("✓ Fitted plot saved as 'lightbulb_survivability_plot_fitted.png'")
    
    print("\n" + "=" * 70)
    print("Simulations complete!")
    print("=" * 70)