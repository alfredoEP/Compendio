#!/usr/bin/env python

__author__ = "Alfredo Espinoza"

"""
monty_hall.py: This script contains a framework to simulate the thought experiment based on the 
Monty Hall's game show "Let's Make a Deal".
Here we run the experiment multiple times and average their winning results.
"""

import random

def monty_hall_show(
    switch_policy: bool=True
)->bool:
    """
    Run of a Monty Hall show with their outcome.

    Parameters:
    switching_policy: We simulate the contestants decision

    Returns:
    bool, either it is True or False that the contestant wins
    """
    
    win = False
    choices = ['goat1', 'goat2', 'car']
    
    random.shuffle(choices)
    contestants_choice = random.choice(choices)
    choices.remove(contestants_choice)
    if 'car' not in choices:
        montys_reveal = random.choice(choices)
        choices.remove(montys_reveal)
    else:
        choices.remove('car')
        montys_reveal = choices[0]
        choices = ['car']
    if switch_policy:
        contestants_choice = choices[0]
    if contestants_choice == 'car':
        win = True
    return win

def series_of_experiments(
    quantity_of_runs: int=1000,
    switching_policy: bool=True
)-> float:
    """
    Multiple runs and get their average.

    Parameters:
    quantity_or_runs: How many times we want to run a Monty Hall show.
    switching_policy: We simulate the contestants decision

    Returns:
    float, the average of the winninings
    """
    experiment_outcomes = [ 1 if monty_hall_show(switching_policy) else 0 for _ in range(quantity_of_runs)  ]
    avg_experiment_outcomes = sum(experiment_outcomes) / quantity_of_runs
    return avg_experiment_outcomes


if __name__ == '__main__':
    q = 1000
    print(f'Switching policy: {series_of_experiments(q, 1)}')
    print(f'Non switching policy: {series_of_experiments(q, 0)}')
