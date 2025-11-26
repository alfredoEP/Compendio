# !/usr/bin/env python

__author__ = "Alfredo Espinoza"

"""
hopfield_associative_memory.py: This script contains a framework to simulate a Hopfield Associative Memory network.
Here we train the network to memorize a capital letter 'A' in a 20x20 binary matrix,
then test retrieval with increasing levels of noise (10% to 90%).
"""

import random
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

LETTER_A_MATRIX = [
        [-1, -1, -1, -1, -1, -1, -1, -1, -1,  1,  1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [-1, -1, -1, -1, -1, -1, -1, -1,  1,  1,  1,  1, -1, -1, -1, -1, -1, -1, -1, -1],
        [-1, -1, -1, -1, -1, -1, -1,  1,  1,  1,  1,  1,  1, -1, -1, -1, -1, -1, -1, -1],
        [-1, -1, -1, -1, -1, -1,  1,  1,  1, -1, -1,  1,  1,  1, -1, -1, -1, -1, -1, -1],
        [-1, -1, -1, -1, -1,  1,  1,  1, -1, -1, -1, -1,  1,  1,  1, -1, -1, -1, -1, -1],
        [-1, -1, -1, -1,  1,  1,  1, -1, -1, -1, -1, -1, -1,  1,  1,  1, -1, -1, -1, -1],
        [-1, -1, -1,  1,  1,  1, -1, -1, -1, -1, -1, -1, -1, -1,  1,  1,  1, -1, -1, -1],
        [-1, -1,  1,  1,  1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,  1,  1,  1, -1, -1],
        [-1, -1,  1,  1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,  1,  1, -1, -1],
        [-1,  1,  1,  1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,  1,  1,  1, -1],
        [-1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1, -1],
        [-1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1, -1],
        [-1,  1,  1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,  1,  1, -1],
        [-1,  1,  1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,  1,  1, -1],
        [ 1,  1,  1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,  1,  1,  1],
        [ 1,  1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,  1,  1],
        [ 1,  1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,  1,  1],
        [ 1,  1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,  1,  1],
        [ 1,  1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,  1,  1],
        [ 1,  1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,  1,  1]
    ]

LETTER_T_MATRIX = [
        [ 1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1],
        [ 1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1],
        [ 1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1],
        [-1, -1, -1, -1, -1, -1, -1, -1,  1,  1,  1,  1, -1, -1, -1, -1, -1, -1, -1, -1],
        [-1, -1, -1, -1, -1, -1, -1, -1,  1,  1,  1,  1, -1, -1, -1, -1, -1, -1, -1, -1],
        [-1, -1, -1, -1, -1, -1, -1, -1,  1,  1,  1,  1, -1, -1, -1, -1, -1, -1, -1, -1],
        [-1, -1, -1, -1, -1, -1, -1, -1,  1,  1,  1,  1, -1, -1, -1, -1, -1, -1, -1, -1],
        [-1, -1, -1, -1, -1, -1, -1, -1,  1,  1,  1,  1, -1, -1, -1, -1, -1, -1, -1, -1],
        [-1, -1, -1, -1, -1, -1, -1, -1,  1,  1,  1,  1, -1, -1, -1, -1, -1, -1, -1, -1],
        [-1, -1, -1, -1, -1, -1, -1, -1,  1,  1,  1,  1, -1, -1, -1, -1, -1, -1, -1, -1],
        [-1, -1, -1, -1, -1, -1, -1, -1,  1,  1,  1,  1, -1, -1, -1, -1, -1, -1, -1, -1],
        [-1, -1, -1, -1, -1, -1, -1, -1,  1,  1,  1,  1, -1, -1, -1, -1, -1, -1, -1, -1],
        [-1, -1, -1, -1, -1, -1, -1, -1,  1,  1,  1,  1, -1, -1, -1, -1, -1, -1, -1, -1],
        [-1, -1, -1, -1, -1, -1, -1, -1,  1,  1,  1,  1, -1, -1, -1, -1, -1, -1, -1, -1],
        [-1, -1, -1, -1, -1, -1, -1, -1,  1,  1,  1,  1, -1, -1, -1, -1, -1, -1, -1, -1],
        [-1, -1, -1, -1, -1, -1, -1, -1,  1,  1,  1,  1, -1, -1, -1, -1, -1, -1, -1, -1],
        [-1, -1, -1, -1, -1, -1, -1, -1,  1,  1,  1,  1, -1, -1, -1, -1, -1, -1, -1, -1],
        [-1, -1, -1, -1, -1, -1, -1, -1,  1,  1,  1,  1, -1, -1, -1, -1, -1, -1, -1, -1],
        [-1, -1, -1, -1, -1, -1, -1, -1,  1,  1,  1,  1, -1, -1, -1, -1, -1, -1, -1, -1],
        [-1, -1, -1, -1, -1, -1, -1, -1,  1,  1,  1,  1, -1, -1, -1, -1, -1, -1, -1, -1]
    ]

LETTER_F_MATRIX = [
        [ 1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1],
        [ 1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1],
        [ 1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1],
        [ 1,  1,  1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [ 1,  1,  1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [ 1,  1,  1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [ 1,  1,  1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [ 1,  1,  1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [ 1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1, -1, -1, -1, -1, -1, -1],
        [ 1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1, -1, -1, -1, -1, -1, -1],
        [ 1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1, -1, -1, -1, -1, -1, -1],
        [ 1,  1,  1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [ 1,  1,  1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [ 1,  1,  1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [ 1,  1,  1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [ 1,  1,  1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [ 1,  1,  1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [ 1,  1,  1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [ 1,  1,  1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [ 1,  1,  1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]
    ]

LETTER_O_MATRIX = [
        [-1, -1, -1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1, -1, -1, -1],
        [-1, -1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1, -1, -1],
        [-1,  1,  1,  1,  1, -1, -1, -1, -1, -1, -1, -1, -1, -1,  1,  1,  1,  1,  1, -1],
        [ 1,  1,  1,  1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,  1,  1,  1,  1,  1],
        [ 1,  1,  1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,  1,  1,  1,  1],
        [ 1,  1,  1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,  1,  1,  1,  1],
        [ 1,  1,  1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,  1,  1,  1,  1],
        [ 1,  1,  1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,  1,  1,  1,  1],
        [ 1,  1,  1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,  1,  1,  1,  1],
        [ 1,  1,  1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,  1,  1,  1,  1],
        [ 1,  1,  1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,  1,  1,  1,  1],
        [ 1,  1,  1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,  1,  1,  1,  1],
        [ 1,  1,  1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,  1,  1,  1,  1],
        [ 1,  1,  1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,  1,  1,  1,  1],
        [ 1,  1,  1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,  1,  1,  1,  1],
        [ 1,  1,  1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,  1,  1,  1,  1],
        [ 1,  1,  1,  1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,  1,  1,  1,  1,  1],
        [-1,  1,  1,  1,  1, -1, -1, -1, -1, -1, -1, -1, -1, -1,  1,  1,  1,  1,  1, -1],
        [-1, -1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1, -1, -1],
        [-1, -1, -1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1, -1, -1, -1]
    ]

LETTER_U_MATRIX = [
        [ 1,  1,  1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,  1,  1,  1,  1],
        [ 1,  1,  1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,  1,  1,  1,  1],
        [ 1,  1,  1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,  1,  1,  1,  1],
        [ 1,  1,  1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,  1,  1,  1,  1],
        [ 1,  1,  1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,  1,  1,  1,  1],
        [ 1,  1,  1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,  1,  1,  1,  1],
        [ 1,  1,  1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,  1,  1,  1,  1],
        [ 1,  1,  1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,  1,  1,  1,  1],
        [ 1,  1,  1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,  1,  1,  1,  1],
        [ 1,  1,  1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,  1,  1,  1,  1],
        [ 1,  1,  1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,  1,  1,  1,  1],
        [ 1,  1,  1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,  1,  1,  1,  1],
        [ 1,  1,  1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,  1,  1,  1,  1],
        [ 1,  1,  1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,  1,  1,  1,  1],
        [ 1,  1,  1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,  1,  1,  1,  1],
        [ 1,  1,  1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,  1,  1,  1,  1],
        [ 1,  1,  1,  1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,  1,  1,  1,  1,  1],
        [-1,  1,  1,  1,  1, -1, -1, -1, -1, -1, -1, -1, -1, -1,  1,  1,  1,  1,  1, -1],
        [-1, -1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1, -1, -1],
        [-1, -1, -1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1, -1, -1, -1]
    ]

LETTER_V_MATRIX = [
        [ 1,  1,  1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,  1,  1,  1],
        [ 1,  1,  1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,  1,  1,  1],
        [ 1,  1,  1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,  1,  1,  1],
        [ 1,  1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,  1,  1],
        [ 1,  1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,  1,  1],
        [ 1,  1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,  1,  1],
        [-1,  1,  1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,  1,  1, -1],
        [-1,  1,  1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,  1,  1, -1],
        [-1,  1,  1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,  1,  1, -1],
        [-1, -1,  1,  1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,  1,  1, -1, -1],
        [-1, -1,  1,  1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,  1,  1, -1, -1],
        [-1, -1,  1,  1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,  1,  1, -1, -1],
        [-1, -1, -1,  1,  1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,  1,  1, -1, -1, -1],
        [-1, -1, -1,  1,  1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,  1,  1, -1, -1, -1],
        [-1, -1, -1,  1,  1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,  1,  1, -1, -1, -1],
        [-1, -1, -1, -1,  1,  1,  1, -1, -1, -1, -1, -1, -1,  1,  1,  1, -1, -1, -1, -1],
        [-1, -1, -1, -1,  1,  1,  1, -1, -1, -1, -1, -1, -1,  1,  1,  1, -1, -1, -1, -1],
        [-1, -1, -1, -1, -1,  1,  1,  1, -1, -1, -1, -1,  1,  1,  1, -1, -1, -1, -1, -1],
        [-1, -1, -1, -1, -1,  1,  1,  1, -1, -1, -1, -1,  1,  1,  1, -1, -1, -1, -1, -1],
        [-1, -1, -1, -1, -1, -1,  1,  1,  1,  1,  1,  1,  1,  1, -1, -1, -1, -1, -1, -1]
    ]

LETTER_R_MATRIX = [
        [ 1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1, -1, -1, -1, -1, -1],
        [ 1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1, -1, -1, -1, -1],
        [ 1,  1,  1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,  1,  1,  1,  1, -1, -1, -1],
        [ 1,  1,  1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,  1,  1,  1, -1, -1, -1],
        [ 1,  1,  1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,  1,  1,  1, -1, -1, -1],
        [ 1,  1,  1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,  1,  1,  1, -1, -1, -1],
        [ 1,  1,  1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,  1,  1,  1, -1, -1, -1],
        [ 1,  1,  1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,  1,  1,  1,  1, -1, -1, -1],
        [ 1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1, -1, -1, -1, -1],
        [ 1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1, -1, -1, -1, -1, -1],
        [ 1,  1,  1, -1, -1, -1, -1, -1, -1,  1,  1,  1,  1,  1, -1, -1, -1, -1, -1, -1],
        [ 1,  1,  1, -1, -1, -1, -1, -1, -1, -1,  1,  1,  1,  1,  1, -1, -1, -1, -1, -1],
        [ 1,  1,  1, -1, -1, -1, -1, -1, -1, -1, -1,  1,  1,  1,  1,  1, -1, -1, -1, -1],
        [ 1,  1,  1, -1, -1, -1, -1, -1, -1, -1, -1, -1,  1,  1,  1,  1, -1, -1, -1, -1],
        [ 1,  1,  1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,  1,  1,  1,  1, -1, -1, -1],
        [ 1,  1,  1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,  1,  1,  1, -1, -1, -1],
        [ 1,  1,  1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,  1,  1,  1,  1, -1, -1],
        [ 1,  1,  1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,  1,  1,  1, -1, -1],
        [ 1,  1,  1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,  1,  1,  1,  1, -1],
        [ 1,  1,  1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,  1,  1,  1,  1]
    ]

LETTER_P_MATRIX = [
        [ 1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1, -1, -1, -1, -1],
        [ 1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1, -1, -1, -1],
        [ 1,  1,  1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,  1,  1,  1,  1,  1, -1, -1],
        [ 1,  1,  1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,  1,  1,  1,  1, -1, -1],
        [ 1,  1,  1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,  1,  1,  1, -1, -1],
        [ 1,  1,  1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,  1,  1,  1, -1, -1],
        [ 1,  1,  1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,  1,  1,  1, -1, -1],
        [ 1,  1,  1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,  1,  1,  1,  1, -1, -1],
        [ 1,  1,  1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,  1,  1,  1,  1,  1, -1, -1],
        [ 1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1, -1, -1, -1],
        [ 1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1, -1, -1, -1, -1],
        [ 1,  1,  1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [ 1,  1,  1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [ 1,  1,  1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [ 1,  1,  1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [ 1,  1,  1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [ 1,  1,  1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [ 1,  1,  1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [ 1,  1,  1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [ 1,  1,  1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]
    ]

def create_memory_matrix(pattern_list: list) -> list:
    """
    Create a Hopfield memory matrix using Hebbian learning rule.
    Based on the original crear_memoria function from hop.py.
    
    Parameters:
    pattern_list: List of patterns to memorize (each pattern is a flattened list).
    
    Returns:
    list, the weight matrix of the Hopfield network.
    """
    X = pattern_list
    matrix = []
    
    for num_x, x in enumerate(X):
        if num_x == 0:
            # Initialize matrix with first pattern
            for num_i, i in enumerate(x):
                row = []
                for num_j, j in enumerate(x):
                    if num_i == num_j:
                        row.append(0)  # No self-connections
                    else:
                        row.append(i * j)  # Hebbian rule: w_ij = x_i * x_j
                matrix.append(row)
        else:
            # Add subsequent patterns
            for num_i, i in enumerate(x):
                for num_j, j in enumerate(x):
                    if num_i == num_j:
                        matrix[num_i][num_j] = 0
                    else:
                        matrix[num_i][num_j] += i * j
    
    return matrix

def retrieve_pattern(
    memory: list,
    corrupted_pattern: list,
    max_iterations: int = 100
) -> list:
    """
    Retrieve a pattern from the Hopfield network.
    Based on the original recuperar_patrón function from hop.py.
    
    Parameters:
    memory: The weight matrix of the Hopfield network.
    corrupted_pattern: The initial (possibly noisy) pattern.
    max_iterations: Maximum number of update iterations.
    
    Returns:
    list, the retrieved pattern.
    """
    x = corrupted_pattern.copy()
    
    for iteration in range(max_iterations):
        x_old = x.copy()
        retrieved = []
        
        # Update all neurons
        for row in memory:
            activation = 0
            for weight, state in zip(row, x):
                activation += weight * state
            
            # Apply activation function
            if activation != 0:
                if activation < 0:
                    retrieved.append(-1)
                else:
                    retrieved.append(1)
            else:
                retrieved.append(0)
        
        x = retrieved
        
        # Check for convergence
        if x == x_old:
            break
    
    return x

def add_noise_to_pattern(pattern: list, noise_level: float) -> list:
    """
    Add random noise to a binary pattern by flipping random bits.
    
    Parameters:
    pattern: The original binary pattern (values should be 1 or -1).
    noise_level: Fraction of bits to flip (0.0 to 1.0).
    
    Returns:
    list, the noisy pattern.
    """
    noisy_pattern = pattern.copy()
    num_bits_to_flip = int(len(pattern) * noise_level)
    
    for _ in range(num_bits_to_flip):
        index = random.randint(0, len(pattern) - 1)
        noisy_pattern[index] = random.choice([-1, 1])
    
    return noisy_pattern

def calculate_accuracy(original: list, retrieved: list) -> float:
    """
    Calculate the accuracy between original and retrieved patterns.
    
    Parameters:
    original: The original pattern.
    retrieved: The retrieved pattern.
    
    Returns:
    float, accuracy percentage (0-100).
    """
    matches = sum(1 for o, r in zip(original, retrieved) if o == r)
    accuracy = (matches / len(original)) * 100
    return accuracy

def pattern_to_matrix(pattern: list, size: int = 20) -> np.ndarray:
    """
    Convert a flattened pattern list to a 2D matrix for visualization.
    
    Parameters:
    pattern: Flattened pattern list.
    size: Side length of the square matrix.
    
    Returns:
    np.ndarray, 2D matrix representation.
    """
    matrix = np.array(pattern).reshape(size, size)
    return matrix

def predict_letter(retrieved_pattern: list, reference_patterns: dict) -> str:
    """
    Predict which letter a retrieved pattern corresponds to by finding the best match.
    
    Parameters:
    retrieved_pattern: The pattern retrieved from the Hopfield network.
    reference_patterns: Dictionary mapping letter names to their original patterns.
    
    Returns:
    str, the name of the predicted letter.
    """
    best_match = None
    best_accuracy = -1
    
    for letter_name, reference_pattern in reference_patterns.items():
        accuracy = calculate_accuracy(reference_pattern, retrieved_pattern)
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_match = letter_name
    
    return best_match

def plot_hopfield_results(
    original_pattern: list,
    noisy_patterns: list,
    retrieved_patterns: list,
    noise_levels: list,
    accuracies: list,
    matrix_size: int = 20,
    letter_name: str = 'A',
    reference_patterns: dict = None
) -> None:
    """
    Create a 3x3 grid visualization of the retrieval results.
    Each subplot shows noisy input (left) and retrieved pattern (right) side by side.
    
    Parameters:
    original_pattern: The original memorized pattern.
    noisy_patterns: List of noisy input patterns.
    retrieved_patterns: List of retrieved patterns.
    noise_levels: List of noise levels used.
    accuracies: List of accuracy percentages.
    matrix_size: Size of the pattern matrix.
    letter_name: The name of the letter being tested.
    reference_patterns: Dictionary of all reference patterns for prediction.
    """
    fig = plt.figure(figsize=(15, 12))
    gs = GridSpec(3, 3, figure=fig, hspace=0.5, wspace=0.4)
    
    for idx in range(9):
        ax = fig.add_subplot(gs[idx // 3, idx % 3])
        
        if idx < len(retrieved_patterns):
            noise_pct = int(noise_levels[idx] * 100)
            accuracy = accuracies[idx]
            
            # Predict which letter was retrieved
            predicted_letter = predict_letter(retrieved_patterns[idx], reference_patterns) if reference_patterns else '?'
            
            # Convert patterns to matrices for display
            noisy_matrix = pattern_to_matrix(noisy_patterns[idx], matrix_size)
            retrieved_matrix = pattern_to_matrix(retrieved_patterns[idx], matrix_size)
            
            # Create side-by-side comparison
            # Add a separator column between the two images
            separator = np.ones((matrix_size, 2)) * -1
            combined = np.hstack([noisy_matrix, separator, retrieved_matrix])
            
            # Display comparison
            ax.imshow(combined, cmap='gray', interpolation='nearest')
            ax.set_title(f'{noise_pct}% Noise → Accuracy: {accuracy:.1f}%\nActual: {letter_name}; Predicted: {predicted_letter}', 
                        fontsize=10, fontweight='bold')
            ax.axis('off')
        else:
            ax.axis('off')
    
    plt.suptitle(f'Hopfield Associative Memory: Letter {letter_name} Retrieval\nNetwork Trained on A, T, F, O, U, V, R, P | Noisy Input vs Retrieved Pattern',
                fontsize=14, fontweight='bold', y=0.98)
    
    filename = f'./assets/hopfield_associative_memory_{letter_name}.png'
    plt.savefig(filename, dpi=100, bbox_inches='tight')
    plt.close(fig)
    print(f"✓ Plot saved as '{filename}'")

if __name__ == '__main__':
    print("=" * 70)
    print("HOPFIELD ASSOCIATIVE MEMORY SIMULATION")
    print("=" * 70)
    
    # ============ CONFIGURATION ============
    MATRIX_SIZE = 20
    NOISE_LEVELS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    MAX_RETRIEVAL_ITERATIONS = 10
    # =======================================
    
    print("\n[Step 1] Creating Letter Patterns")
    print("-" * 70)
    # Flatten all letter patterns
    pattern_A = [pixel for row in LETTER_A_MATRIX for pixel in row]
    pattern_T = [pixel for row in LETTER_T_MATRIX for pixel in row]
    pattern_F = [pixel for row in LETTER_F_MATRIX for pixel in row]
    pattern_O = [pixel for row in LETTER_O_MATRIX for pixel in row]
    pattern_U = [pixel for row in LETTER_U_MATRIX for pixel in row]
    pattern_V = [pixel for row in LETTER_V_MATRIX for pixel in row]
    pattern_R = [pixel for row in LETTER_R_MATRIX for pixel in row]
    pattern_P = [pixel for row in LETTER_P_MATRIX for pixel in row]
    
    print(f"Pattern size: {MATRIX_SIZE}x{MATRIX_SIZE} = {len(pattern_A)} neurons")
    print(f"Created patterns for letters: A, T, F, O, U, V, R, P")
    active_pixels_A = sum(1 for p in pattern_A if p == 1)
    print(f"Letter A has {active_pixels_A} active pixels")
    
    print("\n[Step 2] Training Hopfield Network")
    print("-" * 70)
    patterns_to_memorize = [pattern_A, pattern_T, pattern_F, pattern_O, pattern_U, pattern_V, pattern_R, pattern_P]
    memory_matrix = create_memory_matrix(patterns_to_memorize)
    print(f"Weight matrix size: {len(memory_matrix)}x{len(memory_matrix[0])}")
    print(f"Network trained with {len(patterns_to_memorize)} pattern(s): A, T, F, O, U, V, R, P")
    
    # Test all patterns
    test_patterns = [
        ('A', pattern_A),
        ('T', pattern_T),
        ('F', pattern_F),
        ('O', pattern_O),
        ('U', pattern_U),
        ('V', pattern_V),
        ('R', pattern_R),
        ('P', pattern_P)
    ]
    
    print("\n[Step 3] Testing Retrieval for All Patterns with Varying Noise Levels")
    print("=" * 70)
    
    all_results = []
    
    # Create reference patterns dictionary for prediction
    reference_patterns = {
        'A': pattern_A,
        'T': pattern_T,
        'F': pattern_F,
        'O': pattern_O,
        'U': pattern_U,
        'V': pattern_V,
        'R': pattern_R,
        'P': pattern_P
    }
    
    for letter_name, original_pattern in test_patterns:
        print(f"\n{'=' * 70}")
        print(f"Testing Letter: {letter_name}")
        print("=" * 70)
        
        noisy_patterns = []
        retrieved_patterns = []
        accuracies = []
        predictions = []
        
        for noise_level in NOISE_LEVELS:
            print(f"\nNoise Level: {int(noise_level * 100)}%")
            
            # Add noise to the original pattern
            noisy_pattern = add_noise_to_pattern(original_pattern, noise_level)
            noisy_patterns.append(noisy_pattern)
            
            # Calculate initial accuracy
            initial_accuracy = calculate_accuracy(original_pattern, noisy_pattern)
            print(f"  Initial accuracy: {initial_accuracy:.2f}%")
            
            # Retrieve pattern from network
            retrieved = retrieve_pattern(memory_matrix, noisy_pattern, MAX_RETRIEVAL_ITERATIONS)
            retrieved_patterns.append(retrieved)
            
            # Predict which letter was retrieved
            predicted_letter = predict_letter(retrieved, reference_patterns)
            predictions.append(predicted_letter)
            
            # Calculate final accuracy
            final_accuracy = calculate_accuracy(original_pattern, retrieved)
            accuracies.append(final_accuracy)
            
            print(f"  Final accuracy: {final_accuracy:.2f}%")
            print(f"  Predicted letter: {predicted_letter}")
        
        # Count correct predictions
        correct_predictions = sum(1 for pred in predictions if pred == letter_name)
        
        # Store results for summary
        all_results.append({
            'letter': letter_name,
            'accuracies': accuracies,
            'successful': sum(1 for acc in accuracies if acc > 90),
            'average': np.mean(accuracies),
            'correct_predictions': correct_predictions,
            'predictions': predictions
        })
        
        print(f"\n[Step 4] Generating Visualization for Letter {letter_name}")
        print("-" * 70)
        plot_hopfield_results(
            original_pattern,
            noisy_patterns,
            retrieved_patterns,
            NOISE_LEVELS,
            accuracies,
            MATRIX_SIZE,
            letter_name,
            reference_patterns
        )
        
        print(f"\nSummary for Letter {letter_name}:")
        print("-" * 70)
        successful_retrievals = sum(1 for acc in accuracies if acc > 90)
        print(f"Successful retrievals (>90% accuracy): {successful_retrievals}/{len(NOISE_LEVELS)}")
        print(f"Average final accuracy: {np.mean(accuracies):.2f}%")
        print(f"Best retrieval: {max(accuracies):.2f}% (at {int(NOISE_LEVELS[accuracies.index(max(accuracies))] * 100)}% noise)")
        print(f"Worst retrieval: {min(accuracies):.2f}% (at {int(NOISE_LEVELS[accuracies.index(min(accuracies))] * 100)}% noise)")
    
    print("\n" + "=" * 70)
    print("OVERALL SUMMARY STATISTICS")
    print("=" * 70)
    for result in all_results:
        print(f"\nLetter {result['letter']}:")
        print(f"  Successful retrievals (>90%): {result['successful']}/{len(NOISE_LEVELS)}")
        print(f"  Average accuracy: {result['average']:.2f}%")
        print(f"  Correct predictions: {result['correct_predictions']}/{len(NOISE_LEVELS)}")
        print(f"  Prediction accuracy: {(result['correct_predictions'] / len(NOISE_LEVELS)) * 100:.1f}%")
    
    print("\n" + "=" * 70)
    print("Simulation complete!")
    print("=" * 70)
