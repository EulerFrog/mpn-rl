#!/usr/bin/env python3
"""
Generate random verb-noun combinations for experiment names
Usage: python generate_exp_name.py
Output: running-tiger
"""

import random

VERBS = [
    'running', 'jumping', 'flying', 'swimming', 'dancing', 'climbing',
    'walking', 'racing', 'spinning', 'gliding', 'soaring', 'diving',
    'dashing', 'leaping', 'sprinting', 'flowing', 'rushing', 'charging'
]

NOUNS = [
    'tiger', 'lion', 'eagle', 'shark', 'wolf', 'bear', 'hawk', 'falcon',
    'dragon', 'phoenix', 'panther', 'cheetah', 'leopard', 'jaguar',
    'cobra', 'viper', 'raven', 'condor', 'lynx', 'puma', 'fox', 'otter',
    'neuron', 'synapse', 'cortex', 'network', 'circuit', 'matrix', 'tensor'
]

def generate_suffix():
    """Generate random verb-noun combination"""
    verb = random.choice(VERBS)
    noun = random.choice(NOUNS)
    return f"{verb}-{noun}"

if __name__ == "__main__":
    print(generate_suffix())
