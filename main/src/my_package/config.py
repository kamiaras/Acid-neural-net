"""
Configuration settings for my_package:

- categories_map: mapping of each categorical feature name to its allowed integer codes.
- all_inputs:     list of all feature names (Inputs 1–18, excluding Input 12).
- numeric_inputs: subset of all_inputs treated as continuous for normalization (Inputs 9–18, excluding 12).
"""

# Map of categorical inputs to their categories
categories_map: dict[str, list[int]] = {
    'Input 1':  [1, 2],             # binary feature
    'Input 2':  list(range(1, 6)),  # 5-level categorical
    'Input 3':  list(range(1, 6)),
    'Input 4':  list(range(1, 22)), # 21-level categorical
    'Input 5':  list(range(1, 6)),
    'Input 6':  [1, 2, 3],          # 3-level categorical
    'Input 7':  [1, 2, 3, 4],       # 4-level categorical
    'Input 8':  list(range(1, 11)), # 10-level categorical
}

# All feature names except Input 12 (which is unused)
all_inputs: list[str] = [
    f'Input {i}' for i in range(1, 19) if i != 12
]

# Numeric (continuous) features to be normalized
numeric_inputs: list[str] = [
    f'Input {i}' for i in range(9, 19) if i != 12
]
