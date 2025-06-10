# from typing import Dict, List, Any
# import itertools
#
# class VariationCombiner:
#     """
#     Combines variations across multiple axes to create a comprehensive test suite.
#     """
#
#     def __init__(self, max_combinations: int = 100):
#         """
#         Initialize the combiner.
#
#         Args:
#             max_combinations: Maximum number of combinations to generate
#         """
#         self.max_combinations = max_combinations
#
#     def combine(self, variations_by_axis: Dict[str, List[str]]) -> List[str]:
#         """
#         Generate combinations of variations across multiple axes.
#
#         Args:
#             variations_by_axis: Dictionary mapping axis names to lists of variations
#
#         Returns:
#             List of combined prompt variations
#         """
#         if not variations_by_axis:
#             return []
#
#         # Extract the list of variations for each axis
#         variation_lists = list(variations_by_axis.values())
#
#         # Generate all combinations
#         all_combinations = list(itertools.product(*variation_lists))
#
#         # Limit the number of combinations if necessary
#         if len(all_combinations) > self.max_combinations:
#             # Sample a subset of combinations
#             import random
#             all_combinations = random.sample(all_combinations, self.max_combinations)
#
#         # For each combination, use the first variation as the base
#         # and apply the changes from other axes
#         combined_variations = []
#         for combination in all_combinations:
#             # Start with the first variation
#             combined = combination[0]
#             combined_variations.append(combined)
#
#         return combined_variations