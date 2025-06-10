# """
# A simplified script that applies augmentation based on dimensions in annotations.
# """
# import argparse
# import json
# from typing import Dict, List, Any
#
# import pandas as pd
#
# from src.axis_augmentation.augmentation_pipeline import AugmentationPipeline
# from src.axis_augmentation.context_augmenter import ContextAugmenter
# from src.axis_augmentation.fewshot_augmenter import FewShotAugmenter
# from src.axis_augmentation.multidoc_augmenter import MultiDocAugmenter
# from src.axis_augmentation.multiple_choice_augmenter import MultipleChoiceAugmenter
# from src.axis_augmentation.paraphrase_instruct import Paraphrase
# from src.axis_augmentation.text_surface_augmenter import TextSurfaceAugmenter
# from src.utils.constants import (
#     DEFAULT_ANNOTATIONS_INPUT_FILE,
#     DEFAULT_AUGMENTED_VARIATIONS_OUTPUT_FILE,
#     DEFAULT_VARIATIONS_PER_AXIS
# )
#
# # Define mapping between dimensions and augmenter classes
# DIMENSION_TO_AUGMENTER = {
#     "Paraphrases": Paraphrase,
#     "Non-semantic / structural changes": TextSurfaceAugmenter,
#     "Which few-shot examples": FewShotAugmenter,
#     "How many few-shot examples": FewShotAugmenter,
#     "Add irrelevant context": ContextAugmenter,
#     "Order of provided documents": MultiDocAugmenter,
#     "Enumeration (letters, numbers, etc)": MultipleChoiceAugmenter,
#     "Order of answers": MultipleChoiceAugmenter,
# }
#
#
# def load_annotations(file_path: str) -> List[Dict[str, Any]]:
#     """Load annotations from a JSON file."""
#     with open(file_path, 'r', encoding='utf-8') as f:
#         return json.load(f)
#
#
# def save_results(results: List[Dict[str, Any]], output_file: str):
#     """Save results to a JSON file."""
#     with open(output_file, 'w', encoding='utf-8') as f:
#         json.dump(results, f, indent=2, ensure_ascii=False)
#
#
# def augment_part(
#         text: str,
#         dimensions: List[str],
#         variant_counts: Dict[str, int],
#         part_name: str,
#         annotations: List[Dict[str, Any]],
#         current_index: int
# ) -> List[str]:
#     """
#     Augment a text based on its dimensions.
#
#     Args:
#         text: Text to augment
#         dimensions: List of dimensions to apply
#         variant_counts: Dictionary mapping dimension names to the number of variants to generate
#         part_name: Name of the part (for special handling)
#         annotations: List of all annotations
#         current_index: Index of the annotation being processed
#
#     Returns:
#         List of augmented texts
#     """
#     print(f"\n==== Augmenting {part_name} ====")
#     print(f"Dimensions selected: {dimensions}")
#     print(f"Variant counts: {variant_counts}")
#
#     if (not text and part_name != "examples") or not dimensions:
#         print("No text or dimensions - returning original text")
#         return [text]
#
#     # Select augmenters based on dimensions
#     augmenters = []
#     special_data = {}
#
#     for dim in dimensions:
#         # Get the variant count for this dimension
#         n_augments = variant_counts.get(dim, DEFAULT_VARIATIONS_PER_AXIS)
#         print(f"Processing dimension: {dim} with {n_augments} requested variants")
#
#         if dim in DIMENSION_TO_AUGMENTER:
#             augmenter_class = DIMENSION_TO_AUGMENTER[dim]
#             print(f"Using augmenter class: {augmenter_class.__name__}")
#
#             if augmenter_class == FewShotAugmenter:
#                 import pandas as pd
#
#                 few_shot_data = []
#                 for i, ann in enumerate(annotations):
#                     if i != current_index:
#                         if "context" in ann["annotations"] and ann["annotations"]["context"]:
#                             real_context = ann["annotations"]["context"]["text"]
#                         else:
#                             continue
#                         fs_output = ""
#                         if "output" in ann["annotations"] and ann["annotations"]["output"]:
#                             fs_output = ann["annotations"]["output"]["text"]
#                         elif "output" in ann and ann["output"]:
#                             fs_output = ann["output"]
#
#                         # Only add if we have a valid output
#                         if fs_output:
#                             few_shot_data.append({"input": real_context, "output": fs_output})
#                         else:
#                             print(f"Warning: No output found for example {i}")
#
#                 few_shot_df = pd.DataFrame(few_shot_data)
#                 special_data = {"dataset": few_shot_df}
#
#                 # Set the appropriate mode based on the dimension
#                 if dim == "Which few-shot examples":
#                     mode = "which"
#                 elif dim == "How many few-shot examples":
#                     mode = "how_many"
#                 else:
#                     mode = "both"
#
#                 # Create the augmenter with the appropriate mode
#                 augmenter = augmenter_class(n_augments=n_augments, num_examples=2, mode=mode)
#                 special_data["fewshot_mode"] = mode  # Pass the mode to the augmenter
#                 print(f"Created FewShotAugmenter with mode={mode}, n_augments={n_augments}")
#             else:
#                 augmenter = augmenter_class(n_augments=n_augments)
#                 print(f"Created {augmenter_class.__name__} with n_augments={n_augments}")
#
#             # Special handling for multiple choice
#             if augmenter_class == MultipleChoiceAugmenter and part_name == "choices":
#                 # Simple parsing of options (assuming format like "A) Option1 B) Option2")
#                 parts = text.split(")")
#                 markers = []
#                 options = []
#
#                 for i, part in enumerate(parts[:-1]):  # Skip the last part after final )
#                     marker = part.strip().split()[-1]  # Last word before the )
#                     markers.append(marker)
#
#                     # Get text between this marker and the next one
#                     if i < len(parts) - 2:
#                         next_marker_pos = parts[i + 1].rfind(parts[i + 1].strip().split()[-1])
#                         option_text = parts[i + 1][:next_marker_pos].strip()
#                     else:
#                         option_text = parts[i + 1].strip()
#
#                     options.append(option_text)
#
#                 special_data = {
#                     "question": "Placeholder question",  # You might want to get this from context
#                     "options": options,
#                     "markers": markers
#                 }
#
#             augmenters.append(augmenter)
#         else:
#             print(f"WARNING: No augmenter found for dimension {dim}")
#
#     # If no augmenters selected, return original text
#     if not augmenters:
#         print("No augmenters selected - returning original text")
#         return [text]
#
#     print(f"Created pipeline with {len(augmenters)} augmenters")
#
#     # Create pipeline with selected augmenters
#     pipeline = AugmentationPipeline(augmenters=augmenters, max_variations=10)
#
#     # Apply augmentation
#     variations = pipeline.augment(text, special_data)
#     print(f"Generated {len(variations)} total variations")
#     for i, var in enumerate(variations[:3]):  # Show first 3 variations only to avoid clutter
#         print(f"  Variation {i + 1}: {var[:50]}{'...' if len(var) > 50 else ''}")
#     if len(variations) > 3:
#         print(f"  ...and {len(variations) - 3} more variations")
#
#     return variations
#
#
# def process_annotations(annotations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
#     """Process all annotations and generate variations."""
#     all_results = []
#
#     for idx, annotation in enumerate(annotations):
#         # Get the placeholder format
#         placeholder_format = annotation["placeholder_prompt"]
#
#         # Create results for this annotation
#         result = {
#             "original_prompt": annotation["full_prompt"],
#             "variations": []
#         }
#
#         # Get augmented texts for each part
#         part_variations = {}
#
#         for part_name, part_data in annotation["annotations"].items():
#             if part_name == "output":
#                 continue
#             text = part_data["text"]
#             dimensions = part_data.get("dimensions", [])
#             variant_counts = part_data.get("variant_counts", {})
#             # if part_name == "examples":
#             #     text = annotation["annotations"]["context"]["text"]
#             variations = augment_part(
#                 text,
#                 dimensions,
#                 variant_counts,
#                 part_name,
#                 annotations,
#                 idx
#             )
#             part_variations[part_name] = variations
#             print(f"Generated {len(variations)} variations for {part_name}")
#
#         # Combine variations (limit to 10 combinations per annotation)
#         max_combinations = 20
#         count = 0
#
#         for task_desc in part_variations.get("task_description", [""]):
#             for context in part_variations.get("context", [""]):
#                 for examples in part_variations.get("examples", [""]):
#                     for choices in part_variations.get("choices", [""]):
#                         if count >= max_combinations:
#                             break
#
#                         # Create new prompt
#                         print(f"Start to replace placeholders for {idx} ")
#                         new_prompt = placeholder_format
#                         if pd.notna(task_desc) and task_desc != "":
#                             print(f"try to replace TASK_DESCRIPTION with {task_desc} ")
#                             new_prompt = new_prompt.replace("{TASK_DESCRIPTION}", task_desc)
#                         if "{EXAMPLES}" in new_prompt:
#                             if pd.notna(examples) and examples != "":
#                                 print(f"try to replace EXAMPLES with {examples} ")
#                                 new_prompt = new_prompt.replace("{EXAMPLES}", examples)
#                         else:
#                             if examples and pd.notna(examples) and examples != "":
#                                 print(f"try to replace CONTEXT with {examples}+ CONTEXT ")
#                                 new_prompt = new_prompt.replace("{CONTEXT}", examples + "\n" + "{CONTEXT}")
#                         if pd.notna(context) and context != "":
#                             print(f"try to replace CONTEXT with {context} ")
#                             new_prompt = new_prompt.replace("{CONTEXT}", context)
#                         if pd.notna(choices) and choices != "":
#                             print(f"try to replace CHOICES with {choices} ")
#                             new_prompt = new_prompt.replace("{CHOICES}", choices)
#
#                         print(f"Finish to replace placeholders for {idx} ")
#                         variation_obj = {
#                             "final_prompt": new_prompt,
#                             "parts": {
#                                 "task_description": task_desc,
#                                 "context": context,
#                                 "examples": examples,
#                                 "choices": choices
#                             }
#                         }
#
#                         result["variations"].append(variation_obj)
#                         count += 1
#
#                     if count >= max_combinations:
#                         break
#                 if count >= max_combinations:
#                     break
#             if count >= max_combinations:
#                 break
#
#         all_results.append(result)
#
#     return all_results
#
#
# def main(annotations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
#     """Main function to run the annotation augmentation process."""
#     # Set input and output paths
#     print(f"Loaded {len(annotations)} annotations.")
#
#     print("Processing annotations...")
#     results = process_annotations(annotations)
#     print(f"Generated variations for {len(results)} annotations.")
#     return results
#
#
# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description="Simple Augmenter Script")
#     parser.add_argument(
#         "--input_file",
#         type=str,
#         default=DEFAULT_ANNOTATIONS_INPUT_FILE,
#         help="Path to the input JSON file containing annotations."
#     )
#     parser.add_argument(
#         "--output_file",
#         type=str,
#         default=DEFAULT_AUGMENTED_VARIATIONS_OUTPUT_FILE,
#         help="Path to the output JSON file for augmented results."
#     )
#     args = parser.parse_args()
#
#     print(f"Loading annotations from {args.input_file}...")
#     annotations = load_annotations(args.input_file)
#
#     results = main(annotations)
#     print(f"Saving results to {args.output_file}...")
#     save_results(results, args.output_file)
#     print("Done!")
