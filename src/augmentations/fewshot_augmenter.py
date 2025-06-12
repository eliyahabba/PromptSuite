from typing import Dict, List, Any
import pandas as pd
import random

from src.augmentations.base_augmenter import BaseAxisAugmenter
from src.shared.constants import FewShotConstants, BaseAugmenterConstants


class FewShotAugmenter(BaseAxisAugmenter):
    """
    This augmenter handles few-shot examples for question answering tasks.
    It can vary either the specific examples used or the number of examples.
    """

    def __init__(self, 
                 n_augments: int = BaseAugmenterConstants.DEFAULT_N_AUGMENTS,
                 num_examples: int = FewShotConstants.DEFAULT_NUM_EXAMPLES, 
                 mode: str = "both"):  # Add mode parameter
        """
        Initialize the few-shot augmenter.
        
        Args:
            n_augments: Number of variations to generate
            num_examples: Number of examples to include for each question
            mode: Operation mode - "which" (vary examples), "how_many" (vary count), or "both"
        """
        super().__init__(n_augments=n_augments)
        self.num_examples = num_examples
        self.dataset = None
        self.mode = mode  # Store the mode
        
        # For "how_many" mode, we'll use these counts
        self.example_counts = [1, 2, 3, 5] if num_examples >= 5 else list(range(1, num_examples + 1))

    def get_name(self):
        if self.mode == "which":
            return "Which Few-Shot Examples"
        elif self.mode == "how_many":
            return "How Many Few-Shot Examples"
        else:
            return "Few-Shot Examples"

    def set_dataset(self, dataset: pd.DataFrame):
        """
        Set the dataset to use for few-shot examples.
        
        Args:
            dataset: DataFrame with 'input' and 'output' columns
        """
        if "input" not in dataset.columns or "output" not in dataset.columns:
            raise ValueError("Dataset must contain columns - 'input', 'output'")
        self.dataset = dataset

    def augment(self, prompt: str, identification_data: Dict[str, Any] = None) -> List[str]:
        """
        Generate few-shot variations of the prompt based on the selected mode.
        
        Args:
            prompt: The original prompt text
            identification_data: Optional data containing a dataset to use
            
        Returns:
            List of variations with few-shot examples
        """
        # If no dataset is provided, try to use identification_data or return original prompt
        dataset = self.dataset
        if dataset is None and identification_data and "dataset" in identification_data:
            dataset = identification_data["dataset"]
        
        if dataset is None:
            return [prompt]
        
        # Check if a specific mode is requested in identification_data
        requested_mode = identification_data.get("fewshot_mode", self.mode) if identification_data else self.mode
        
        if requested_mode == "which":
            return self._augment_which_examples(prompt, dataset)
        elif requested_mode == "how_many":
            return self._augment_how_many_examples(prompt, dataset)
        else:  # "both" or any other value
            return self._augment_both(prompt, dataset)

    def _augment_which_examples(self, prompt: str, dataset: pd.DataFrame) -> List[str]:
        """Vary which specific examples are used while keeping the count constant."""
        variations = []
        used_variations = set()
        attempts = 0
        
        while len(variations) < self.n_augments and attempts < self.n_augments * 2:
            # Get random examples for this variation with fixed count
            examples = self._get_examples_for_question(prompt, dataset, 
                                                      count=self.num_examples,
                                                      random_state=None)
            formatted = self.format_examples(examples)
            
            # Only add if it's new
            if formatted not in used_variations:
                variations.append(formatted)
                used_variations.add(formatted)
            attempts += 1
        
        return variations[:self.n_augments]

    def _augment_how_many_examples(self, prompt: str, dataset: pd.DataFrame) -> List[str]:
        """Vary the number of examples while trying to keep the specific examples consistent."""
        variations = []
        
        # First, get a pool of examples that we'll use
        max_examples = max(self.example_counts)
        example_pool = self._get_examples_for_question(prompt, dataset, 
                                                     count=max_examples,
                                                     random_state=42)  # Fixed seed for consistency
        
        # Now create variations with different counts
        for count in self.example_counts:
            if count <= len(example_pool):
                examples = example_pool[:count]  # Take the first 'count' examples
                formatted = self.format_examples(examples)
                variations.append(formatted)
                
                # If we have enough variations, stop
                if len(variations) >= self.n_augments:
                    break
        
        return variations[:self.n_augments]

    def _augment_both(self, prompt: str, dataset: pd.DataFrame) -> List[str]:
        """Vary both which examples are used and how many."""
        # Get variations for each mode
        which_variations = self._augment_which_examples(prompt, dataset)
        how_many_variations = self._augment_how_many_examples(prompt, dataset)
        
        # Combine and shuffle
        combined = which_variations + how_many_variations
        random.shuffle(combined)
        
        return combined[:self.n_augments]

    def _get_examples_for_question(self, question: str, df, count=None, random_state=None) -> List[str]:
        """
        Get few-shot examples for a specific question, skipping the question itself.
        
        Args:
            question: The question to get examples for
            df: DataFrame with examples
            count: Number of examples to get (defaults to self.num_examples)
            random_state: Random state for reproducibility
            
        Returns:
            List of formatted example strings
        """
        result = []
        temp_df = df.copy()

        # Filter out the current question
        temp_df = temp_df[temp_df["input"] != question]

        if len(temp_df) == 0:
            return []

        # Use specified count or default
        num_examples = count if count is not None else self.num_examples
        num_examples = min(num_examples, len(temp_df))
        
        temp_df = temp_df.sample(
            n=num_examples,
            random_state=random_state,
            replace=False
        )

        # Format the examples
        for i in range(num_examples):
            example_input = temp_df.iloc[i]["input"]
            example_output = temp_df.iloc[i]["output"]
            result.append(FewShotConstants.EXAMPLE_FORMAT.format(example_input, example_output))

        return result

    def augment_all_questions(self, df) -> Dict[str, List[str]]:
        """
        Process all questions in the dataframe and return few-shot examples for each.
        
        Args:
            df: DataFrame with 'input' and 'output' columns
            
        Returns:
            Dictionary where keys are input questions and values are lists of
            few-shot example strings
        """
        if "input" not in df.columns or "output" not in df.columns:
            raise ValueError("Dataframe must contain columns - 'input', 'output'")

        result = {}

        # Process each question in the dataframe
        for _, row in df.iterrows():
            question = row["input"]
            examples = self._get_examples_for_question(question, df)
            result[question] = examples

        return result

    def format_examples(self, examples: List[str]) -> str:
        """
        Format the few-shot examples into a string.
        
        Args:
            examples: list of formatted example strings
            
        Returns:
            formatted string of examples
        """
        return FewShotConstants.EXAMPLE_SEPARATOR.join(examples)

    def create_few_shot_prompt(self, test_question: str, example_pairs: List[tuple]) -> str:
        """
        Create a few-shot prompt with provided examples and a test question.
        
        Args:
            test_question: The question to answer (will be placed at the end)
            example_pairs: List of (question, answer) tuples to use as examples
            
        Returns:
            A formatted few-shot prompt string
        """
        examples = []
        
        # Add the provided examples
        for question, answer in example_pairs:
            examples.append(FewShotConstants.EXAMPLE_FORMAT.format(question, answer))
        
        # Add the test question
        examples.append(FewShotConstants.QUESTION_FORMAT.format(test_question))
        
        # Format and return the prompt
        return self.format_examples(examples)

    def augment_with_examples(self, test_question: str, example_pool: List[tuple]) -> List[str]:
        """
        Create multiple few-shot prompt variations by sampling different examples
        and varying their order.
        
        Args:
            test_question: The question to answer (will be placed at the end)
            example_pool: List of (question, answer) tuples to sample from
            
        Returns:
            List of formatted few-shot prompt variations
        """
        if len(example_pool) < self.num_examples:
            # Not enough examples to sample from
            return [self.create_few_shot_prompt(test_question, example_pool)]
        
        variations = []
        
        # Create n_augments variations
        for _ in range(self.n_augments):
            # Sample examples
            sampled_examples = random.sample(example_pool, min(self.num_examples, len(example_pool)))
            
            # Optionally shuffle the order (50% chance)
            if random.random() > 0.5:
                random.shuffle(sampled_examples)
            
            # Create the prompt
            prompt = self.create_few_shot_prompt(test_question, sampled_examples)
            variations.append(prompt)
        
        # Remove duplicates while preserving order
        unique_variations = []
        for var in variations:
            if var not in unique_variations:
                unique_variations.append(var)
        
        return unique_variations


if __name__ == "__main__":
    # Create sample data
    print("Creating sample data...")
    sample_data = pd.DataFrame({
        "input": [
            "What is the capital of France?",
            "What is the largest planet in our solar system?",
            "Who wrote Romeo and Juliet?",
            "What is the boiling point of water?",
            "What is the chemical symbol for gold?"
        ],
        "output": [
            "Paris",
            "Jupiter",
            "William Shakespeare",
            "100 degrees Celsius",
            "Au"
        ]
    })
    print(f"Created sample data with {len(sample_data)} examples")
    
    # Test question
    test_question = "What is the tallest mountain in the world?"
    
    # Test 1: "which" mode - vary which examples are used
    print("\n===== Testing 'which' mode =====")
    which_augmenter = FewShotAugmenter(n_augments=3, num_examples=2, mode="which")
    which_augmenter.set_dataset(sample_data)
    which_variations = which_augmenter.augment(test_question)
    
    print(f"Generated {len(which_variations)} variations:")
    for i, var in enumerate(which_variations):
        print(f"\nVariation {i+1}:")
        print(var)
    
    # Test 2: "how_many" mode - vary the number of examples
    print("\n===== Testing 'how_many' mode =====")
    how_many_augmenter = FewShotAugmenter(n_augments=3, num_examples=5, mode="how_many")
    how_many_augmenter.set_dataset(sample_data)
    how_many_variations = how_many_augmenter.augment(test_question)
    
    print(f"Generated {len(how_many_variations)} variations:")
    for i, var in enumerate(how_many_variations):
        print(f"\nVariation {i+1}:")
        print(var)
    
    # Test 3: "both" mode - vary both which examples and how many
    print("\n===== Testing 'both' mode =====")
    both_augmenter = FewShotAugmenter(n_augments=5, num_examples=3, mode="both")
    both_augmenter.set_dataset(sample_data)
    both_variations = both_augmenter.augment(test_question)
    
    print(f"Generated {len(both_variations)} variations:")
    for i, var in enumerate(both_variations):
        print(f"\nVariation {i+1}:")
        print(var)
    
    # Test 4: Using identification_data
    print("\n===== Testing with identification_data =====")
    id_augmenter = FewShotAugmenter(n_augments=2, num_examples=2, mode="which")
    
    # Create identification_data with dataset and mode override
    identification_data = {
        "dataset": pd.DataFrame({
            "input": [
                "What is the deepest ocean?",
                "Who discovered electricity?",
                "What is the smallest planet?",
                "What is the capital of Japan?"
            ],
            "output": [
                "Pacific Ocean (Mariana Trench)",
                "Benjamin Franklin",
                "Mercury",
                "Tokyo"
            ]
        }),
        "fewshot_mode": "how_many"  # Override the mode
    }
    
    id_variations = id_augmenter.augment(test_question, identification_data)
    
    print(f"Generated {len(id_variations)} variations using identification_data:")
    for i, var in enumerate(id_variations):
        print(f"\nVariation {i+1}:")
        print(var)
