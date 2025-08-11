from promptsuite.augmentations.base import BaseAxisAugmenter
from typing import List, Optional
import ast
from promptsuite.shared.model_client import get_completion
from promptsuite.core.template_keys import PARAPHRASE_WITH_LLM
import os

instruction_template = """Help me write creative variations of an instruction prompt to an LLM for the following task description. 

IMPORTANT: The instruction may contain placeholders in curly braces like {{subject}}, {{topic}}, {{field}}, etc. These placeholders MUST be preserved EXACTLY as they appear in ALL variations.

Provide {n_augments} creative versions while:
1. Preserving the original meaning and intent
2. Keeping ALL placeholders {{}} unchanged in their exact positions
3. Varying the instructional language around the placeholders
4. NEVER introduce new placeholders - if the original has no placeholders, the variations must have none

Output only a Python list of strings with the alternatives. Do not include any explanation or additional text.

Original instruction: '''{prompt}'''"""
class Paraphrase(BaseAxisAugmenter):
    def __init__(self, n_augments: int = 1, api_key: str = None, seed: Optional[int] = None, 
                 model_name: Optional[str] = None, api_platform: Optional[str] = None):
        """
        Initialize the paraphrse augmenter.

        Args:
            n_augments: number of paraphrase needed
            api_key: API key for the language model service
            seed: Random seed for reproducibility
            model_name: Name of the model to use
            api_platform: Platform to use ("TogetherAI" or "OpenAI")
        """
        super().__init__(n_augments=n_augments, seed=seed)
        self.api_key = api_key
        self.model_name = model_name
        self.api_platform = api_platform

    def build_rephrasing_prompt(self, template: str, n_augments: int, prompt: str) -> str:
        return template.format(n_augments=n_augments, prompt=prompt)

    def augment(self, prompt: str) -> List[str]:
        """
        Generate paraphrase variations of the prompt.
        
        Args:
            prompt: The text to paraphrase
            
        Returns:
            List of paraphrased variations
        """
        rephrasing_prompt = self.build_rephrasing_prompt(instruction_template, self.n_augments, prompt)
        response = get_completion(
            rephrasing_prompt, 
            api_key=self.api_key, 
            model_name=self.model_name, 
            platform=self.api_platform
        )
        return ast.literal_eval(response)

    def _generate_simple_paraphrases(self, prompt: str) -> List[str]:
        """
        Generate simple paraphrase variations without using external API.
        
        Args:
            prompt: The text to paraphrase
            
        Returns:
            List of simple paraphrased variations
        """
        variations = [prompt]  # Always include original
        
        # Simple paraphrasing rules
        paraphrase_rules = [
            # Question reformulations
            lambda t: t.replace("What is", "Can you tell me what") + "?" if "What is" in t and not t.endswith("?") else t,
            lambda t: t.replace("How do", "What is the way to") if "How do" in t else t,
            lambda t: t.replace("Why is", "What is the reason that") if "Why is" in t else t,
            
            # Statement reformulations
            lambda t: "Please " + t.lower() if not t.lower().startswith(("please", "can you", "could you")) else t,
            lambda t: "Could you " + t.lower() if not t.lower().startswith(("could you", "can you", "please")) else t,
            lambda t: t.replace("I need", "I require") if "I need" in t else t,
            lambda t: t.replace("I want", "I would like") if "I want" in t else t,
            lambda t: t.replace("Classify", "Determine") if "Classify" in t else t,
            lambda t: t.replace("Answer", "Respond to") if "Answer" in t else t,
            lambda t: t.replace("Explain", "Describe") if "Explain" in t else t,
            lambda t: t.replace("Choose", "Select") if "Choose" in t else t,
            lambda t: t.replace("Find", "Identify") if "Find" in t else t,
        ]
        
        for rule in paraphrase_rules:
            if len(variations) >= self.n_augments + 1:  # +1 for original
                break
            paraphrased = rule(prompt)
            if paraphrased != prompt and paraphrased not in variations:
                variations.append(paraphrased)
        
        # Fill with simple variations if needed
        while len(variations) < self.n_augments + 1:
            if not prompt.endswith("."):
                variations.append(prompt + ".")
            elif "the" in prompt.lower():
                variations.append(prompt.replace("the ", "this ").replace("The ", "This "))
            else:
                break
        
        return variations[:self.n_augments + 1]


if __name__ == '__main__':
    text = 'The following are multiple choice questions (with answers) about {subject}.'
    
    # Example usage with 10 variations
    # Configure paraphraser with default parameters from constants
    paraphraser = Paraphrase(
        n_augments=10,
        api_key=os.getenv("TOGETHERAI_API_KEY"),  # Will use environment variable or default
        seed=42,       # TASK_DEFAULT_RANDOM_SEED
        model_name="meta-llama/Llama-3.3-70B-Instruct-Turbo-Free",  # TASK_DEFAULT_MODEL_NAME
        api_platform="TogetherAI",   # TASK_DEFAULT_PLATFORM,
    )

    print("=== Paraphrase Example ===")
    print(f"Original text: '{text}'")
    print(f"Number of variations: {paraphraser.n_augments}")
    print(f"Model: {paraphraser.model_name}")
    print(f"Platform: {paraphraser.api_platform}")
    print("-" * 50)
    
    print("\n2. LLM-based paraphrases (requires API key):")
    try:
        llm_variations = paraphraser.augment(text)
        for i, variation in enumerate(llm_variations):
            print(f"   {i+1}. '{variation}'")
    except Exception as e:
        print(f"   ❌ API call failed: {e}")
        print("   💡 Make sure to set your API key in environment variables")
    
    print("\n✅ Example completed!")

