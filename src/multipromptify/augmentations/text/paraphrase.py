from multipromptify.augmentations.base import BaseAxisAugmenter
from typing import List, Optional
import ast
from multipromptify.shared.model_client import get_completion
from multipromptify.core.template_keys import PARAPHRASE_WITH_LLM


#moran's gpt3.5 templates, changes to general LLM and {k} times and the
# return style
llm_template = (
    "Rephrase the follbuild_rephrasing_promptowing prompt, providing {k} alternative versions that are better suited for an LLM while preserving the original meaning. Output only a Python list of strings with the alternatives. Do not include any explanation or additional text. \n"
    "Prompt: '''{prompt}'''"
)

#moran's begining but adding specifications, restriction on the output and
# the word "creative"
talkative_template = (
    "Can you help me write a prompt to an LLM for the following task "
    "description? Providing {n_augments} creative versions while preserving the "
    "original meaning. \nOutput only a Python list of strings with the "
    "alternatives. Do not include any explanation or additional text. \n"
    "Prompt: '''{prompt}'''"
)

instruction_template = """You are a prompt rewriting assistant. Your task is to create {n_augments} creative versions of the given prompt template.

CRITICAL REQUIREMENTS:
1. PRESERVE ALL PLACEHOLDERS: Any text within curly braces {{}} must appear EXACTLY as given in all variations
2. Keep the same semantic meaning and task structure
3. Vary the instructional language while maintaining clarity
4. Ensure each variation would produce the same type of response

PLACEHOLDERS TO PRESERVE (copy exactly):
- Do NOT modify anything inside {{}} brackets
- Common placeholders include: {{text}}, {{category}}, {{question}}, {{options}}, {{answer}}, etc.

WHAT YOU CAN CHANGE:
- The instructional phrases (e.g., "Answer the following" → "Please respond to")
- Word order and sentence structure (while keeping meaning)
- Formatting style (but keep placeholders in logical positions)
- Level of formality or politeness

OUTPUT FORMAT:
Return ONLY a Python list of strings. Each string should be a complete prompt variation.

Original prompt: '''{prompt}'''

Generate {n_augments} creative variations:"""


instruction_template = """Can you help me write variations of an instruction prompt to an LLM for the following task description? 

IMPORTANT: The instruction may contain placeholders in curly braces like {{subject}}, {{topic}}, {{field}}, etc. These placeholders MUST be preserved EXACTLY as they appear in ALL variations.

Provide {n_augments} creative versions while:
1. Preserving the original meaning and intent
2. Keeping ALL placeholders {{}} unchanged in their exact positions
3. Varying the instructional language around the placeholders

Output only a Python list of strings with the alternatives. Do not include any explanation or additional text.

Original instruction: '''{prompt}'''"""
class Paraphrase(BaseAxisAugmenter):
    def __init__(self, n_augments: int = 1, api_key: str = None, seed: Optional[int] = None):
        """
        Initialize the paraphrse augmenter.

        Args:
            n_augments: number of paraphrase needed
            api_key: API key for the language model service
            seed: Random seed for reproducibility
        """
        super().__init__(n_augments=n_augments, seed=seed)
        self.api_key = api_key

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
        response = get_completion(rephrasing_prompt)
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
    para = Paraphrase(3)
    print(para.augment("Describe a historical figure you admire"))

