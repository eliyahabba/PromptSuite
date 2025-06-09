from src.axis_augmentation.base_augmenter import BaseAxisAugmenter
from typing import List
from src.utils.model_client import get_completion
import ast


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


class Paraphrase(BaseAxisAugmenter):
    def __init__(self, n_augments: int = 1):
        """
        Initialize the paraphrse augmenter.

        Args:
            k: number of paraphrase needed
        """
        super().__init__(n_augments=n_augments)

    def build_rephrasing_prompt(self, template: str, n_augments: int, prompt: str) -> \
            str:
        return template.format(n_augments=n_augments, prompt=prompt)

    def augment(self, prompt:str) -> List[str]:
        prompt = self.build_rephrasing_prompt(talkative_template, self.n_augments, prompt)
        response = get_completion(prompt)
        return ast.literal_eval(response)


if __name__ == '__main__':
    para = Paraphrase(10)
    print(para.augment("Describe a historical figure you admire"))

