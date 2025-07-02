import re
import sys
from pathlib import Path

# Add src to Python path for imports
script_dir = Path(__file__).parent
project_root = script_dir.parent
src_path = project_root / "src"
sys.path.insert(0, str(src_path))

from promptsuite.core.template_keys import (
    PROMPT_FORMAT, PROMPT_FORMAT_VARIATIONS, QUESTION_KEY, GOLD_KEY, FEW_SHOT_KEY, OPTIONS_KEY, CONTEXT_KEY,
    PROBLEM_KEY,
    GOLD_FIELD, INSTRUCTION_VARIATIONS, INSTRUCTION,
    PARAPHRASE_WITH_LLM, CONTEXT_VARIATION, SHUFFLE_VARIATION, MULTIDOC_VARIATION, ENUMERATE_VARIATION
)

REPLACEMENTS = {
    'instruction': INSTRUCTION,
    'instruction variations': INSTRUCTION_VARIATIONS,
    'prompt format': PROMPT_FORMAT,
    'prompt format variations': PROMPT_FORMAT_VARIATIONS,
    'question': QUESTION_KEY,
    'gold': GOLD_KEY,
    'few_shot': FEW_SHOT_KEY,
    'options': OPTIONS_KEY,
    'context': CONTEXT_KEY,
    'problem': PROBLEM_KEY,
    'gold_field': GOLD_FIELD,
    'paraphrase': PARAPHRASE_WITH_LLM,
    'context_variation': CONTEXT_VARIATION,
    'shuffle': SHUFFLE_VARIATION,
    'multidoc': MULTIDOC_VARIATION,
    'enumerate': ENUMERATE_VARIATION,
}

DOCS_ROOTS = [
    project_root / 'README.md',
    project_root / 'docs',
]


def update_docs():
    for root in DOCS_ROOTS:
        if root.is_file():
            files = [root]
        elif root.is_dir():
            files = list(root.rglob('*.md'))
        else:
            print(f"Skipping {root} (not found)")
            continue
        for file in files:
            text = file.read_text(encoding='utf-8')
            for old, new in REPLACEMENTS.items():
                text = re.sub(rf"(['\"]){old}(['\"])", rf"\1{new}\2", text)
                text = re.sub(rf":{old}\b", f":{new}", text)
            file.write_text(text, encoding='utf-8')
            print(f"Updated {file}")


if __name__ == '__main__':
    update_docs()
