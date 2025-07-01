#!/usr/bin/env python3
"""
Shared metrics calculation functions for different tasks.
"""

import re
from typing import Dict, Any, List, Tuple
import evaluate
from sklearn.metrics import mean_squared_error

def calculate_text_generation_metrics(prediction: str, reference: str) -> Dict[str, float]:
    """
    Calculate BLEU, ROUGE, and SacreBLEU metrics for text generation tasks.
    
    Args:
        prediction: Model prediction text
        reference: Reference/gold text
        
    Returns:
        Dictionary with metric scores
    """
    try:
        # Load evaluation metrics
        bleu = evaluate.load("bleu")
        rouge = evaluate.load("rouge")
        sacrebleu = evaluate.load("sacrebleu")

        # Calculate BLEU
        bleu_score = bleu.compute(predictions=[prediction], references=[[reference]])

        # Calculate ROUGE
        rouge_score = rouge.compute(predictions=[prediction], references=[reference])

        # Calculate SacreBLEU
        sacrebleu_score = sacrebleu.compute(predictions=[prediction], references=[[reference]])

        return {
            "bleu": bleu_score.get("bleu", 0.0),
            "rouge1": rouge_score.get("rouge1", 0.0),
            "rouge2": rouge_score.get("rouge2", 0.0),
            "rougeL": rouge_score.get("rougeL", 0.0),
            "sacrebleu": sacrebleu_score.get("score", 0.0)
        }
    except Exception as e:
        print(f"Error calculating text generation metrics: {e}")
        return {"bleu": 0.0, "rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0, "sacrebleu": 0.0}


def calculate_summarization_metrics(variation: dict, model_response: str, gold_field: str = "highlights"):
    """
    Calculate BLEU, ROUGE, and SacreBLEU for summarization tasks.
    Assumes the gold summary is in variation['gold_updates'][gold_field].
    """
    gold_updates = variation.get('gold_updates', {})
    gold_summary = gold_updates.get(gold_field)
    if not gold_summary:
        return f"No gold summary in gold_updates['{gold_field}']", False, {}

    # Calculate metrics using shared function
    metrics = calculate_text_generation_metrics(model_response, gold_summary)
    return gold_summary, None, metrics


def calculate_translation_correctness_and_metrics(variation: Dict[str, Any], model_response: str, gold_field: str = None) -> tuple:
    """
    Calculate translation metrics and determine correctness for translation tasks.
    
    Args:
        variation: The variation dictionary containing gold_updates
        model_response: The model's response string
        gold_field: Specific field to look for in gold_updates (if None, auto-detect language codes)
        
    Returns:
        tuple: (gold_answer_text, is_correct, translation_metrics)
    """
    try:
        gold_updates = variation.get('gold_updates', {})

        # If gold_field is specified, use it directly
        if gold_field:
            translation_gold = gold_updates.get(gold_field)
            if translation_gold is None:
                return f"No translation gold found in gold_updates['{gold_field}']", False, {}
        else:
            # Auto-detect by looking for language codes in gold_updates
            translation_languages = ['en', 'cs', 'de', 'fr', 'hi', 'ru']
            translation_gold = None
            for lang in translation_languages:
                if lang in gold_updates:
                    translation_gold = gold_updates[lang]
                    break

            if translation_gold is None:
                return "No translation gold found", False, {}

        # Calculate translation metrics using shared function
        translation_metrics = calculate_text_generation_metrics(model_response, translation_gold)

        # For translation, we consider it "correct" based on BLEU score threshold
        # is_correct = translation_metrics.get("bleu", 0.0) > 0.1  # 10% BLEU threshold

        return translation_gold, None, translation_metrics

    except Exception as e:
        return f"Error calculating translation metrics: {str(e)}", False, {}


def calculate_mmlu_correctness_and_metrics(variation: Dict[str, Any], model_response: str, gold_field: str = "answer") -> tuple:
    """
    Calculate correctness for MMLU multiple choice tasks.
    
    Args:
        variation: The variation dictionary containing gold_updates and choices
        model_response: The model's response string
        gold_field: Field name in gold_updates containing the answer (default: "answer")
        
    Returns:
        tuple: (gold_answer_text, is_correct, empty_metrics_dict)
    """
    try:
        gold_updates = variation.get('gold_updates', {})

        # Handle multiple choice (MMLU style)
        gold_index = gold_updates.get(gold_field)
        if gold_index is None:
            return f"No gold answer in gold_updates['{gold_field}']", False, {}

        # Get the choices from field_values
        field_values = variation.get('configuration', {}).get('field_values', {})
        choices_str = field_values.get('choices', '')

        if not choices_str:
            return "No choices found", False, {}

        # Split choices by lines and clean them
        choices = [choice.strip() for choice in choices_str.split('\n') if choice.strip()]

        # Get the correct answer by index
        try:
            gold_index_int = int(gold_index)
            if 0 <= gold_index_int < len(choices):
                gold_answer_text = choices[gold_index_int]
            else:
                return f"Invalid index {gold_index}", False, {}
        except (ValueError, TypeError):
            return f"Invalid gold index: {gold_index}", False, {}

        # Check if model response is correct
        model_response_clean = model_response.strip().lower()
        gold_answer_clean = gold_answer_text.strip().lower()

        # Check for exact match or if model response contains the gold answer
        is_correct = (model_response_clean == gold_answer_clean or
                      gold_answer_clean in model_response_clean or
                      model_response_clean in gold_answer_clean)

        return gold_answer_text, is_correct, {}

    except Exception as e:
        return f"Error calculating MMLU correctness: {str(e)}", False, {}


def calculate_sentiment_correctness_and_metrics(variation: dict, model_response: str, gold_field: str = "label") -> tuple:
    """
    Calculate sentiment analysis correctness and metrics for continuous scoring.
    
    Args:
        variation: The variation dictionary containing gold_updates
        model_response: The model's response string
        gold_field: Field name in gold_updates containing the sentiment score (default: "label")
        
    Returns:
        tuple: (gold_answer_text, is_correct, sentiment_metrics)
    """
    try:
        gold_updates = variation.get('gold_updates', {})

        # Get the gold sentiment score
        gold_score = gold_updates.get(gold_field)
        if gold_score is None:
            return f"No gold sentiment score in gold_updates['{gold_field}']", False, {}

        gold_score = float(gold_score)

        # Extract predicted score from model response
        predicted_score = float(model_response.strip())

        # Try to extract a number between 0.0 and 1.0 from the response
        # Look for patterns like "0.7", "0.85", "1.0", etc.
        # score_patterns = [
        #     r'\b([01]?\.\d+)\b',  # Decimal numbers like 0.7, 1.0
        #     r'\b(0\.\d+)\b',      # Numbers starting with 0.
        #     r'\b(1\.0+)\b',       # 1.0, 1.00, etc.
        #     r'\b([01])\b'         # Just 0 or 1
        # ]
        #
        # predicted_score = None
        # for pattern in score_patterns:
        #     matches = re.findall(pattern, model_response_clean)
        #     if matches:
        #         try:
        #             score = float(matches[0])
        #             if 0.0 <= score <= 1.0:
        #                 predicted_score = score
        #                 break
        #         except ValueError:
        #             continue
        #
        # if predicted_score is None:
        #     # If no valid score found, try to infer from text
        #     response_lower = model_response_clean.lower()
        #     if any(word in response_lower for word in ['very positive', 'extremely positive', 'highly positive']):
        #         predicted_score = 0.9
        #     elif any(word in response_lower for word in ['positive', 'good', 'great']):
        #         predicted_score = 0.7
        #     elif any(word in response_lower for word in ['neutral', 'okay', 'average']):
        #         predicted_score = 0.5
        #     elif any(word in response_lower for word in ['negative', 'bad', 'poor']):
        #         predicted_score = 0.3
        #     elif any(word in response_lower for word in ['very negative', 'extremely negative', 'highly negative']):
        #         predicted_score = 0.1
        #     else:
        #         predicted_score = 0.5  # Default to neutral if unclear

        # Calculate metrics
        mse = mean_squared_error([gold_score], [predicted_score])
        mae = abs(gold_score - predicted_score)  # Mean Absolute Error

        # Consider "correct" if within 0.2 of the gold score (20% tolerance)
        sentiment_metrics = {
            'predicted_score': predicted_score,
            'mse': mse,
            'mae': mae,
            'absolute_error': mae
        }

        return f"{gold_score:.2f}", None, sentiment_metrics

    except Exception as e:
        return f"Error calculating sentiment metrics: {str(e)}", False, {}


def calculate_qa_correctness_and_metrics(variation: dict, model_response: str, gold_field: str = "answer") -> tuple:
    """
    Calculate correctness and metrics for Question Answering tasks.
    Includes exact match, word-level F1 score, precision, recall, and text generation metrics.
    
    Args:
        variation: The variation dictionary containing gold_updates
        model_response: The model's response string
        gold_field: Field name in gold_updates containing the answer (default: "answer")
        
    Returns:
        tuple: (gold_answer_text, is_correct, qa_metrics)
    """
    try:
        import re
        
        gold_updates = variation.get('gold_updates', {})

        # Get the gold answer
        gold_answer = gold_updates.get(gold_field)
        if gold_answer is None:
            return f"No gold answer in gold_updates['{gold_field}']", False, {}

        # Clean and normalize both answers for comparison
        def normalize_text(text: str) -> str:
            """Normalize text for comparison."""
            if not text:
                return ""
            # Convert to lowercase and strip whitespace
            text = text.strip().lower()
            # Remove extra whitespace
            text = re.sub(r'\s+', ' ', text)
            # Remove common punctuation at the end
            text = re.sub(r'[.!?]+$', '', text)
            return text

        def extract_words(text: str) -> set:
            """Extract words from text for F1 calculation."""
            # Simple word extraction - split on whitespace and punctuation
            words = re.findall(r'\b\w+\b', text.lower())
            return set(words)

        # Normalize answers
        gold_normalized = normalize_text(str(gold_answer))
        predicted_normalized = normalize_text(model_response)

        # Exact match accuracy (case-insensitive, normalized)
        exact_match = gold_normalized == predicted_normalized

        # Calculate word-level F1 score for overlap between predicted and gold answers
        gold_words = extract_words(str(gold_answer))
        predicted_words = extract_words(model_response)

        # Calculate precision, recall, and F1 at word level
        if not predicted_words:
            word_precision = 0.0
            word_recall = 0.0
            word_f1 = 0.0
        elif not gold_words:
            word_precision = 0.0
            word_recall = 0.0
            word_f1 = 0.0
        else:
            # Calculate overlap
            overlap = gold_words.intersection(predicted_words)
            word_precision = len(overlap) / len(predicted_words) if predicted_words else 0.0
            word_recall = len(overlap) / len(gold_words) if gold_words else 0.0
            word_f1 = (2 * word_precision * word_recall) / (word_precision + word_recall) if (word_precision + word_recall) > 0 else 0.0

        # Calculate text generation metrics for additional evaluation
        text_metrics = calculate_text_generation_metrics(model_response, str(gold_answer))

        # Combine all metrics
        qa_metrics = {
            'exact_match': exact_match,
            'word_precision': word_precision,
            'word_recall': word_recall,
            'word_f1': word_f1,
            **text_metrics  # Include BLEU, ROUGE, etc.
        }

        # For QA, we consider it "correct" based on exact match or high word F1 score
        is_correct = exact_match or word_f1 > 0.5  # 50% word F1 threshold

        return str(gold_answer), is_correct, qa_metrics

    except Exception as e:
        return f"Error calculating QA metrics: {str(e)}", False, {}


def calculate_math_correctness_and_metrics(variation: Dict[str, Any], model_response: str, gold_field: str = "answer") -> tuple:
    """
    Calculate correctness for math problem solving tasks.
    
    Args:
        variation: The variation dictionary containing gold_updates
        model_response: The model's response string
        gold_field: Field name in gold_updates containing the answer (default: "answer")
        
    Returns:
        tuple: (gold_answer_text, is_correct, metrics_dict_with_parsed_answers)
    """
    try:
        gold_updates = variation.get('gold_updates', {})

        # Get the gold answer (full GSM8K format answer)
        gold_answer = gold_updates.get(gold_field)
        if gold_answer is None:
            return f"No gold answer in gold_updates['{gold_field}']", False, {}

        # Extract numeric value from model response
        def extract_numeric_from_response(response: str) -> float:
            """Extract numeric answer from model response."""
            import re

            # First try to find #### pattern (if model follows GSM8K format)
            gsm8k_pattern = r"#### (\-?[0-9\.\,]+)"
            gsm8k_matches = re.findall(gsm8k_pattern, response)
            if gsm8k_matches:
                try:
                    return float(gsm8k_matches[0].replace(',', ''))
                except ValueError:
                    pass

            # Fallback: look for any numbers in the response, prioritizing the last one
            numbers = re.findall(r'-?\d+(?:\.\d+)?', response.strip())
            if numbers:
                try:
                    return float(numbers[-1].replace(',', ''))
                except ValueError:
                    pass

            return None

        # Extract numeric value from gold answer (GSM8K format)
        def extract_numeric_from_gold_answer(answer_text: str) -> float:
            """Extract the final numeric answer from GSM8K answer format."""
            import re

            # GSM8K answers end with "#### [number]" format
            regex_pattern = r"#### (\-?[0-9\.\,]+)"
            matches = re.findall(regex_pattern, answer_text)

            if matches:
                # Take the first (and should be only) match
                numeric_part = matches[0]
                # Remove commas and return the clean number
                try:
                    return float(numeric_part.replace(',', ''))
                except ValueError:
                    pass

            # If no #### pattern found, this is an error in the dataset
            raise ValueError(f"No numeric answer found in expected format '#### [number]' in: {answer_text}")

        # Extract numeric answers from both gold and model response
        try:
            gold_numeric = extract_numeric_from_gold_answer(str(gold_answer))
        except (ValueError, TypeError) as e:
            return f"Invalid gold answer format: {gold_answer} - {str(e)}", False, {}

        # Extract predicted numeric answer
        predicted_numeric = extract_numeric_from_response(model_response)

        if predicted_numeric is None:
            return str(gold_answer), False, {
                'gold_numeric_answer': gold_numeric,
                'parsed_answer': None
            }

        # Check if answers match (allowing for small floating point differences)
        is_correct = abs(gold_numeric - predicted_numeric) < 1e-6

        return str(gold_answer), is_correct, {
            'gold_numeric_answer': gold_numeric,
            'parsed_answer': predicted_numeric
        }

    except Exception as e:
        return f"Error calculating math correctness: {str(e)}", False, {}


def extract_final_answer_from_response(response: str) -> str:
    import re
    response = response.strip()
    lines = [line.strip() for line in response.split('\n') if line.strip()]

    # 1. חפש שורה שמתחילה ב-Final answer: או Answer: או Choice:
    for line in reversed(lines):
        m = re.match(r'^(Final answer|Answer|Choice)[:.\s-]+(.+)$', line, flags=re.IGNORECASE)
        if m:
            answer = m.group(2).strip()
            return answer

    # 1b. חפש שורה שמכילה רק Answer: או Choice: ואז בשורה הבאה את התשובה
    for i, line in enumerate(lines):
        if re.match(r'^(Final answer|Answer|Choice)[:.\s-]*$', line, flags=re.IGNORECASE):
            if i + 1 < len(lines):
                return lines[i + 1].strip()

    # 2. חפש תשובה מודגשת (bold) או underline או inline code (כמו קודם)
    all_bold = re.findall(r'\*\*([^*]+)\*\*', response)
    if not all_bold:
        all_bold = re.findall(r'__([^_]+)__', response)
    if not all_bold:
        all_bold = re.findall(r'`([^`]+)`', response)
    if all_bold:
        for candidate in reversed(all_bold):
            candidate = candidate.strip()
            # אם candidate מתחיל ב-Answer/Choice וכו', נסה להוציא את החלק שאחריהם
            m = re.match(r'^(Answer|Choice)[:.\s-]*([a-dA-D1-4IVXivx]+[.:]?\s*.+)', candidate, flags=re.IGNORECASE)
            if m:
                return m.group(2).strip()
            m2 = re.match(r'^(Answer|Choice)[:.\s-]*$', candidate, flags=re.IGNORECASE)
            if m2:
                continue  # זה רק prefix, לא תשובה
            if len(candidate) > 100 or any(word in candidate.lower() for word in ['dimethyl', 'sulfane', 'methane', 'reagent', 'chlorochromate']):
                continue
            return candidate

    # 3. חפש תבניות של אות/מספר/רומאי בתחילת שורה (ללא נרמול)
    for line in reversed(lines):
        m = re.match(r'^([a-dA-D1-4IVXivx]+)[.:]\s*(.+)', line)
        if m:
            return f'{m.group(1)}. {m.group(2).strip()}'
        # גם מקרים של רק אות/מספר/רומאי
        m2 = re.match(r'^([a-dA-D1-4IVXivx]+)[.]?\s*$', line)
        if m2 and len(line) <= 5:
            return f'{m2.group(1)}.'

    # 4. fallback: תחזיר None
    return None


def calculate_gpqa_correctness_and_metrics(variation: Dict[str, Any], model_response: str, gold_field: str = "answer") -> tuple:
    """
    Calculate correctness for GPQA tasks.

    Args:
        variation: The variation dictionary containing gold_updates
        model_response: The model's response string
        gold_field: Field name in gold_updates containing the correct answer index (default: "answer")

    Returns:
        tuple: (gold_answer_text, is_correct, metrics_dict_with_parsed_answer)
    """
    try:
        gold_updates = variation.get('gold_updates', {})
        gold_index = gold_updates.get(gold_field)
        if gold_index is None:
            return f"No gold answer in gold_updates['{gold_field}']", False, {}
        field_values = variation.get('configuration', {}).get('field_values', {})
        choices_str = field_values.get('choices', '')
        if not choices_str:
            return "No choices found", False, {}
        choices = [choice.strip() for choice in choices_str.split('\n') if choice.strip()]
        try:
            gold_index_int = int(gold_index)
            if 0 <= gold_index_int < len(choices):
                gold_answer_text = choices[gold_index_int]
            else:
                return f"Invalid index {gold_index}", False, {}
        except (ValueError, TypeError):
            return f"Invalid gold index: {gold_index}", False, {}
        # Use the shared extraction function
        predicted_answer = extract_final_answer_from_response(model_response)
        # Clean gold answer for comparison
        gold_answer_clean = gold_answer_text.strip().lower()
        # Normalize answers for comparison (remove dots, spaces, etc.)
        def roman_to_int(roman: str) -> int:
            roman = roman.upper()
            roman_numerals = {'I': 1, 'V': 5, 'X': 10}
            result = 0
            prev_value = 0
            for char in reversed(roman):
                value = roman_numerals.get(char, 0)
                if value < prev_value:
                    result -= value
                else:
                    result += value
                prev_value = value
            return result

        def normalize_answer(answer: str) -> str:
            import re
            if not answer:
                return ""
            # Remove dots, spaces, and lowercase
            answer = answer.strip()
            normalized = re.sub(r'[.\s]+', '', answer.lower())
            # Convert roman numerals at start to number
            match = re.match(r'^([ivx]+)(\d*)$', normalized)
            if match:
                roman = match.group(1)
                rest = match.group(2)
                try:
                    number = str(roman_to_int(roman))
                    return number + rest
                except Exception:
                    pass
            # Convert letter to number (a=1, b=2, ...)
            match = re.match(r'^([a-d])([0-9]*)$', normalized)
            if match:
                letter = match.group(1)
                rest = match.group(2)
                letter_map = {'a': '1', 'b': '2', 'c': '3', 'd': '4'}
                if letter in letter_map:
                    return letter_map[letter] + rest
            # If already number at start
            match = re.match(r'^(\d+)(\d*)$', normalized)
            if match:
                return match.group(1) + match.group(2)
            return normalized

        gold_normalized = normalize_answer(gold_answer_clean)

        is_correct = False
        if predicted_answer:
            # Normalize both answers (handles roman, letter, number)
            predicted_normalized = normalize_answer(predicted_answer)
            # Check for exact match after normalization
            is_correct = predicted_normalized == gold_normalized

        # If no structured answer found, fall back to checking if gold answer appears anywhere
        if not is_correct and not predicted_answer:
            model_response_clean = model_response.strip().lower()
            is_correct = (gold_answer_clean in model_response_clean or
                         model_response_clean in gold_answer_clean)

        return gold_answer_text, is_correct, {
            'parsed_answer': predicted_answer
        }

    except Exception as e:
        return f"Error calculating GPQA correctness: {str(e)}", False, {}


def calculate_code_generation_correctness_and_metrics(variation: Dict[str, Any], model_response: str, gold_field: str = "canonical_solution") -> tuple:
    """
    Simple code generation metrics - just saves the code without evaluation.
    Evaluation will be done separately after all runs are complete.
    
    Args:
        variation: The variation dictionary containing gold_updates and task information
        model_response: The model's response string (generated code)
        gold_field: Field name in gold_updates containing the canonical solution (default: "canonical_solution")
        
    Returns:
        tuple: (gold_answer_text, is_correct, code_metrics)
    """
    try:
        gold_updates = variation.get('gold_updates', {})

        # Get the gold canonical solution
        canonical_solution = gold_updates.get(gold_field)
        if canonical_solution is None:
            return f"No canonical solution in gold_updates['{gold_field}']", False, {}

        # Get task_id from the variation data
        original_row_index = variation.get('original_row_index', 0)
        task_id = f"HumanEval/{original_row_index}"  # Simple task_id format
        
        # Just store basic information without evaluation
        code_metrics = {
            'task_id': task_id,
            'code_length': len(model_response),
            'canonical_length': len(canonical_solution)
        }

        # Return None for is_correct since we'll evaluate later
        return canonical_solution, None, code_metrics

    except Exception as e:
        return f"Error storing code generation data: {str(e)}", False, {}


def calculate_musique_correctness_and_metrics(variation: Dict[str, Any], model_response: str, gold_field: str = "answer") -> tuple:
    """
    Calculate correctness and metrics for MuSiQue multi-hop question answering tasks.

    Args:
        variation: The variation dictionary containing gold_updates
        model_response: The model's response string
        gold_field: Field name in gold_updates containing the answer (default: "answer")

    Returns:
        tuple: (gold_answer_text, is_correct, musique_metrics)
    """
    try:
        from sklearn.metrics import f1_score
        import re

        gold_updates = variation.get('gold_updates', {})

        # Get the gold answer
        gold_answer = gold_updates.get(gold_field)
        if gold_answer is None:
            return f"No gold answer in gold_updates['{gold_field}']", False, {}

        # Clean and normalize both answers for comparison
        def normalize_text(text: str) -> str:
            """Normalize text for comparison."""
            if not text:
                return ""
            # Convert to lowercase and strip whitespace
            text = text.strip().lower()
            # Remove extra whitespace
            text = re.sub(r'\s+', ' ', text)
            # Remove common punctuation at the end
            text = re.sub(r'[.!?]+$', '', text)
            return text

        def extract_words(text: str) -> set:
            """Extract words from text for F1 calculation."""
            # Simple word extraction - split on whitespace and punctuation
            words = re.findall(r'\b\w+\b', text.lower())
            return set(words)

        # Normalize answers
        gold_normalized = normalize_text(str(gold_answer))
        predicted_normalized = normalize_text(model_response)

        # Exact match accuracy (case-insensitive)
        is_correct = gold_normalized == predicted_normalized

        # Calculate word-level F1 score for overlap between predicted and gold answers
        gold_words = extract_words(str(gold_answer))
        predicted_words = extract_words(model_response)

        # Calculate precision, recall, and F1 at word level
        if not predicted_words:
            precision = 0.0
            recall = 0.0
            f1 = 0.0
        elif not gold_words:
            precision = 0.0
            recall = 0.0
            f1 = 0.0
        else:
            # Calculate overlap
            overlap = gold_words.intersection(predicted_words)
            precision = len(overlap) / len(predicted_words) if predicted_words else 0.0
            recall = len(overlap) / len(gold_words) if gold_words else 0.0
            f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        # Calculate text generation metrics for additional evaluation
        text_metrics = calculate_text_generation_metrics(model_response, str(gold_answer))

        # Combine all metrics
        musique_metrics = {
            'word_precision': precision,
            'word_recall': recall,
            'word_f1': f1,
            'exact_match': is_correct,
            **text_metrics  # Include BLEU, ROUGE, etc.
        }

        return str(gold_answer), is_correct, musique_metrics

    except Exception as e:
        return f"Error calculating MuSiQue metrics: {str(e)}", False, {}


def calculate_bertscore_metrics(predictions: list, references: list, lang: str = "en") -> list:
    """
    Calculate BERTScore for a list of predictions and references.
    
    Args:
        predictions: List of model predictions
        references: List of reference texts
        lang: Language code for BERTScore (default: "en")
        
    Returns:
        List of BERTScore F1 scores
    """
    bertscore = evaluate.load("bertscore")
    results = bertscore.compute(predictions=predictions, references=references, lang=lang)
    return results.get('f1', [])