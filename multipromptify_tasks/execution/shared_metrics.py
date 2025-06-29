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
    Calculate correctness for Question Answering tasks.
    
    Args:
        variation: The variation dictionary containing gold_updates
        model_response: The model's response string
        gold_field: Field name in gold_updates containing the answer (default: "answer")
        
    Returns:
        tuple: (gold_answer_text, is_correct, qa_metrics)
    """
    try:
        gold_updates = variation.get('gold_updates', {})
        
        # Get the gold answer
        gold_answer = gold_updates.get(gold_field)
        if gold_answer is None:
            return f"No gold answer in gold_updates['{gold_field}']", False, {}
        
        # Calculate text generation metrics
        qa_metrics = calculate_text_generation_metrics(model_response, gold_answer)
        
        # For QA, we can consider it "correct" based on ROUGE-L or BLEU score
        rouge_l_threshold = 0.3  # 30% ROUGE-L threshold
        is_correct = qa_metrics.get("rougeL", 0.0) > rouge_l_threshold
        
        return gold_answer, is_correct, qa_metrics
        
    except Exception as e:
        return f"Error calculating QA metrics: {str(e)}", False, {}


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