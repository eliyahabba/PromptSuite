import evaluate
from typing import Dict, Any


def calculate_text_generation_metrics(prediction: str, reference: str) -> Dict[str, float]:
    """
    Calculate BLEU, ROUGE, and SacreBlEU metrics for text generation tasks.
    Used by both summarization and translation tasks.
    
    Args:
        prediction: The model's generated text
        reference: The gold/reference text
        
    Returns:
        Dict containing metric scores
    """
    # Clean inputs
    prediction = prediction.strip()
    reference = reference.strip()
    
    # Load metrics
    bleu = evaluate.load("bleu")
    rouge = evaluate.load("rouge")
    sacrebleu = evaluate.load("sacrebleu")
    
    # Calculate metrics
    bleu_score = bleu.compute(predictions=[prediction], references=[[reference]])
    rouge_scores = rouge.compute(predictions=[prediction], references=[reference])
    sacrebleu_score = sacrebleu.compute(predictions=[prediction], references=[[reference]])
    
    return {
        "bleu": bleu_score.get("bleu", 0.0),
        "rouge1": rouge_scores.get("rouge1", 0.0),
        "rouge2": rouge_scores.get("rouge2", 0.0),
        "rougeL": rouge_scores.get("rougeL", 0.0),
        "sacrebleu": sacrebleu_score.get("score", 0.0)
    }


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


def calculate_translation_correctness_and_metrics(variation: Dict[str, Any], model_response: str) -> tuple:
    """
    Calculate translation metrics and determine correctness for translation tasks.
    
    Args:
        variation: The variation dictionary containing gold_updates
        model_response: The model's response string
        
    Returns:
        tuple: (gold_answer_text, is_correct, translation_metrics)
    """
    try:
        gold_updates = variation.get('gold_updates', {})

        # Check if this is a translation task by looking for language codes in gold_updates
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


def calculate_mmlu_correctness_and_metrics(variation: Dict[str, Any], model_response: str) -> tuple:
    """
    Calculate correctness for MMLU multiple choice tasks.
    
    Args:
        variation: The variation dictionary containing gold_updates and choices
        model_response: The model's response string
        
    Returns:
        tuple: (gold_answer_text, is_correct, empty_metrics_dict)
    """
    try:
        gold_updates = variation.get('gold_updates', {})
        
        # Handle multiple choice (MMLU style)
        gold_index = gold_updates.get('answer')
        if gold_index is None:
            return "No gold answer", False, {}

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