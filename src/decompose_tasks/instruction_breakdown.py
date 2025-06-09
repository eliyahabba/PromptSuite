import os
# Set thread limits before any other imports
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

# Now import the rest of the libraries
import pandas as pd
import time
import json
from typing import List, Dict, Any, Tuple, Set
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
import argparse
import traceback # Added for better error reporting

# Load environment variables
load_dotenv()

# --- Generic Annotation Loading ---

def load_annotation_examples(file_path: str) -> List[Dict[str, Any]]:
    """
    Load examples from a JSON file with either of these structures:
    1. [ { "prompt": "input text", "dimensions": { "key1": {...}, "key2": {...} } }, ... ]
    2. [ { "full_prompt": "...", "placeholder_prompt": "...", "annotations": { "key1": {...}, ... } }, ... ]

    Args:
        file_path: Path to the JSON file containing prompts and dimensions

    Returns:
        List of example dictionaries in a standardized format.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        if not isinstance(data, list):
            raise ValueError(f"Annotation file '{file_path}' should contain a JSON list.")

        examples = []

        for i, item in enumerate(data):
            if not isinstance(item, dict):
                print(f"Warning: Skipping item {i} in annotation file, not a dictionary.")
                continue
                
            # Check which format we're dealing with
            if "prompt" in item and "dimensions" in item:
                # Format 1: Original email_annotation.json format
                if not isinstance(item["dimensions"], dict):
                    print(f"Warning: Skipping item {i}, 'dimensions' is not a dictionary.")
                    continue
                examples.append(item)
                
            elif "full_prompt" in item and "annotations" in item:
                # Format 2: final_annotations.json format
                # Convert to the standard format expected by the rest of the code
                converted_item = {
                    "prompt": item["full_prompt"],
                    "dimensions": {}
                }
                
                # Convert annotations to dimensions format
                for key, annotation in item["annotations"].items():
                    if "text" in annotation:
                        # Create a dimension entry with the annotation text as a highlight
                        converted_item["dimensions"][key] = {
                            "name": key.replace("_", " ").title(),  # Convert snake_case to Title Case
                            "highlights": [
                                {
                                    "text": annotation["text"]
                                    # No start/end positions
                                }
                            ]
                        }
                
                examples.append(converted_item)
            else:
                print(f"Warning: Skipping item {i}, missing required keys. Found: {list(item.keys())}")
                continue

        print(f"Successfully loaded {len(examples)} examples from {file_path}")
        if not examples:
             print(f"Warning: No valid examples loaded from {file_path}. Check file structure.")
        return examples

    except FileNotFoundError:
        print(f"Error: Annotation file not found at {file_path}")
        return []
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON from {file_path}: {e}")
        return []
    except Exception as e:
        print(f"Error loading examples from JSON ({file_path}): {e}")
        traceback.print_exc()
        return []

# --- Generic Prompt Creation ---

def create_decomposition_prompt(examples: List[Dict[str, Any]]) -> str:
    """
    Create a few-shot prompt for task decomposition based on annotation examples.

    Args:
        examples: List of dictionaries containing 'prompt' and 'dimensions' keys

    Returns:
        A formatted few-shot prompt string
    """
    prompt = "Analyze the following input text and break it down into its key components based on the provided examples.\n\n"

    # Add few-shot examples
    for i, example in enumerate(examples):
        prompt += f"Example {i+1}:\nInput:\n{example['prompt']}\n\nBreakdown:\n"

        # Add each dimension with its highlights
        for dim_key, dim_value in example["dimensions"].items():
            # Use the key as the name, or a 'name' field if present
            dimension_name = dim_value.get("name", dim_key)
            prompt += f"{dimension_name}:\n"

            # Add highlighted sections if available
            highlights = dim_value.get("highlights", [])
            if highlights and isinstance(highlights, list):
                 for highlight in highlights:
                    # Check if highlight is a dict and has 'text'
                    if isinstance(highlight, dict) and "text" in highlight:
                        highlight_text = highlight.get("text", "")
                        if highlight_text:
                            prompt += f"- {str(highlight_text).strip()}\n" # Ensure text is string
                    # Handle cases where highlight might just be a string (less ideal)
                    elif isinstance(highlight, str):
                         prompt += f"- {highlight.strip()}\n"

            prompt += "\n" # Add space after each dimension's highlights

        prompt += "---\n\n" # Separator between examples

    # Add the template for the new input
    prompt += "Now, break down this input:\nInput:\n{input_text}\n\nBreakdown:"

    return prompt

# --- LLM Interaction ---

def get_completion(prompt_template: str, input_text: str, model_id: str = "meta-llama/llama-3-3-70b-instruct", 
                  provider: str = "together") -> str:
    """
    Get completion using either Together API or RITS endpoints.

    Args:
        prompt_template: The few-shot prompt template (expects '{input_text}')
        input_text: The specific text to process
        model_id: The model identifier (default is meta-llama/llama-3-3-70b-instruct)
        provider: The API provider to use ('together' or 'rits')

    Returns:
        The generated breakdown string.
    """
    try:
        # Ensure input_text is a string
        if not isinstance(input_text, str):
            input_text = str(input_text)

        formatted_prompt = prompt_template.format(input_text=input_text)
        
        # System prompt + user content
        system_content = "You are an AI assistant skilled at analyzing text and breaking it down into predefined components based on examples. Follow the format of the examples precisely."

        if provider.lower() == "together":
            # Use Together API
            from src.utils.model_client import get_model_response
            
            messages = [
                {"role": "system", "content": system_content},
                {"role": "user", "content": formatted_prompt}
            ]
            
            return get_model_response(messages, model_id)
            
        elif provider.lower() == "rits":
            # Configure for RITS (IBM)
            rits_host = os.getenv("RITS_HOST")
            rits_api_key = os.getenv("RITS_API_KEY")

            if not rits_host or not rits_api_key:
                raise ValueError("RITS_HOST and RITS_API_KEY environment variables must be set")

            # Extract just the model name (e.g., "llama-3-3-70b-instruct")
            model_name = model_id.split('/')[-1]
            # Use the correct URL format directly with the model name
            rits_base_url = f'{rits_host}/{model_name}/v1'
            
            print(f"Using RITS URL: {rits_base_url}")
            print(f"Using RITS Model ID: {model_id}")

            llm = ChatOpenAI(
                model=model_id,
                api_key='/',  # RITS uses header auth
                base_url=rits_base_url,
                default_headers={'RITS_API_KEY': rits_api_key},
                max_retries=2,
                temperature=0.7,
                max_tokens=1500
            )

            # Use LangChain's invoke method with SystemMessage and HumanMessage
            from langchain.schema import SystemMessage, HumanMessage
            
            messages = [
                SystemMessage(content=system_content),
                HumanMessage(content=formatted_prompt)
            ]
            
            response = llm.invoke(messages)
            return response.content.strip()
        else:
            raise ValueError(f"Unknown provider: {provider}. Must be 'together' or 'rits'.")

    except Exception as e:
        print(f"Error getting completion for input starting with '{input_text[:50]}...': {e}")
        traceback.print_exc() # Print full traceback for debugging
        # Return a specific error string to indicate failure
        return f"ERROR_GENERATING_BREAKDOWN: {str(e)}"

# --- Text Processing and Structure Extraction ---

def parse_llm_breakdown(text_breakdown: str) -> Tuple[str, Dict[str, List[str]]]:
    """
    Parses the raw text breakdown from the LLM into a structured dictionary.
    Assumes format like:
    DimensionName1:
    - Highlight 1
    - Highlight 2

    DimensionName2:
    - Highlight 3

    Args:
        text_breakdown: The raw string output from the LLM.

    Returns:
        A tuple containing:
        - The original text_breakdown string.
        - A dictionary mapping dimension names to lists of highlight strings.
    """
    structured_dimensions: Dict[str, List[str]] = {}
    current_dimension = None

    # Handle potential error marker from get_completion
    if text_breakdown.startswith("ERROR_GENERATING_BREAKDOWN"):
        print(f"Warning: Skipping parsing due to generation error: {text_breakdown}")
        return text_breakdown, {} # Return empty dict

    lines = text_breakdown.split('\n')
    for line in lines:
        stripped_line = line.strip()
        if not stripped_line: # Skip empty lines
            continue

        # Check if it's a dimension header (ends with ':' and doesn't start with '-')
        if stripped_line.endswith(':') and not stripped_line.startswith('-'):
            current_dimension = stripped_line[:-1].strip() # Remove colon and strip whitespace
            if current_dimension: # Ensure dimension name is not empty
                 structured_dimensions[current_dimension] = []
            else:
                 current_dimension = None # Invalid header
        # Check if it's a highlight bullet point under a valid dimension
        elif stripped_line.startswith('-') and current_dimension:
            highlight_text = stripped_line[1:].strip() # Remove '-' and strip whitespace
            if highlight_text: # Only add non-empty highlights
                structured_dimensions[current_dimension].append(highlight_text)
        # Reset current_dimension if the line doesn't fit the pattern
        # else:
        #     current_dimension = None # Or potentially log unexpected lines

    return text_breakdown, structured_dimensions

# --- DataFrame Processing ---

def process_dataframe_with_structure(
    df: pd.DataFrame,
    annotation_examples: List[Dict[str, Any]],
    input_column: str,
    model_id: str = "meta-llama/llama-3-3-70b-instruct",
    delay_seconds: float = 0.5, # Added delay parameter
    provider: str = "together" # Added provider parameter
) -> pd.DataFrame:
    """
    Processes a dataframe by applying LLM-based decomposition to an input column.
    Adds columns for the raw breakdown, structured JSON, and each identified dimension.

    Args:
        df: Input dataframe.
        annotation_examples: List of annotation examples from JSON.
        input_column: Column containing text to analyze.
        model_id: The model identifier.
        delay_seconds: Delay between API calls to avoid rate limiting.
        provider: The API provider to use ('together' or 'rits')

    Returns:
        Dataframe with added breakdown, dimensions_json, and individual dimension columns.
    """
    if input_column not in df.columns:
        raise ValueError(f"Input column '{input_column}' not found in DataFrame. Available columns: {list(df.columns)}")
    if not annotation_examples:
        raise ValueError("Annotation examples list is empty. Cannot create prompt.")

    result_df = df.copy()
    prompt_template = create_decomposition_prompt(annotation_examples)

    raw_breakdowns = []
    structured_results = []
    all_dimension_keys: Set[str] = set() # To collect all unique dimension keys found

    total_rows = len(df)
    print(f"Analyzing {total_rows} rows from column '{input_column}'...")

    for i, input_text in enumerate(result_df[input_column]):
        print(f"Processing row {i+1}/{total_rows}...")
        # Ensure input is string, handle potential NaN/None
        if pd.isna(input_text):
            print(f"Warning: Skipping row {i+1} due to missing input text (NaN).")
            raw_text_breakdown = "SKIPPED_EMPTY_INPUT"
            structured_dimensions = {}
        else:
            input_text_str = str(input_text)
            raw_text_breakdown = get_completion(prompt_template, input_text_str, model_id, provider)
            _, structured_dimensions = parse_llm_breakdown(raw_text_breakdown)

        raw_breakdowns.append(raw_text_breakdown)
        structured_results.append(structured_dimensions)
        all_dimension_keys.update(structured_dimensions.keys()) # Update set of keys

        # Add delay
        if delay_seconds > 0:
            time.sleep(delay_seconds)

    # Add base result columns
    result_df["breakdown_text"] = raw_breakdowns
    result_df["breakdown_json"] = [json.dumps(res) for res in structured_results]

    # Add dynamic columns for each unique dimension found
    print(f"Identified dimensions: {sorted(list(all_dimension_keys))}")
    for dim_key in sorted(list(all_dimension_keys)):
        # Extract data for this dimension, joining highlights with newline
        # Use .get(dim_key, []) to handle rows where the dimension wasn't found
        dim_data = ["\n".join(res.get(dim_key, [])) for res in structured_results]
        # Sanitize column name if needed (e.g., replace spaces, special chars)
        col_name = f"dim_{dim_key.lower().replace(' ', '_').replace('-', '_')}"
        result_df[col_name] = dim_data
        print(f"Added column: {col_name}")

    return result_df

# --- Main Execution Logic ---

def main(
    annotation_file,
    input_csv,
    output_csv,
    input_column="prompt",
    model_id=None,
    delay=0.5,
    provider="together",
    memory_mode=False,
    annotations_data=None,
    csv_data=None
):
    """
    Main function for breaking down instructions.
    
    Args:
        annotation_file: Path to the annotation file or "memory://" prefix if memory_mode=True
        input_csv: Path to the input CSV file or "memory://" prefix if memory_mode=True
        output_csv: Path to the output CSV file or "memory://" prefix if memory_mode=True
        input_column: Name of the column in input_csv that contains the prompts
        model_id: ID of the model to use
        delay: Delay between requests
        provider: Provider to use ("together" or "openai")
        memory_mode: If True, use data from memory instead of files
        annotations_data: Annotations data if memory_mode=True
        csv_data: CSV data as DataFrame if memory_mode=True
    
    Returns:
        DataFrame with predictions if memory_mode=True, None otherwise
    """
    print(f"Starting instruction breakdown with memory_mode={memory_mode}")
    
    # If memory mode, use the provided data
    if memory_mode:
        if annotations_data is None or csv_data is None:
            raise ValueError("annotations_data and csv_data must be provided when memory_mode=True")
        
        # Use the data from memory
        annotation_examples = load_annotation_examples_from_memory(annotations_data)
        df = csv_data
    else:
        # Load data from files
        annotation_examples = load_annotation_examples(annotation_file)
        df = pd.read_csv(input_csv)
    
    # Process the data with the original function
    results_df = process_dataframe_with_structure(
        df=df,
        annotation_examples=annotation_examples,
        input_column=input_column,
        model_id=model_id,
        delay_seconds=delay,
        provider=provider
    )
    
    # Save or return results
    if memory_mode:
        return results_df
    else:
        results_df.to_csv(output_csv, index=False)
        return None

def load_annotation_examples_from_memory(annotations_data):
    """
    Convert annotations data from memory to the format expected by the processing function.
    This is similar to load_annotation_examples but works with data already in memory.
    
    Args:
        annotations_data: Annotations data as a list of dictionaries
        
    Returns:
        List of example dictionaries in the standard format
    """
    examples = []
    
    for item in annotations_data:
        if "full_prompt" in item and "annotations" in item:
            # Convert to the standard format expected by the rest of the code
            converted_item = {
                "prompt": item["full_prompt"],
                "dimensions": {}
            }
            
            # Convert annotations to dimensions format
            for key, annotation in item["annotations"].items():
                if isinstance(annotation, dict) and "text" in annotation:
                    # Create a dimension entry with the annotation text as a highlight
                    converted_item["dimensions"][key] = {
                        "name": key.replace("_", " ").title(),  # Convert snake_case to Title Case
                        "highlights": [
                            {
                                "text": annotation["text"]
                                # No start/end positions
                            }
                        ]
                    }
                elif isinstance(annotation, str) and annotation:
                    # Handle the case where annotation is directly a string
                    converted_item["dimensions"][key] = {
                        "name": key.replace("_", " ").title(),
                        "highlights": [
                            {
                                "text": annotation
                            }
                        ]
                    }
            
            examples.append(converted_item)
    
    print(f"Loaded {len(examples)} annotation examples from memory")
    return examples

# --- Command Line Argument Parsing ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Decompose text based on examples using an LLM.")

    parser.add_argument("--annotation", type=str, required=True,
                        help="Path to the JSON file with annotation examples (e.g., email_annotation.json).")
    parser.add_argument("--input", type=str, required=True,
                        help="Path to the input CSV file containing text to process (e.g., email_response_examples.csv).")
    parser.add_argument("--output", type=str, required=True,
                        help="Path to save the output CSV file with decomposed results.")
    parser.add_argument("--column", type=str, required=True,
                        help="Name of the column in the input CSV containing the text to decompose (e.g., 'input', 'prompt').")
    parser.add_argument("--model", type=str, default="meta-llama/llama-3-3-70b-instruct",
                        help="Model identifier for the LLM (default: 'meta-llama/llama-3-3-70b-instruct'). Examples: 'ibm/granite-13b-instruct-v2'.")
    parser.add_argument("--delay", type=float, default=0.5,
                        help="Delay in seconds between LLM API calls (default: 0.5). Set to 0 to disable.")
    parser.add_argument("--provider", type=str, default="together", choices=["together", "rits"],
                        help="API provider to use (default: 'together'). Options: 'together', 'rits'.")

    args = parser.parse_args()

    # Set API keys from environment variables (ensure .env file is present or vars are set)
    load_dotenv()
    
    # Check if required configuration is present based on provider
    if args.provider.lower() == "together":
        together_api_key = os.getenv("TOGETHER_API_KEY")
        if not together_api_key:
            print("Error: TOGETHER_API_KEY environment variable must be set.")
            print("Please create a .env file or set it in your environment.")
            exit()  # Stop execution if config is missing
    elif args.provider.lower() == "rits":
        rits_host = os.getenv("RITS_HOST")
        rits_api_key = os.getenv("RITS_API_KEY")
        if not rits_host or not rits_api_key:
            print("Error: RITS_HOST and RITS_API_KEY environment variables must be set.")
            print("Please create a .env file or set them in your environment.")
            exit()  # Stop execution if config is missing

    main(
        annotation_file=args.annotation,
        input_csv=args.input,
        output_csv=args.output,
        input_column=args.column,
        model_id=args.model,
        delay=args.delay,
        provider=args.provider
    ) 