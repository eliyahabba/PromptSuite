# File: pages/2_define_dimensions.py
import time

import streamlit as st

DEFAULT_DIMENSIONS = [
  {
    "id": "order_of_composition",
    "name": "Order of composition",
    "description": "Vary the order in which components of the input (e.g., task instruction, context, examples) are presented.",
    "examples": [
      "Place the instruction before the context.",
      "Give examples first, followed by task instruction."
    ]
  },
  {
    "id": "paraphrases",
    "name": "Paraphrases",
    "description": "Rephrase the input without changing its semantic meaning.",
    "examples": [
      "Reword the question: 'What is the capital of France?' → 'Can you tell me which city is the capital of France?'",
      "Change phrasing of instructions: 'Summarize the following text' → 'Write a short summary of this passage.'"
    ]
  },
  {
    "id": "non_semantic",
    "name": "Non-semantic / structural changes",
    "description": "Change surface-level aspects like punctuation, formatting, or variable names without altering meaning.",
    "examples": [
      "Change line breaks or bullet points without changing the content.",
      "Replace variable name 'x' with 'temp' in a code-related task."
    ]
  },
  {
    "id": "which_few_shot",
    "name": "Which few-shot examples",
    "description": "Vary the specific few-shot examples used while keeping the task the same.",
    "examples": [
      "Use examples that highlight edge cases vs. typical cases.",
      "Swap examples drawn from different subdomains (e.g., sports vs. politics in sentiment analysis)."
    ]
  },
  {
    "id": "how_many_few_shot",
    "name": "How many few-shot examples",
    "description": "Change the number of few-shot examples provided in the prompt.",
    "examples": [
      "Use 1, 3, or 5 examples to observe changes in performance.",
      "Try zero-shot, one-shot, and few-shot versions of the same task."
    ]
  },
  {
    "id": "irrelevant_context",
    "name": "Add irrelevant context",
    "description": "Insert unrelated or off-topic information alongside the main input to test robustness.",
    "examples": [
      "Include a weather report before a math problem.",
      "Add a random Wikipedia paragraph before the instruction."
    ]
  },
  {
    "id": "multi_doc_order",
    "name": "Order of provided documents",
    "description": "Change the order in which multiple source documents are provided for a multi-document task.",
    "examples": [
      "Provide source A before B, then swap to B before A.",
      "Shuffle document order in summarization or QA tasks."
    ]
  },
  {
    "id": "multi_doc_concat",
    "name": "How to concatenate documents",
    "description": "Change the formatting or method used to combine multiple documents.",
    "examples": [
      "Separate documents using headings, newline delimiters, or bullet points.",
      "Merge all text into a single block vs. clearly delineated sources."
    ]
  },
  {
    "id": "multi_doc_irrelevant",
    "name": "Add irrelevant documents",
    "description": "Add unrelated documents to a multi-document context to test the model’s ability to focus on relevant information.",
    "examples": [
      "Include a document on cooking in a QA task about science.",
      "Add a news article on sports to distract from relevant financial reports."
    ]
  },
  {
    "id": "mc_order_of_answers",
    "name": "Order of answers",
    "description": "Change the sequence in which answer options are presented in multiple-choice questions.",
    "examples": [
      "Shuffle the order of options A, B, C, D.",
      "Place the correct answer first vs. last."
    ]
  },
  {
    "id": "mc_enumeration",
    "name": "Enumeration (letters, numbers, etc)",
    "description": "Change the enumeration style used to list answer choices.",
    "examples": [
      "Use A/B/C/D vs. 1/2/3/4 vs. no enumeration.",
      "List answers with dashes or bullets instead of labels."
    ]
  }
]

def render():
    st.title("Step 3: Define or Review Dimensions")
    st.write("Here are the current default and custom dimensions. You can add more if needed.")

    # Initialize base dimensions if not present
    if "base_dimensions" not in st.session_state:
        st.session_state.base_dimensions = DEFAULT_DIMENSIONS

    if "custom_dimensions" not in st.session_state:
        st.session_state.custom_dimensions = []

    all_dimensions = st.session_state.base_dimensions + st.session_state.custom_dimensions

    # Display current dimensions
    st.markdown("### 📋 Current Dimensions")
    # with st.expander("📋 Current Dimensions"):
    for dim in all_dimensions:
        st.markdown(f"""
        - **Name**: {dim['name']}  
          **Explanation**: {dim.get('description', '-') or '-'}  
          **Example**: _{', '.join(dim.get('examples', [])) or '-'}_
        """)

    # Add a new dimension
    st.write("### ➕ Add a new dimension")

    with st.form("add_new_dimension"):
        name = st.text_input("Dimension Name")
        description = st.text_area("Description")
        example = st.text_input("Example (optional)")

        submitted = st.form_submit_button("Add Dimension")

        if submitted:
            if name.strip() == "":
                st.warning("Name is required.")
            else:
                new_dim = {
                    "id": name.lower().replace(" ", "_"),
                    "name": name,
                    "description": description,
                    "examples": [example] if example else []
                }
                st.session_state.custom_dimensions.append(new_dim)
                st.success(f"Added new dimension: {name}")
                # stop for 2 seconds
                time.sleep(1.5)
                st.rerun()

    # Button to continue
    if st.button("Continue to Assign Dimensions"):
        st.session_state.page = 4
        st.rerun()
