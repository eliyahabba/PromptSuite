import pandas as pd


def map_csv_to_json(df, annotations):
    output_jsons = []
    dimensions_to_each_dim = {}
    for dim, dim_val in annotations[0]["annotations"].items():
        dimensions_to_each_dim[dim] = dim_val["dimensions"]
    num_of_var_to_each_dim = {}
    for dim, dim_val in annotations[0]["annotations"].items():
        num_of_var_to_each_dim[dim] = dim_val["variant_counts"]
    for _, row in df.iterrows():
        annotations = {}
        full_prompt = row["prompt"]
        output_text = row.get("output", "")  # Get output if it exists
        placeholder_prompt = full_prompt
        for col in df.columns:
            if col in ["dim_breakdown", "output"]:  # Skip output and dim_breakdown columns
                continue
            if col.startswith("dim_"):
                print(f"Processing column: {col}")
                dim_name = col.replace("dim_", "")
                annotations[dim_name] = {"text": row[col], "dimensions": dimensions_to_each_dim[dim_name], "variant_counts": num_of_var_to_each_dim[dim_name]}
                if pd.notna(row[col]) and row[col] != "":
                    placeholder_prompt = placeholder_prompt.replace(row[col], "{" + dim_name.upper() + "}")
                else:
                    print(f"Skipping empty or NaN value in column: {col}")
                print(f"Updated placeholder prompt: {placeholder_prompt}")
        # Add output to annotations
        annotations["output"] = {"text": output_text}
        
        # Also add examples placeholder
        annotations["examples"] = {"text": None, "dimensions": dimensions_to_each_dim.get("examples", []),
                                 "variant_counts": num_of_var_to_each_dim.get("examples", {})}
        
        current_sample_json = {
            "full_prompt": full_prompt,
            "placeholder_prompt": placeholder_prompt,
            "annotations": annotations,
        }
        output_jsons.append(current_sample_json)
    return output_jsons
