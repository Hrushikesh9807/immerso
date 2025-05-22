import json
import os
import re

# --- Configuration ---
# This should match the directory where your llava_S3_structured_summary... files are located
RESULTS_BASE_OUTPUT_DIR = "/home/ldvgpu/anirud/projects/immerso/results/mrbean/pipeline_results"

# List of method names you processed.
# If you ran all three scripts, it would be:
METHOD_NAMES = ["image_embedding"]
# If you only ran for caption_embedding and got the error for that one,
# you can just use: METHOD_NAMES = ["caption_embedding"]

# --- Helper Function to Extract XML-like Sections ---
def extract_sections_xml(text):
    """
    Parses text containing <Background>, <Characters>, and <Story> tags.
    Returns a dictionary with these sections.
    """
    sections = {
        "background": "Not specified or missing.",
        "characters": "Not specified or missing.",
        "story": "Not specified or missing."
    }
    if not text:
        return sections

    # Case-insensitive and dotall to match across newlines
    bg_match = re.search(r"<Background>(.*?)</Background>", text, re.IGNORECASE | re.DOTALL)
    if bg_match:
        sections["background"] = bg_match.group(1).strip()

    char_match = re.search(r"<Characters>(.*?)</Characters>", text, re.IGNORECASE | re.DOTALL)
    if char_match:
        sections["characters"] = char_match.group(1).strip()

    story_match = re.search(r"<Story>(.*?)</Story>", text, re.IGNORECASE | re.DOTALL)
    if story_match:
        sections["story"] = story_match.group(1).strip()
        # Explicitly check if the extracted story is just a placeholder
        if sections["story"].lower() in ["not specified.", "no story segment found."]:
             sections["story"] = "Not specified or missing." # Standardize placeholder

    return sections

# --- Main Processing ---
def create_final_summary_files(base_dir, methods):
    print(f"Processing summaries in: {base_dir}")
    for method_name in methods:
        print(f"\n--- Processing method: {method_name} ---")

        input_filename = f"llava_S3_structured_summary_{method_name}.json"
        input_filepath = os.path.join(base_dir, input_filename)

        output_filename = f"final_summary_{method_name}.json" # New desired filename
        output_filepath = os.path.join(base_dir, output_filename)

        if not os.path.exists(input_filepath):
            print(f"Input file not found: {input_filepath}. Skipping.")
            continue

        try:
            with open(input_filepath, 'r') as f:
                data = json.load(f)

            # The raw LLaVA output is expected to be the value of the first (and likely only) key
            # e.g., {"llava_s3_structured_summary_caption_embedding": "<Background>...</Background>..."}
            raw_llava_output_text = ""
            if isinstance(data, dict) and data:
                raw_llava_output_text = next(iter(data.values()), "") # Get first value
            elif isinstance(data, str): # If the file directly contains the string (less likely based on original script)
                 raw_llava_output_text = data
            else:
                print(f"Unexpected data format in {input_filepath}. Expected a dict with one entry or a raw string.")
                continue

            if not raw_llava_output_text:
                print(f"No text content found in {input_filepath}. Skipping.")
                continue

            print(f"Parsing content from: {input_filepath}")
            parsed_sections = extract_sections_xml(raw_llava_output_text)

            # Prepare the final JSON structure
            final_summary_data = {
                "method": method_name,
                "source_file": input_filename, # To trace back
                "parsed_summary": {
                    "background": parsed_sections["background"],
                    "characters": parsed_sections["characters"],
                    "story": parsed_sections["story"]
                }
            }
            # Add the raw text as well, could be useful
            final_summary_data["raw_llava_structured_output"] = raw_llava_output_text


            with open(output_filepath, 'w') as f:
                json.dump(final_summary_data, f, indent=4)
            print(f"✅ Successfully created final summary: {output_filepath}")

            # Additionally, you can check if the story is valid here
            if parsed_sections["story"] in ["Not specified or missing.", "Not specified.", "no story segment found."]:
                print(f"⚠️ Note: The extracted story for '{method_name}' is a placeholder or missing.")


        except json.JSONDecodeError:
            print(f"Error decoding JSON from {input_filepath}. Skipping.")
        except Exception as e:
            print(f"An error occurred while processing {input_filepath}: {e}")

if __name__ == "__main__":
    # Example for the 'mrbean' results path you provided:
    # RESULTS_BASE_OUTPUT_DIR = "/home/ldvgpu/anirud/projects/immerso/results/mrbean/pipeline_results"

    # Example for the 'elevator' results path (if you switched videos):
    RESULTS_BASE_OUTPUT_DIR = "/home/ldvgpu/anirud/projects/immerso/results_bajirao/10/pipeline_results"


    # Ensure the METHOD_NAMES list is correct for the files you want to process
    # If you only ran caption_embedding for mrbean and it failed, use:
    # METHOD_NAMES_TO_PROCESS = ["caption_embedding"]
    # If you ran all for elevator and want to process all:
    METHOD_NAMES_TO_PROCESS = ["image_embedding","caption_embedding"]


    if not os.path.isdir(RESULTS_BASE_OUTPUT_DIR):
        print(f"ERROR: The specified results directory does not exist: {RESULTS_BASE_OUTPUT_DIR}")
    else:
        create_final_summary_files(RESULTS_BASE_OUTPUT_DIR, METHOD_NAMES_TO_PROCESS)