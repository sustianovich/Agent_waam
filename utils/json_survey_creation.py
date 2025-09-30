import json
import re

def extract_survey_json(input_html_path, output_json_path):
    # Read the HTML file content
    with open(input_html_path, 'r', encoding='utf-8') as file:
        html_content = file.read()

    # Regex pattern to extract JSON inside the <script id="survey-data"> tag
    pattern = r'<script\s+type="application/json"\s+id="survey-data">\s*(\{.*?\})\s*</script>'
    match = re.search(pattern, html_content, re.DOTALL)

    if not match:
        raise ValueError("Could not find the embedded JSON with id='survey-data'")

    json_string = match.group(1)

    # Parse the JSON string
    try:
        survey_data = json.loads(json_string)
    except json.JSONDecodeError as e:
        raise ValueError("Failed to decode JSON: " + str(e))

    # Write to output file
    with open(output_json_path, 'w', encoding='utf-8') as out_file:
        json.dump(survey_data, out_file, indent=2, ensure_ascii=False)

    print(f"Survey JSON successfully extracted to '{output_json_path}'")

# Example usage:
extract_survey_json('templates\survey.html', 'output_survey.json')
