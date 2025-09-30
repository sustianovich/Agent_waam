import os
import sqlite3
import json
import numpy as np
import logging
from datetime import datetime

import matplotlib
matplotlib.use('Agg')  # ✅ Usa backend sin interfaz gráfica

import matplotlib.pyplot as plt

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Directory where survey database files are stored
SURVEY_FOLDER = "survey_dbs"

def list_surveys():
    if not os.path.exists(SURVEY_FOLDER):
        os.makedirs(SURVEY_FOLDER)
    return [f for f in os.listdir(SURVEY_FOLDER) if f.endswith(".db")]

def fetch_responses_from_db(db_path):
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT section, comparison, answer FROM survey")
        responses = cursor.fetchall()
        conn.close()
        return [{"section": row[0], "comparison": row[1], "answer": row[2]} for row in responses]
    except Exception as e:
        logging.error(f"❌ Error reading database {db_path}: {e}")
        return None

def create_ahp_matrix(comparisons_list):
    factors = set()
    for comparison in comparisons_list:
        f1, f2 = comparison["comparison"].split(" vs ")
        factors.add(f1.strip())
        factors.add(f2.strip())

    factors = sorted(list(factors))
    factor_index = {factor: idx for idx, factor in enumerate(factors)}
    num_factors = len(factors)

    matrix = np.ones((num_factors, num_factors))

    for comparison in comparisons_list:
        f1, f2 = comparison["comparison"].split(" vs ")
        i = factor_index[f1.strip()]
        j = factor_index[f2.strip()]
        value = int(comparison["answer"])

        if value < 5:
            ratio = 1 / (10 - value)
        elif value > 5:
            ratio = value - 4
        else:
            ratio = 1.0

        matrix[i, j] = ratio
        matrix[j, i] = 1 / ratio

    return matrix, factors


def compute_priority_vector(matrix):
    eigenvalues, eigenvectors = np.linalg.eig(matrix)
    if np.any(eigenvalues < 0):
        logging.warning(f"⚠️ Negative eigenvalue found: {eigenvalues}")
    max_eigenvalue_index = np.argmax(eigenvalues)
    lambda_max = np.real(eigenvalues[max_eigenvalue_index])
    priority_vector = np.real(eigenvectors[:, max_eigenvalue_index])
    priority_vector /= np.sum(priority_vector)
    return priority_vector, lambda_max


def compute_consistency_ratio(lambda_max, num_factors):
    lambda_max = max(lambda_max, num_factors)  # Clamp to avoid negatives
    CI = (lambda_max - num_factors) / (num_factors - 1)
    RI_values = {1: 0.00, 2: 0.00, 3: 0.58, 4: 0.90, 5: 1.12, 6: 1.24, 7: 1.32, 8: 1.41, 9: 1.45}
    RI = RI_values.get(num_factors, 1.45)
    CR = CI / RI if RI > 0 else 0
    return round(CI, 5), round(CR, 5)

def plot_ahp_results(factors, priorities, section_title, output_dir="static/plots"):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    filename = f"{section_title.replace(' ', '_')}.png"
    filepath = os.path.join(output_dir, filename)

    plt.figure(figsize=(10, 6))
    plt.barh(factors, priorities, color='skyblue')
    plt.xlabel('Priority')
    plt.title(f'AHP Priority Vector - {section_title}')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(filepath)
    plt.close()  # ❗ Important: avoid Tkinter crash

    return filepath



def create_geometric_aggregated_matrix(all_comparisons_lists):
    """Builds a consensus matrix using geometric mean from multiple comparison lists."""
    # First collect all unique factors from all comparison lists
    all_factors = set()
    for comparisons in all_comparisons_lists:
        for entry in comparisons:
            f1, f2 = entry["comparison"].split(" vs ")
            all_factors.add(f1.strip())
            all_factors.add(f2.strip())

    factors = sorted(list(all_factors))
    factor_index = {f: i for i, f in enumerate(factors)}
    n = len(factors)

    # Initialize list of matrices (one per survey)
    matrices = []

    for comparisons in all_comparisons_lists:
        matrix = np.ones((n, n))
        for entry in comparisons:
            i = factor_index[entry["comparison"].split(" vs ")[0].strip()]
            j = factor_index[entry["comparison"].split(" vs ")[1].strip()]
            value = int(entry["answer"])

            if value < 5:
                val = 1 / (10 - value)
            elif value > 5:
                val = value - 4
            else:
                val = 1.0

            matrix[i, j] = val
            matrix[j, i] = 1 / val
        matrices.append(matrix)

    # Aggregate using geometric mean
    product_matrix = np.ones((n, n))
    for matrix in matrices:
        product_matrix *= matrix  # element-wise multiply

    aggregated_matrix = product_matrix ** (1 / len(matrices))  # geometric mean
    return aggregated_matrix, factors


def run_analysis_from_data(comparisons_list, is_multiple=False):
    """
    Perform AHP analysis on a single set or multiple sets of comparisons.
    
    Args:
        comparisons_list: Either a single list of comparisons (dicts), or a list of such lists.
        is_multiple (bool): If True, will treat comparisons_list as multiple sets and use geometric aggregation.

    Returns:
        dict: Results containing priority vector, consistency metrics, and ranking.
    """
    if not comparisons_list:
        return {"error": "No data provided for AHP analysis."}

    if is_multiple:
        matrix, factors = create_geometric_aggregated_matrix(comparisons_list)
        section_title = "Aggregated AHP Analysis"
    else:
        matrix, factors = create_ahp_matrix(comparisons_list)
        section_title = "Single AHP Analysis"

    priority_vector, lambda_max = compute_priority_vector(matrix)
    CI, CR = compute_consistency_ratio(lambda_max, len(factors))
    ranking = sorted(zip(factors, priority_vector), key=lambda x: x[1], reverse=True)

    print(f"\n📈 Consistency Index: {CI}")
    print(f"📈 Consistency Ratio: {CR}")
    print("📊 AHP Matrix:")
    print(matrix)

    return {
        "factors": factors,
        "priority_vector": priority_vector.tolist(),
        "ranking": ranking,
        "lambda_max": round(lambda_max, 5),
        "consistency_index": CI,
        "consistency_ratio": CR
    }


def run_ahp_analysis(selected_surveys):
    results = {}
    for survey_file in selected_surveys:
        survey_path = os.path.join(SURVEY_FOLDER, survey_file)
        logging.info(f"🔍 Running AHP on: {survey_file}")
        responses = fetch_responses_from_db(survey_path)
        if not responses:
            logging.warning(f"⚠️ No data found in {survey_file}. Skipping...")
            continue

        sections = {}
        for entry in responses:
            section = entry["section"]
            if section not in sections:
                sections[section] = []
            sections[section].append(entry)

        ahp_results = {}
        for section, comparisons_list in sections.items():
            logging.info(f"📊 Processing AHP for section: {section}")
            matrix, factors = create_ahp_matrix(comparisons_list)
            priority_vector, lambda_max = compute_priority_vector(matrix)
            CI, CR = compute_consistency_ratio(lambda_max, len(factors))
            ranking = sorted(zip(factors, priority_vector), key=lambda x: x[1], reverse=True)

            print(f"\n📈 Section: {section}")
            print(f"📈 Consistency Index: {CI}")
            print(f"📈 Consistency Ratio: {CR}")
            img_path = plot_ahp_results(factors, priority_vector, section)


            ahp_results[section] = {
                    "factors": factors,
                    "priority_vector": priority_vector.tolist(),
                    "ranking": ranking,
                    "lambda_max": round(lambda_max, 5),
                    "consistency_index": CI,
                    "consistency_ratio": CR,
                    "plot_path": img_path
                }


        results[survey_file] = ahp_results

    return results

if __name__ == "__main__":
    surveys = list_surveys()

    if not surveys:
        print("❌ No survey databases found in 'survey_dbs' folder.")
    else:
        print("\n📂 Available Surveys:")
        for idx, file in enumerate(surveys, 1):
            print(f"{idx}. {file}")

        choices = input("\nEnter survey numbers to analyze (comma-separated) or 'all': ").strip()

        if choices.lower() == "all":
            selected_surveys = surveys
        else:
            try:
                selected_surveys = [surveys[int(idx) - 1] for idx in choices.split(",")]
            except (ValueError, IndexError):
                print("❌ Invalid input. Exiting.")
                exit()

        ahp_results = run_ahp_analysis(selected_surveys)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"ahp_results_{timestamp}.json"
        with open(output_file, "w", encoding="utf-8") as file:
            json.dump(ahp_results, file, ensure_ascii=False, indent=2)

        print(f"\n✅ AHP analysis completed. Results saved to {output_file}.")
