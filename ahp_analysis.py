from __future__ import annotations

import json
import logging
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import List, Mapping, MutableMapping, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")  # Non-interactive backend for headless environments.
import matplotlib.pyplot as plt
import numpy as np


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

SURVEY_FOLDER = Path("survey_dbs")
PLOT_FOLDER = Path("static/plots")


def list_surveys(folder: Path = SURVEY_FOLDER) -> List[Path]:
    folder.mkdir(exist_ok=True)
    return sorted(folder.glob("*.db"))


def fetch_responses_from_db(db_path: Path) -> List[MutableMapping[str, str]]:
    try:
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT section, comparison, answer FROM survey")
            rows = cursor.fetchall()
    except Exception as exc:
        logging.error("Error reading database %s: %s", db_path, exc)
        return []

    return [
        {"section": row[0], "comparison": row[1], "answer": row[2]}
        for row in rows
    ]


def _ensure_comparison_format(record: Mapping[str, object]) -> Tuple[str, str, int]:
    if "comparison" in record:
        left, right = str(record["comparison"]).split(" vs ", 1)
    else:
        left = str(record["factor_1"])
        right = str(record["factor_2"])
    answer = int(record["answer"])
    return left.strip(), right.strip(), answer


def create_ahp_matrix(comparisons: Sequence[Mapping[str, object]]) -> Tuple[np.ndarray, List[str]]:
    factors = set()
    prepared: List[Tuple[str, str, int]] = []
    for entry in comparisons:
        left, right, answer = _ensure_comparison_format(entry)
        prepared.append((left, right, answer))
        factors.add(left)
        factors.add(right)

    ordered = sorted(factors)
    index_lookup = {name: idx for idx, name in enumerate(ordered)}

    matrix = np.ones((len(ordered), len(ordered)))
    for left, right, answer in prepared:
        i = index_lookup[left]
        j = index_lookup[right]
        if answer < 5:
            ratio = 1.0 / (10 - answer)
        elif answer > 5:
            ratio = float(answer - 4)
        else:
            ratio = 1.0
        matrix[i, j] = ratio
        matrix[j, i] = 1.0 / ratio

    return matrix, ordered


def create_geometric_aggregated_matrix(
    comparison_sets: Sequence[Sequence[Mapping[str, object]]],
) -> Tuple[np.ndarray, List[str]]:
    factor_names = set()
    prepared_sets: List[List[Tuple[str, str, int]]] = []
    for comparisons in comparison_sets:
        prepared: List[Tuple[str, str, int]] = []
        for entry in comparisons:
            left, right, answer = _ensure_comparison_format(entry)
            prepared.append((left, right, answer))
            factor_names.add(left)
            factor_names.add(right)
        prepared_sets.append(prepared)

    ordered = sorted(factor_names)
    index_lookup = {name: idx for idx, name in enumerate(ordered)}
    dimension = len(ordered)
    product_matrix = np.ones((dimension, dimension))

    for prepared in prepared_sets:
        matrix = np.ones((dimension, dimension))
        for left, right, answer in prepared:
            i = index_lookup[left]
            j = index_lookup[right]
            if answer < 5:
                ratio = 1.0 / (10 - answer)
            elif answer > 5:
                ratio = float(answer - 4)
            else:
                ratio = 1.0
            matrix[i, j] = ratio
            matrix[j, i] = 1.0 / ratio
        product_matrix *= matrix

    aggregated = product_matrix ** (1.0 / len(prepared_sets))
    return aggregated, ordered


def compute_priority_vector(matrix: np.ndarray) -> Tuple[np.ndarray, float]:
    eigenvalues, eigenvectors = np.linalg.eig(matrix)
    max_idx = int(np.argmax(eigenvalues))
    lambda_max = float(np.real(eigenvalues[max_idx]))
    vector = np.real(eigenvectors[:, max_idx])
    vector = vector / np.sum(vector)
    return vector, lambda_max


def compute_consistency_ratio(lambda_max: float, dimension: int) -> Tuple[float, float]:
    lambda_max = max(lambda_max, float(dimension))
    ci = (lambda_max - dimension) / (dimension - 1) if dimension > 1 else 0.0
    random_index = {1: 0.00, 2: 0.00, 3: 0.58, 4: 0.90, 5: 1.12, 6: 1.24, 7: 1.32, 8: 1.41, 9: 1.45}.get(dimension, 1.45)
    cr = ci / random_index if random_index else 0.0
    return round(ci, 5), round(cr, 5)


def plot_ahp_results(factors: Sequence[str], priorities: Sequence[float], title: str) -> Path:
    PLOT_FOLDER.mkdir(parents=True, exist_ok=True)
    filename = f"{title.replace(' ', '_')}.png"
    output_path = PLOT_FOLDER / filename

    plt.figure(figsize=(10, 6))
    plt.barh(factors, priorities, color="skyblue")
    plt.xlabel("Priority")
    plt.title(f"AHP Priority Vector - {title}")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

    return output_path


def run_analysis_from_data(
    comparisons: Sequence[Mapping[str, object]],
    title: str = "AHP Analysis",
) -> Mapping[str, object]:
    if not comparisons:
        return {"error": "No comparisons provided for analysis."}

    matrix, factors = create_ahp_matrix(comparisons)
    priority_vector, lambda_max = compute_priority_vector(matrix)
    consistency_index, consistency_ratio = compute_consistency_ratio(lambda_max, len(factors))
    ranking = [
        (factor, float(score))
        for factor, score in sorted(zip(factors, priority_vector), key=lambda item: item[1], reverse=True)
    ]

    logging.info("Consistency Index: %s", consistency_index)
    logging.info("Consistency Ratio: %s", consistency_ratio)

    return {
        "factors": factors,
        "priority_vector": priority_vector.tolist(),
        "ranking": ranking,
        "lambda_max": round(lambda_max, 5),
        "consistency_index": consistency_index,
        "consistency_ratio": consistency_ratio,
        "matrix": matrix.tolist(),
        "plot_path": str(plot_ahp_results(factors, priority_vector, title)),
    }


def run_aggregated_analysis(
    comparison_sets: Sequence[Sequence[Mapping[str, object]]],
    title: str = "Aggregated AHP Analysis",
) -> Mapping[str, object]:
    if not comparison_sets:
        return {"error": "No comparison sets provided for aggregation."}

    matrix, factors = create_geometric_aggregated_matrix(comparison_sets)
    priority_vector, lambda_max = compute_priority_vector(matrix)
    consistency_index, consistency_ratio = compute_consistency_ratio(lambda_max, len(factors))
    ranking = [
        (factor, float(score))
        for factor, score in sorted(zip(factors, priority_vector), key=lambda item: item[1], reverse=True)
    ]

    logging.info("Aggregated Consistency Index: %s", consistency_index)
    logging.info("Aggregated Consistency Ratio: %s", consistency_ratio)

    return {
        "factors": factors,
        "priority_vector": priority_vector.tolist(),
        "ranking": ranking,
        "lambda_max": round(lambda_max, 5),
        "consistency_index": consistency_index,
        "consistency_ratio": consistency_ratio,
        "matrix": matrix.tolist(),
        "plot_path": str(plot_ahp_results(factors, priority_vector, title)),
    }


def _group_by_section(records: Sequence[Mapping[str, object]]) -> Mapping[str, List[Mapping[str, object]]]:
    sections: MutableMapping[str, List[Mapping[str, object]]] = {}
    for record in records:
        section = str(record.get("section", ""))
        sections.setdefault(section, []).append(record)
    return sections


def run_ahp_analysis(selected_surveys: Sequence[Path]) -> Mapping[str, Mapping[str, Mapping[str, object]]]:
    summary = {}
    for db_path in selected_surveys:
        logging.info("Running AHP on %s", db_path.name)
        responses = fetch_responses_from_db(db_path)
        if not responses:
            logging.warning("No data found in %s. Skipping.", db_path.name)
            continue

        section_groups = _group_by_section(responses)
        per_section = {}
        for section, items in section_groups.items():
            title = section or db_path.stem
            per_section[section] = run_analysis_from_data(items, title=title)
        summary[db_path.name] = per_section
    return summary


def save_results(data: Mapping[str, object], filename: Path) -> None:
    with filename.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)


def main() -> None:
    surveys = list_surveys()
    if not surveys:
        print("No survey databases found in 'survey_dbs'.")
        return

    print("\nAvailable surveys:")
    for idx, path in enumerate(surveys, start=1):
        print(f"{idx}. {path.name}")

    selection = input("\nEnter survey numbers to analyse (comma-separated) or 'all': ").strip()
    if selection.lower() == "all":
        chosen = surveys
    else:
        try:
            indexes = [int(x) - 1 for x in selection.split(",")]
            chosen = [surveys[i] for i in indexes]
        except (ValueError, IndexError):
            print("Invalid selection. Exiting.")
            return

    results = run_ahp_analysis(chosen)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = Path(f"ahp_results_{timestamp}.json")
    save_results(results, output_file)

    print(f"\nAHP analysis complete. Results saved to {output_file}.")


if __name__ == "__main__":
    main()
