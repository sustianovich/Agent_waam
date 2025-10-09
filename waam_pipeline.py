import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, Mapping

from dotenv import load_dotenv
from langchain_community.tools.ddg_search import DuckDuckGoSearchRun
from langchain_ollama import OllamaLLM
from langsmith import traceable

from agent_profiles import AGENT_PROFILES
from ahp_analysis import run_analysis_from_data
from utils import build_explain_question, ddg_to_text, get_agent, load_comparisons, slugify_label

DEFAULT_AGENT = "WAAM Expert"

# Load .env variables before touching external services.
load_dotenv(dotenv_path=".env", override=True)

# Shared tools / models.
MODEL_NAME = "llama3"
llm = OllamaLLM(model=MODEL_NAME)
_MAX_CHARS = 8000  # optional: cap context size
web_search_tool = DuckDuckGoSearchRun()

@traceable(name="search_web")
def search(main_question: str, instructions: str, factor_1: str, factor_2: str) -> str:
    query = build_explain_question(main_question, instructions, factor_1, factor_2)
    if not query:
        return ""

    try:
        raw = web_search_tool.invoke(query)
        text = ddg_to_text(raw).strip()
        return text if len(text) <= _MAX_CHARS else text[: _MAX_CHARS - 3] + "..."
    except Exception as err:
        logging.warning("DuckDuckGo search failed: %s", err)
        return ""


PROMPT_TEMPLATE = """You are {description}
Your communication style is: {style}
Your top decision priorities are: {priorities}

Question:
{question}

Context:
{context}

Answer:"""


@traceable(name="explain_with_prompt")
def explain(
    agent_profile: Mapping[str, Any],
    main_question: str,
    instructions: str,
    factor_1: str,
    factor_2: str,
    context: str,
) -> str:
    question = build_explain_question(main_question, instructions, factor_1, factor_2)
    priorities = agent_profile.get("priorities", [])
    priorities_text = ", ".join(priorities) if isinstance(priorities, list) else str(priorities)
    formatted = PROMPT_TEMPLATE.format(
        description=agent_profile.get("description", "a helpful expert"),
        style=agent_profile.get("style", "Clear and concise"),
        priorities=priorities_text or "N/A",
        question=question,
        context=context or "No supplementary context available.",
    )
    response = llm.invoke(formatted)
    return response.strip() if isinstance(response, str) else response


@traceable(name="generate_ai_response")
def generate_ai_response(
    agent_name: str,
    factor_1: str,
    factor_2: str,
    main_question: str,
    instructions: str,
    question_index: int,
    use_memory: bool = True,
):
    profile = AGENT_PROFILES.get(agent_name, AGENT_PROFILES["WAAM Expert"])

    supplemental_instructions = instructions or ""
    try:
        search_context = search(main_question, instructions, factor_1, factor_2)
        explain_rationale = (
            explain(profile, main_question, instructions, factor_1, factor_2, search_context)
            if search_context
            else ""
        )
        if explain_rationale:
            prefix = f"{supplemental_instructions}\n\n" if supplemental_instructions else ""
            supplemental_instructions = f"{prefix}Supplementary context:\n{explain_rationale}"
    except Exception as err:
        logging.warning("Context enrichment failed: %s", err)

    agent = get_agent(agent_name, profile, llm, use_memory=use_memory)
    if question_index == 0 or not use_memory:
        agent.reset_memory()
    return agent.evaluate(factor_1, factor_2, main_question, supplemental_instructions)



def main(agent_name: str = DEFAULT_AGENT) -> None:
    survey_path = Path("input_data/google_form/00_google_form.json")
    try:
        with survey_path.open("r", encoding="utf-8") as fh:
            survey_metadata = json.load(fh)
    except FileNotFoundError:
        logging.error("Survey file not found at %s", survey_path.resolve())
        return

    comparisons = load_comparisons(survey_path)
    survey_title = survey_metadata.get("title", "Survey")
    analysis_label = slugify_label(survey_title, agent_name)

    ahp_records = []
    for idx, comparison in enumerate(comparisons):
        section_title = comparison.get("title", "")
        raw_instructions = comparison.get("instructions", "")
        if section_title:
            header = f"Section: {section_title}"
            combined_instructions = f"{header}\n\n{raw_instructions}" if raw_instructions else header
        else:
            combined_instructions = raw_instructions

        answer = generate_ai_response(
            agent_name,
            comparison["factor_1"],
            comparison["factor_2"],
            comparison["main_question"],
            combined_instructions,
            idx,
            use_memory=True,
        )
        ahp_records.append(
            {
                "section": comparison["section"],
                "title": section_title,
                "survey_title": survey_title,
                "agent": agent_name,
                "comparison": f"{comparison['factor_1']} vs {comparison['factor_2']}",
                "answer": answer["answer"],
            }
        )
        comparison_label = f"{comparison['factor_1']} vs {comparison['factor_2']}"
        prefix = f"[{section_title}] " if section_title else ""
        print(f"{prefix}Q{idx + 1}: {comparison_label} -> {answer['answer']} ({answer['agent_reasoning']})")

    analysis = run_analysis_from_data(ahp_records, title=analysis_label)
    if "error" in analysis:
        print(f"\nAHP analysis skipped: {analysis['error']}")
    else:
        print("\nAHP ranking:")
        for factor, score in analysis["ranking"]:
            print(f"- {factor}: {score:.4f}")
        print(f"\nConsistency index: {analysis['consistency_index']}")
        print(f"Consistency ratio: {analysis['consistency_ratio']}")
        print(f"Plot saved to: {analysis['plot_path']}")


if __name__ == "__main__":
    chosen_agent = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_AGENT
    main(chosen_agent)
