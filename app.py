from __future__ import annotations

import json
import logging
import os
import uuid
from pathlib import Path
from typing import Any, Dict, List, Sequence

from flask import Flask, jsonify, render_template, request, send_file, session

from agent_profiles import AGENT_PROFILES
from ahp_analysis import (
    run_aggregated_analysis,
    run_analysis_from_data,
    plot_ahp_results,
)
from utils import load_comparisons, slugify_label
from waam_pipeline import DEFAULT_AGENT, generate_ai_response


logging.basicConfig(level=logging.INFO)

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "dev-secret-key")

DEFAULT_SURVEY_PATH = Path("input_data/google_form/00_google_form.json")

PIPELINE_STATES: Dict[str, Dict[str, Any]] = {}
SESSION_RESULTS: Dict[str, List[List[Dict[str, Any]]]] = {}
AGENT_RUN_HISTORY: Dict[str, List[List[Dict[str, Any]]]] = {}
COMPLETED_SESSIONS: set[str] = set()


def _get_session_id() -> str:
    sid = session.get("session_id")
    if not sid:
        sid = uuid.uuid4().hex
        session["session_id"] = sid
    return sid


def _parse_bool(value: str | None, default: bool = True) -> bool:
    if value is None:
        return default
    return value.lower() in {"1", "true", "yes", "on"}


def _initialise_state(
    session_id: str,
    agent_name: str,
    num_runs: int,
    memory_enabled: bool,
) -> Dict[str, Any]:
    with DEFAULT_SURVEY_PATH.open("r", encoding="utf-8") as fh:
        survey_metadata = json.load(fh)
    survey_title = survey_metadata.get("title", "Survey")
    comparisons = load_comparisons(DEFAULT_SURVEY_PATH)
    state = {
        "agent": agent_name,
        "num_runs": max(1, num_runs),
        "memory": memory_enabled,
        "current_run": 1,
        "question_index": 0,
        "comparisons": comparisons,
        "pending_records": [],
        "survey_title": survey_title,
    }
    PIPELINE_STATES[session_id] = state
    SESSION_RESULTS[session_id] = []
    COMPLETED_SESSIONS.discard(session_id)
    return state


def _advance_or_complete(session_id: str, state: Dict[str, Any]) -> bool:
    if state["pending_records"]:
        snapshot = [record.copy() for record in state["pending_records"]]
        agent_name = state["agent"]
        AGENT_RUN_HISTORY.setdefault(agent_name, []).append(snapshot)
        SESSION_RESULTS.setdefault(session_id, []).append(snapshot)
        state["pending_records"] = []

    if state["current_run"] >= state["num_runs"]:
        PIPELINE_STATES.pop(session_id, None)
        COMPLETED_SESSIONS.add(session_id)
        return False

    state["current_run"] += 1
    state["question_index"] = 0
    return True


@app.get("/")
def index() -> Any:
    session_id = _get_session_id()
    # Touch state storage so Flask session persists even on reset
    PIPELINE_STATES.pop(session_id, None)
    COMPLETED_SESSIONS.discard(session_id)
    agents = sorted(AGENT_PROFILES.keys())
    return render_template("index.html", agents=agents, default_agent=DEFAULT_AGENT)


@app.get("/survey")
def survey() -> Any:
    return render_template("survey.html")


@app.get("/get_next_question")
def get_next_question() -> Any:
    session_id = _get_session_id()
    agent_name = request.args.get("agent", DEFAULT_AGENT)
    num_runs = int(request.args.get("num_runs", "1") or "1")
    memory_param = request.args.get("memory")

    if memory_param is not None:
        memory_enabled = _parse_bool(memory_param, default=True)
        try:
            state = _initialise_state(session_id, agent_name, num_runs, memory_enabled)
        except FileNotFoundError:
            return jsonify({"error": f"Survey definition not found at {DEFAULT_SURVEY_PATH}."}), 500
        except Exception as exc:  # pragma: no cover
            logging.exception("Failed to initialise survey state: %s", exc)
            return jsonify({"error": "Unable to start survey run."}), 500
        return jsonify(
            {
                "initialized": True,
                "total_questions": len(state["comparisons"]),
                "num_runs": state["num_runs"],
                "memory": state["memory"],
            }
        )

    state = PIPELINE_STATES.get(session_id)
    if not state:
        if session_id in COMPLETED_SESSIONS:
            return jsonify({"completed": True})
        return jsonify({"error": "Survey state not initialised. Start the survey first."}), 400

    comparisons: Sequence[Dict[str, Any]] = state["comparisons"]
    if not comparisons:
        return jsonify({"error": "No comparisons available in the survey definition."}), 500

    total_questions = len(comparisons)
    if state["question_index"] >= total_questions:
        has_more = _advance_or_complete(session_id, state)
        if not has_more:
            return jsonify({"completed": True})

    comparison = comparisons[state["question_index"]]
    section_title = comparison.get("title", "")
    raw_instructions = comparison.get("instructions", "")
    if section_title:
        header = f"Section: {section_title}"
        combined_instructions = f"{header}\n\n{raw_instructions}" if raw_instructions else header
    else:
        combined_instructions = raw_instructions
    run_number = state["current_run"]
    within_run_index = state["question_index"]

    result = generate_ai_response(
        agent_name=agent_name,
        factor_1=comparison["factor_1"],
        factor_2=comparison["factor_2"],
        main_question=comparison["main_question"],
        instructions=combined_instructions,
        question_index=within_run_index,
        use_memory=state["memory"],
    )

    state["pending_records"].append(
        {
            "section": comparison["section"],
            "title": section_title,
            "survey_title": state.get("survey_title"),
            "agent": agent_name,
            "comparison": f"{comparison['factor_1']} vs {comparison['factor_2']}",
            "answer": result["answer"],
        }
    )

    state["question_index"] += 1
    answered = state["question_index"]
    if state["question_index"] >= total_questions:
        _advance_or_complete(session_id, state)

    response_payload = {
        "run_number": run_number,
        "question_index": answered,
        "total_questions": total_questions,
        "comparison": f"{comparison['factor_1']} vs {comparison['factor_2']}",
        "section_title": section_title,
        "answer": result["answer"],
        "agent_reasoning": result["agent_reasoning"],
        "completed": False,
    }

    return jsonify(response_payload)


@app.post("/run_ahp")
def run_ahp() -> Any:
    payload = request.get_json(silent=True) or {}
    mode = payload.get("mode", "single")
    session_id = _get_session_id()

    if mode == "single":
        runs = SESSION_RESULTS.get(session_id)
        if not runs:
            return jsonify({"error": "No completed survey runs available for this session."}), 400
        comparisons = runs[-1]
        survey_title = (comparisons[0].get("survey_title") if comparisons else None) or "session"
        agent_label = (comparisons[0].get("agent") if comparisons else None) or DEFAULT_AGENT
        analysis_label = slugify_label(survey_title, agent_label, session_id[-8:])
        analysis = run_analysis_from_data(comparisons, title=analysis_label)
    elif mode == "agent":
        agent_name = payload.get("agent")
        if not agent_name:
            return jsonify({"error": "Agent parameter is required for agent mode."}), 400
        runs = AGENT_RUN_HISTORY.get(agent_name)
        if not runs:
            return jsonify({"error": f"No completed runs recorded for agent '{agent_name}'."}), 400
        analysis = run_aggregated_analysis(runs, title=slugify_label(agent_name, "aggregated"))
    elif mode == "all":
        runs = [run for run_list in AGENT_RUN_HISTORY.values() for run in run_list]
        if not runs:
            return jsonify({"error": "No completed runs recorded across agents yet."}), 400
        analysis = run_aggregated_analysis(runs, title=slugify_label("all", "agents"))
    else:
        return jsonify({"error": f"Unsupported mode '{mode}'."}), 400

    if "error" in analysis:
        return jsonify(analysis), 400

    return jsonify(
        {
            "factors": analysis["factors"],
            "priority_vector": analysis["priority_vector"],
            "ranking": analysis["ranking"],
            "consistency_index": analysis["consistency_index"],
            "consistency_ratio": analysis["consistency_ratio"],
            "plot_path": analysis.get("plot_path"),
        }
    )


@app.post("/plot_ahp")
def plot_ahp() -> Any:
    payload = request.get_json(silent=True) or {}
    factors = payload.get("factors") or []
    priorities = payload.get("priority_vector") or []
    title = payload.get("title", "AHP_Result")

    if not factors or not priorities:
        return jsonify({"error": "Both factors and priority_vector are required."}), 400

    if len(factors) != len(priorities):
        return jsonify({"error": "Factors and priority_vector must have the same length."}), 400

    try:
        numeric_priorities = [float(value) for value in priorities]
    except (TypeError, ValueError):
        return jsonify({"error": "Priority vector must contain numeric values."}), 400

    output_path = plot_ahp_results(factors, numeric_priorities, title)
    return send_file(str(output_path), mimetype="image/png")


if __name__ == "__main__":
    debug_mode = _parse_bool(os.environ.get("FLASK_DEBUG"), default=False)
    app.run(debug=debug_mode)
