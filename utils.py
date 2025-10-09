from pathlib import Path
from typing import Any, Dict, List, Mapping, Tuple
import json

from agent_class import SupportsInvoke, WaaMAgent


# For the SEARCH function

def _clean_html_breaks(text: str) -> str:
    """Normalise <br> tags to newlines to keep prompts readable."""
    return (
        text.replace("<br/>", "\n")
        .replace("<br />", "\n")
        .replace("<br>", "\n")
        .strip()
    )


def build_explain_question(
    main_question: str,
    instructions: str,
    factor_1: str,
    factor_2: str,
) -> str:
    """Compose the text we feed to search/explain with full comparison context."""
    parts: List[str] = []
    if main_question:
        parts.append(main_question.strip())
    if instructions:
        parts.append(f"Instructions:\n{_clean_html_breaks(instructions)}")
    parts.append(f"Compare the following factors:\n- {factor_1}\n- {factor_2}")
    return "\n\n".join(parts)


def ddg_to_text(res: Any) -> str:
    """Flatten possible DuckDuckGo result shapes into plain text."""
    if isinstance(res, str):
        return res

    if isinstance(res, list):
        lines = []
        for item in res:
            if isinstance(item, dict):
                title = item.get("title") or item.get("heading")
                snippet = item.get("snippet") or item.get("body") or item.get("abstract")
                link = item.get("link") or item.get("href") or item.get("url")
                parts = [p for p in (title, snippet, link) if p]
                lines.append(" - ".join(map(str, parts)))
            else:
                lines.append(str(item))
        return "\n".join(lines)

    if isinstance(res, dict):
        # last-resort stringify
        return json.dumps(res, ensure_ascii=False)

    return str(res)


# For the EXPLAIN function

_AGENT_CACHE: Dict[Tuple[str, bool], WaaMAgent] = {}


def get_agent(
    agent_name: str,
    profile: Mapping[str, Any],
    llm: SupportsInvoke,
    use_memory: bool = True,
) -> WaaMAgent:
    cache_key = (agent_name, use_memory)
    agent = _AGENT_CACHE.get(cache_key)
    if agent is None:
        agent = WaaMAgent(agent_name, profile, llm, use_memory=use_memory)
        _AGENT_CACHE[cache_key] = agent
    return agent


# For the GENERATE_AI_RESPONSE function

def load_comparisons(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8") as fh:
        survey_data = json.load(fh)

    comparisons: List[Dict[str, str]] = []
    for section in survey_data.get("sections", []):
        question_text = section.get("main_question") or section.get("description") or ""
        instructions = section.get("instructions", "")
        for comparison in section.get("comparisons", []):
            comparisons.append(
                {
                    "section": section.get("section", ""),
                    "title": section.get("title", ""),
                    "main_question": question_text,
                    "instructions": instructions,
                    "factor_1": comparison["factor_1"],
                    "factor_2": comparison["factor_2"],
                }
            )
    return comparisons


def slugify_label(*parts: str) -> str:
    combined = "_".join(part.strip() for part in parts if part)
    if not combined:
        return "analysis"
    safe = combined.replace(" ", "_")
    return "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in safe)


__all__ = [
    "build_explain_question",
    "ddg_to_text",
    "get_agent",
    "load_comparisons",
    "slugify_label",
]
