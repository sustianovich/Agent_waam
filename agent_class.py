from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Protocol
import re

from langchain.prompts import PromptTemplate
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from langchain.memory import ConversationBufferMemory


class SupportsInvoke(Protocol):
    def invoke(self, prompt: str, **kwargs: Any) -> Any: ...


def _clean_html_breaks(text: str) -> str:
    if not text:
        return ""
    return (
        text.replace("<br/>", " ")
        .replace("<br />", " ")
        .replace("<br>", " ")
        .strip()
    )


class WaaMAgent:
    """Persona-aligned agent that scores a pair of factors from 1–9.

    Output contract (returned dict):
      - answer: int in [1, 9]
      - agent_reasoning: short string (1–3 sentences)
    """

    def __init__(self, name: str, profile: Mapping[str, Any], llm: SupportsInvoke, use_memory: bool = True):
        self.name = name
        self.profile = profile
        self.llm = llm
        self.use_memory = use_memory

        self.memory: Optional[ConversationBufferMemory]
        if self.use_memory:
            self.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        else:
            self.memory = None  # No memory used

        # Define output parser
        self.parser = StructuredOutputParser.from_response_schemas([
            ResponseSchema(
                name="answer",
                description="An integer value between 1 and 9 inclusive, representing the preference strength"
            ),
            ResponseSchema(
                name="agent_reasoning",
                description="A short, concise explanation (1-3 sentences) justifying the chosen value"
            )
        ])

        # Define prompt template (now includes optional chat history and agent name)
        self.prompt_template = PromptTemplate(
            input_variables=[
                "agent_name",
                "main_question",
                "instructions",
                "factor_1",
                "factor_2",
                "description",
                "style",
                "priorities",
                "format_instructions",
                "chat_history"
            ],
            template=(
                "You are {description}\n"
                "Your communication style is: {style}\n"
                "Your top decision priorities are: {priorities}\n\n"
                "Agent: {agent_name}\n"
                "{chat_history}\n"
                "Main Question: {main_question}\n"
                "Instructions: {instructions}\n\n"
                "Now compare the following two factors:\n"
                "- {factor_1}\n"
                "- {factor_2}\n\n"
                "Choose a number between 1 and 9 and respond in the required JSON format.\n"
                "Guidance:\n"
                "- 1–4 -> {factor_1} is more important\n"
                "- 5 -> Equal importance\n"
                "- 6–9 -> {factor_2} is more important\n\n"
                "{format_instructions}\n"
            )
        )

    # ------------------------------
    # Public methods
    # ------------------------------
    def build_prompt(self, factor_1: str, factor_2: str, main_question: str, instructions: str) -> str:
        """Create the prompt string used to evaluate the two factors."""
        priorities = ", ".join(self.profile.get("priorities", []))
        chat_history = ""
        if self.memory:
            # Minimal, compact history formatting
            msgs = self.memory.load_memory_variables({}).get("chat_history", [])
            if msgs:
                lines: List[str] = []
                for m in msgs:
                    role = getattr(m, "type", getattr(m, "role", "user"))
                    content = getattr(m, "content", "")
                    lines.append(f"[{role}] {content}")
                chat_history = "Previous context:\n" + "\n".join(lines) + "\n\n"

        return self.prompt_template.format(
            agent_name=self.name,
            factor_1=factor_1,
            factor_2=factor_2,
            main_question=main_question,
            instructions=_clean_html_breaks(instructions),
            description=self.profile["description"],
            style=self.profile["style"],
            priorities=priorities,
            format_instructions=self.parser.get_format_instructions(),
            chat_history=chat_history,
        )

    def evaluate(self, factor_1: str, factor_2: str, main_question: str, instructions: str) -> Dict[str, Any]:
        try:
            prompt = self.build_prompt(factor_1, factor_2, main_question, instructions)
            raw = self.llm.invoke(prompt)
            response = raw.strip() if isinstance(raw, str) else str(raw)

            # Save to memory (best-effort)
            if self.memory:
                try:
                    self.memory.save_context({"input": prompt}, {"output": response})
                except Exception:
                    pass

            # Primary: structured parse
            try:
                parsed = self.parser.parse(response)
            except Exception:
                parsed = self._fallback_parse(response)

            # Post-validate / coerce
            ans = self._coerce_answer(parsed.get("answer"))
            reason = self._clean_reason(parsed.get("agent_reasoning", ""), factor_1, factor_2)
            return {"answer": ans, "agent_reasoning": reason}

        except Exception as e:
            return {
                "answer": 5,
                "agent_reasoning": f"Fallback: Equal importance due to model failure ({type(e).__name__})."
            }

    def reset_memory(self) -> None:
        if self.memory:
            self.memory.clear()

    # ------------------------------
    # Helpers
    # ------------------------------
    @staticmethod
    def _coerce_answer(x: Any) -> int:
        # Try int
        try:
            n = int(str(x).strip())
        except Exception:
            # Extract first 1–9 digit from free text
            m = re.search(r"\b([1-9])\b", str(x))
            n = int(m.group(1)) if m else 5
        # Clamp to [1, 9]
        return max(1, min(9, n))

    @staticmethod
    def _clean_reason(reason: str, f1: str, f2: str) -> str:
        reason = re.sub(r"\s+", " ", (reason or "")).strip()
        if not reason:
            # Minimal auto-justification if empty
            return f"Tie-breaker around 5 due to comparable importance of {f1} and {f2}."
        # Keep it concise (<= 2 sentences when possible)
        # Soft trim to ~240 chars
        if len(reason) > 240:
            reason = reason[:237].rstrip() + "…"
        return reason

    def _fallback_parse(self, text: str) -> Dict[str, Any]:
        """Be liberal in what we accept: pull an integer 1–9 and a brief reason."""
        # Try JSON inside text
        m = re.search(r"\{.*?\}", text, flags=re.S)
        if m:
            import json
            try:
                obj = json.loads(m.group(0))
                if isinstance(obj, dict):
                    return obj
            except Exception:
                pass
        # Extract number and a short reason line
        num_match = re.search(r"\b([1-9])\b", text)
        reason_match = re.search(r"(?i)(because|since|due to|reason|rationale)[:\s-]+(.+)$", text)
        return {
            "answer": int(num_match.group(1)) if num_match else 5,
            "agent_reasoning": reason_match.group(2).strip() if reason_match else text.strip()[:200],
        }
