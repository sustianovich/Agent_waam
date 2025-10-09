# `waam_pipeline.py` Execution Flow

This document summarizes the inputs, processing steps, and outputs that occur when `waam_pipeline.py` runs as a standalone script (`python waam_pipeline.py [AGENT_NAME]`).

---

## 1. Initialization
- **Environment variables**: Loads values from `.env`.
- **Defaults**:
  - `DEFAULT_AGENT = "WAAM Expert"` (used when no CLI argument is provided).
  - `MODEL_NAME = "llama3"` (Ollama model).
  - `_MAX_CHARS = 8000` (search result truncation limit).
- **Shared services**:
  - `llm = OllamaLLM(model=MODEL_NAME)`
  - `web_search_tool = DuckDuckGoSearchRun()`

---

## 2. Entry Point (`main(agent_name=DEFAULT_AGENT)`)
1. **Pick agent**: `agent_name` comes from `sys.argv[1]` or falls back to `DEFAULT_AGENT`.
2. **Load survey definition**: Reads `input_data/google_form/00_google_form.json`
   - Fails with a logged error if the file is missing.
   - Extracts the survey title and the list of comparisons using `load_comparisons(...)`.
3. **Prepare analysis label**: Uses `slugify_label(survey_title, agent_name)` for later reporting.

---

## 3. Question Processing Loop
For each comparison record (fields include `section`, `title`, `instructions`, `factor_1`, `factor_2`, `main_question`):

1. **Assemble instructions**:
   - Prefixes instructions with `Section: <title>` when a section title exists.
2. **Generate AI response**:

   a. **Call signature**
   ```python
   generate_ai_response(
       agent_name,
       comparison["factor_1"],
       comparison["factor_2"],
       comparison["main_question"],
       combined_instructions,
       question_index,
       use_memory=True,
   )
   ```

   b. **Steps inside `generate_ai_response`**
   - **Agent profile lookup**: Pulls metadata (description, style, priorities) from `AGENT_PROFILES`; falls back to `"WAAM Expert"`.
   - **Context enrichment**:
     1. `search(...)` builds a query from the main question, instructions, and both factors (via `build_explain_question`). It runs the DuckDuckGo tool (`DuckDuckGoSearchRun.invoke`) and converts the result to plain text using `ddg_to_text`. The text is trimmed to `_MAX_CHARS` and returned, or `""` if the query or lookup fails.
     2. If search text is available, `explain(...)` formats `PROMPT_TEMPLATE` with the agent profile, question, and search context, then calls `llm.invoke` (Ollama/`llama3`). The LLM response is stripped and treated as supplemental rationale. This rationale is appended to the instructions under “Supplementary context”.
   - **Agent construction**: Uses `get_agent(agent_name, profile, llm, use_memory=use_memory)` to obtain an evaluator capable of scoring the comparison.
   - **Memory reset**: Invokes `agent.reset_memory()` when `question_index == 0` or `use_memory` is `False`.
   - **Evaluation**: Calls `agent.evaluate(factor_1, factor_2, main_question, supplemental_instructions)` and returns the resulting dict (keys include `answer`, `agent_reasoning`, and any agent-specific fields).
   Returns a dict containing at least `answer` and `agent_reasoning`.
3. **Collect for AHP**: Appends a record with section/title/agent/comparison and the numeric answer.
4. **Console log**: Prints a line such as  
   `[Section] Qn: factor_1 vs factor_2 -> answer (agent_reasoning)`

---

## 4. `generate_ai_response(...)`
Inputs: `agent_name`, `factor_1`, `factor_2`, `main_question`, `instructions`, `question_index`, `use_memory`.

Steps:
1. **Select agent profile** from `AGENT_PROFILES` (defaults to `"WAAM Expert"` profile if missing).
2. **Context enrichment** (best effort):
   - `search(...)` builds a query from the question/instructions/factors, uses DuckDuckGo, trims to `_MAX_CHARS`.
   - If search text exists, `explain(...)` prompts the LLM with agent description, style, priorities, the question, and the search context to produce supplementary reasoning.
   - Supplemental text is appended to the instructions under “Supplementary context”.
3. **Instantiate agent** via `get_agent(agent_name, profile, llm, use_memory=use_memory)`.
4. **Memory handling**: Resets agent memory when `question_index == 0` or `use_memory` is `False`.
5. **Evaluation**: Calls `agent.evaluate(factor_1, factor_2, main_question, supplemental_instructions)` and returns its result.

---

## 5. Post-Processing
1. **AHP analysis**: `run_analysis_from_data(ahp_records, title=analysis_label)`
   - On success prints ranked factors, consistency index/ratio, and the saved plot path.
   - On error prints the returned error message.
2. **Script termination**: No explicit return; all outputs are console text plus the generated plot image.

---

## Inputs & Outputs Overview
| Stage | Inputs | Outputs |
|-------|--------|---------|
| Initialization | `.env`, agent profiles, config constants | LLM + search tool instances |
| Survey loading | Survey JSON (`00_google_form.json`) | `survey_title`, comparisons list |
| Generate response | Factors, main question, instructions, agent profile | Dict with `answer`, `agent_reasoning`, etc. |
| AHP accumulation | Response dicts | List of formatted records |
| Analysis | Records list | Console summary + saved plot (`analysis["plot_path"]`) |

---

### Command-Line Usage
```bash
python waam_pipeline.py            # uses DEFAULT_AGENT
python waam_pipeline.py "Agent X"  # runs with the specified agent profile
```

Outputs appear in the terminal and, if analysis succeeds, a plot image file is generated at the path reported in the console.
