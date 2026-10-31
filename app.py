import os
import json
from typing import Any, Dict, List, Optional

import streamlit as st
import re
import pandas as pd

def parse_plan_to_tables(plan: str) -> Dict[int, pd.DataFrame]:
    """Parse the markdown plan into a mapping of day number -> DataFrame.

    Assumes the plan contains sections like ``## Day X`` followed by a markdown
    table with three columns: Exercise, Sets, and Reps / Duration.
    """
    tables: Dict[int, pd.DataFrame] = {}
    lines = plan.splitlines()
    current_day: Optional[int] = None
    rows: List[List[str]] = []
    for line in lines:
        header_match = re.match(r"##\s*Day\s*(\d+)", line.strip())
        if header_match:
            # Save previous day table
            if current_day is not None and rows:
                df = pd.DataFrame(rows, columns=["Exercise", "Sets", "Reps / Duration"])
                df = df.iloc[1:]
                tables[current_day] = df
            # Start new day
            current_day = int(header_match.group(1))
            rows = []
            continue
        # Table rows start with '|' and second line is header separator
        if line.startswith("|") and not line.startswith("|---"):
            parts = [p.strip() for p in line.split("|") if p.strip()]
            if len(parts) >= 3:
                rows.append(parts[:3])
    # Capture last day
    if current_day is not None and rows:
        df = pd.DataFrame(rows, columns=["Exercise", "Sets", "Reps / Duration"])
        tables[current_day] = df.iloc[1:]
    return tables

import httpx

from dotenv import load_dotenv

# Load environment variables from .env (if present)
load_dotenv()

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
# Groq API endpoint and model – users should set their own GROQ_API_KEY env var.
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
DEFAULT_MODEL = os.getenv("GROQ_MODEL", "llama3-70b-8192")  # Model can be overridden via .env

# ---------------------------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------------------------

# Import default plan generator
from default_plans import get_default_plan

# ---------------------------------------------------------------------------

def extract_exercises_for_day(plan: str, day: int) -> List[str]:
    """Parse the markdown plan and return a list of exercise names for the given day.
    Assumes each day section starts with "## Day X" and contains a table where the first column
    is the exercise name.
    """
    lines = plan.splitlines()
    day_header = f"## Day {day}"
    try:
        start_idx = next(i for i, line in enumerate(lines) if line.strip().startswith(day_header))
    except StopIteration:
        return []
    # Find end of this day (next header or end of list)
    end_idx = len(lines)
    for i in range(start_idx + 1, len(lines)):
        if lines[i].strip().startswith("## Day "):
            end_idx = i
            break
    exercises = []
    # Look for table rows after header (skip the two header lines of the table)
    for line in lines[start_idx:end_idx]:
        if line.startswith("|") and not line.startswith("|---"):
            parts = line.split("|")
            if len(parts) >= 2:
                ex = parts[1].strip()
                if ex:
                    exercises.append(ex)
    return exercises[1:]

def get_api_key() -> str:
    """Retrieve Groq API key from environment.

    Returns
    -------
    str
        The API key.
    """
    key = os.getenv("GROQ_API_KEY")
    if not key:
        st.error("Groq API key not found. Please set the `GROQ_API_KEY` environment variable.")
    return key or ""


def build_prompt(data: Dict[str, Any]) -> str:
    """Create a textual prompt for the LLM based on user input.

    The LLM is asked to return a **markdown formatted weekly plan** where each day
    is a heading (e.g. `## Day 1`) followed by a table with columns:
    `Exercise | Sets | Reps / Duration`. The response should respect the user's
    selected equipment, any injury/limitations, and the number of training days.
    """
    prompt = (
        "You are a personal fitness trainer. Create a weekly workout plan based on the following specifications."
        " The output must be **markdown** with a separate section for each day (e.g. `## Day 1`)."
        " Under each day include a table with the columns `Exercise | Sets | Reps / Duration`."
        " Provide sensible sets/reps for strength exercises and duration for cardio/endurance."
        " Respect the user's equipment choices and any injuries or limitations."
        " Do NOT include any prose outside the tables.\n\n"
    )
    prompt += f"- Fitness goal: {data['goal']}\n"
    prompt += f"- Experience level: {data['experience']}\n"
    prompt += f"- Days per week: {data['days']}\n"
    prompt += f"- Equipment: {', '.join(data['equipment'])}\n"
    injuries = data.get("injuries")
    if injuries:
        prompt += f"- Injuries / limitations: {injuries}\n"
        prompt += "\nInclude a short disclaimer that the plan does not replace professional medical advice.\n"
    else:
        prompt += "\nNo specific injury or limitation noted.\n"
    return prompt


def fetch_alternative_exercise(old_ex: str, user_data: Dict[str, Any]) -> str:
    """Ask Groq for a single alternative exercise to replace *old_ex*.

    The prompt tells the model to respect the same constraints (goal, equipment,
    injuries) and to return **only the name** of a suitable replacement.
    """
    prompt = (
        "You are a personal fitness trainer. Suggest ONE alternative exercise that can replace"
        f" '{old_ex}' in a workout plan. Respect the user's fitness goal, equipment access"
        f" and any injuries/limitations. Return only the exercise name, no extra formatting."
    )
    # Include the same user context to help the model stay consistent
    context = (
        f"Goal: {user_data.get('goal')}, Equipment: {', '.join(user_data.get('equipment', []))},"
        f" Injuries: {user_data.get('injuries') or 'None'}"
    )
    messages = [
        {"role": "system", "content": "You are a helpful fitness assistant."},
        {"role": "user", "content": f"{prompt}\n{context}"},
    ]
    result = call_groq_api(messages)
    if not result:
        return ""
    # Return first line stripped – the model is asked to give just the name
    return result.strip().splitlines()[0].strip()

def call_groq_api(messages: List[Dict[str, str]]) -> Optional[str]:

    """Call Groq's chat completion endpoint.

    Parameters
    ----------
    messages:
        List of message dicts as required by the OpenAI‑compatible API.

    Returns
    -------
    Optional[str]
        The generated content or ``None`` on failure.
    """
    api_key = get_api_key()
    if not api_key:
        return None
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {
        "model": DEFAULT_MODEL,
        "messages": messages,
        "temperature": 0.7,
    }
    try:
        response = httpx.post(GROQ_API_URL, headers=headers, json=payload, timeout=30.0)
        response.raise_for_status()
        data = response.json()
        # The OpenAI compatible format returns choices[0].message.content
        content = data.get("choices", [{}])[0].get("message", {}).get("content")
        return content
    except httpx.HTTPStatusError as exc:
        st.error(f"API request failed: {exc.response.status_code} – {exc.response.text}")
    except httpx.RequestError as exc:
        st.error(f"Network error while contacting Groq API: {exc}")
    except Exception as exc:  # pragma: no cover – generic safety net
        st.error(f"Unexpected error during API call: {exc}")
    return None


def generate_plan(user_data: Dict[str, Any]) -> str:
    """Generate a workout plan using the LLM, with a fallback default plan.

    Parameters
    ----------
    user_data:
        Dictionary of sanitized user inputs.
    """
    prompt = build_prompt(user_data)
    messages = [
        {"role": "system", "content": "You are a helpful fitness assistant."},
        {"role": "user", "content": prompt},
    ]
    result = call_groq_api(messages)
    # If the API call fails or returns an empty/malformed response, use a built‑in default plan.
    if not result:
        st.warning("Groq API call failed or returned unexpected format – using a generic default workout plan.")
        return get_default_plan(user_data)
    return result.strip()


def swap_exercise(plan: str, day: int, old_ex: str, new_ex: str) -> str:
    """Replace a single exercise within the plan for the specified day.

    The plan uses markdown tables under a "## Day X" header. This function finds the
    table row whose first column matches ``old_ex`` (case‑insensitive) on the given
    day and substitutes the exercise name with ``new_ex``.
    """
    try:
        lines = plan.splitlines()
        day_header = f"## Day {day}"
        start_idx = next(i for i, line in enumerate(lines) if line.strip().startswith(day_header))
        # Locate the end of the day's section (next day header or end of file)
        end_idx = len(lines)
        for i in range(start_idx + 1, len(lines)):
            if lines[i].strip().startswith("## Day "):
                end_idx = i
                break
        # Find the first table row where the exercise matches old_ex
        for i in range(start_idx, end_idx):
            line = lines[i]
            if line.startswith("|") and not line.startswith("|---"):
                # Split keeping the leading/trailing pipes
                parts = line.split("|")
                # parts[1] is the exercise column (after the leading empty string)
                if len(parts) > 1:
                    current_ex = parts[1].strip()
                    if current_ex.lower() == old_ex.lower():
                        # Replace the exercise name while preserving surrounding cols
                        parts[1] = f" {new_ex} "
                        lines[i] = "|".join(parts)
                        break
        return "\n".join(lines)
    except Exception:
        # If anything goes wrong, fall back to the original plan.
        return plan


# ---------------------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------------------

def main() -> None:
    st.set_page_config(page_title="Workout Plan Generator", layout="centered")
    st.title("🏋️ Workout Plan Generator")

    # ------ Input Section ------
    with st.form(key="input_form"):
        goal = st.selectbox(
            "Fitness goal",
            ["Build muscle", "Lose fat", "General fitness", "Improve endurance"],
        )
        experience = st.selectbox(
            "Experience level",
            ["Beginner", "Intermediate", "Advanced"],
        )
        days = st.slider("Days per week", min_value=1, max_value=7, value=3, step=1)
        equipment = st.multiselect(
            "Equipment access",
            ["No equipment", "Home dumbbells", "Full gym"],
            default=["No equipment"],
        )
        injuries = st.text_area("Injuries / limitations (optional)")
        generate_btn = st.form_submit_button("Generate Plan")
        regenerate_btn = st.form_submit_button("Regenerate")

    # Initialize session state containers
    if "plan" not in st.session_state:
        st.session_state.plan = ""
    if "last_input" not in st.session_state:
        st.session_state.last_input = {}

    # ------ Validation ------
    def validate_inputs() -> bool:
        if days < 1:
            st.warning("Please select at least one day per week.")
            return False
        if not equipment:
            st.warning("Please select at least one equipment option.")
            return False
        return True

    # ------ Generate / Regenerate Logic ------
    if generate_btn or regenerate_btn:
        if not validate_inputs():
            st.stop()
        user_data = {
            "goal": goal,
            "experience": experience,
            "days": days,
            "equipment": equipment,
            "injuries": injuries.strip() or None,
        }
        # Store for potential regeneration
        st.session_state.last_input = user_data
        with st.spinner("Generating your personalized workout plan…"):
            plan = generate_plan(user_data)
        st.session_state.plan = plan

    # ------ Display Result ------
    if st.session_state.plan:
        st.subheader("Your Workout Plan")
        # Disclaimer (only show when injuries were provided)
        if st.session_state.last_input.get("injuries"):
            st.info(
                "*Disclaimer*: This plan does not replace professional medical advice. "
                "Consult a qualified healthcare provider before beginning any new exercise regimen."
            )
        # Parse the markdown plan into pandas tables per day
        tables = parse_plan_to_tables(st.session_state.plan)
        if tables:
            for day, df in tables.items():
                st.subheader(f"Day {day}")
                st.table(df)
        else:
            st.code(st.session_state.plan, language="markdown")

        # ------ Swap Exercise Mini‑Feature ------
        with st.expander("Swap an exercise", expanded=True):
            # Day selector
            day_num = st.number_input(
                "Day number",
                min_value=1,
                max_value=st.session_state.last_input.get("days", 7),
                step=1,
                key="swap_day",
            )

            # Populate exercise dropdown based on selected day (disabled until plan exists)
            exercise_options: List[str] = []
            if st.session_state.plan:
                exercise_options = extract_exercises_for_day(st.session_state.plan, int(day_num))
            old_ex = st.selectbox(
                "Exercise to replace",
                options=exercise_options,
                disabled=not exercise_options,
                key="old_ex_select",
            )

            # Fetch a new alternative exercise via Groq (only after old_ex chosen)
            new_ex = ""
            if old_ex:
                fetch_btn = st.button("Fetch new exercise")
                if fetch_btn:
                    new_ex = fetch_alternative_exercise(old_ex, st.session_state.last_input)
                    # Store in session state so it persists across reruns
                    st.session_state.fetched_new_ex = new_ex
                else:
                    new_ex = st.session_state.get("fetched_new_ex", "")
                # Show the fetched exercise (read‑only)
                st.text_input("New exercise", value=new_ex, key="new_ex_display")

                # Update plan button
                if new_ex:
                    if st.button("Update plan"):
                        updated = swap_exercise(st.session_state.plan, int(day_num), old_ex, new_ex)
                        st.session_state.plan = updated
                        # Clear fetched value for next swap
                        st.session_state.fetched_new_ex = ""                       
                        st.success("Exercise swapped and plan updated!")
                        st.rerun()
                        st.session_state.section_open = True


        # ------ Download Button ------
        def convert_to_bytes(txt: str) -> bytes:
            return txt.encode("utf-8")

        txt_bytes = convert_to_bytes(st.session_state.plan)
        st.download_button(
            label="Download plan as .txt",
            data=txt_bytes,
            file_name="workout_plan.txt",
            mime="text/plain",
        )
        st.download_button(
            label="Download plan as .md",
            data=txt_bytes,
            file_name="workout_plan.md",
            mime="text/markdown",
        )

    # ------ Footer ------
    st.caption("Powered by Groq LLM – ensure you have a valid `GROQ_API_KEY` set in your environment.")

if __name__ == "__main__":
    main()

