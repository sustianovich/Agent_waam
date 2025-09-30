import io
import re
import json
import logging
import os
import datetime
import sqlite3
from flask import Flask, render_template, request, jsonify, session, g, send_file
from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy import inspect
from langchain_ollama import OllamaLLM
import matplotlib.pyplot as plt
from datetime import UTC

from agent import AGENT_PROFILES
from decision_agent import WaaMAgent

from ahp_analysis import run_analysis_from_data

# Initialize Flask app
app = Flask(__name__)
app.secret_key = "supersecretkey"
SURVEY_FOLDER = "instance/survey_dbs"

# Ensure the survey_dbs folder exists
os.makedirs(SURVEY_FOLDER, exist_ok=True)

# Initialize LLaMA via Ollama
model_name = 'llama3'
llm = OllamaLLM(model=model_name)
# models gemma3:12b, llama3

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Define Base for SQLAlchemy ORM models
Base = declarative_base()

# Function to generate a unique database file name per survey run
def generate_db_name(agent, run_index, model_name, use_memory=True):
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    agent_clean = agent.replace(" ", "_")
    model_clean = model_name.split(":")[0].lower()[:10].replace(" ", "_")
    memory_flag = "memory_on" if use_memory else "memory_off"

    filename = f"{agent_clean}_{model_clean}_Run{run_index}_{memory_flag}_{timestamp}.db"
    return os.path.join(os.path.abspath(os.path.dirname(__file__)), SURVEY_FOLDER, filename)



def get_engine_and_session(db_path):
    engine = create_engine(f"sqlite:///{db_path}")
    session_factory = sessionmaker(bind=engine)
    return session_factory()

# Define Survey Model
class Survey(Base):
    __tablename__ = 'survey'
    id = Column(Integer, primary_key=True)
    agent = Column(String(50), nullable=False)
    run_number = Column(Integer, nullable=False)
    question_index = Column(Integer, nullable=False)
    section = Column(String(100), nullable=False)
    comparison = Column(String(200), nullable=False)
    answer = Column(Integer, nullable=False)
    agent_reasoning = Column(Text, nullable=False)
    timestamp = Column(DateTime, default=lambda: datetime.now(UTC))

# Load Google Form JSON
def load_form(filename="00_google_form.json"):
    try:
        with open(filename, "r", encoding="utf-8") as file:
            return json.load(file)
    except Exception as e:
        logging.error(f"❌ Error loading {filename}: {e}")
        return None
    
def normalize_name(text):
    return text.replace(" ", "_").replace("-", "_").lower()

def generate_ai_response(agent_name, factor_1, factor_2, main_question, instructions, question_index):
    profile = AGENT_PROFILES.get(agent_name, AGENT_PROFILES["WAAM Expert"])
    use_memory = session.get("use_memory", True)

    # Store or reuse agent instance
    if "agent_obj" not in g:
        g.agent_obj = WaaMAgent(agent_name, profile, llm, use_memory=use_memory)

    # Reset memory only at the beginning of a new run
    if question_index == 0:
        g.agent_obj.reset_memory()

    return g.agent_obj.evaluate(factor_1, factor_2, main_question, instructions)


# Function to get a new DB session for each survey run (Ensures new `.db` for each run)
def get_db_session(agent, run_index):
    # Generate a new DB path and store in session
    db_path = generate_db_name(agent, run_index, model_name)
    session["db_path"] = db_path
    session["current_run"] = run_index

    print(f"🔹 Creating new database for run {run_index}: {db_path}")

    # Close any previous session if it exists
    if "db_session" in g:
        g.db_session.close()

    # Create engine and check if 'survey' table exists
    engine = create_engine(f"sqlite:///{db_path}")
    inspector = inspect(engine)
    if 'survey' not in inspector.get_table_names():
        Base.metadata.create_all(engine)

    # Create a new session bound to the engine
    session_factory = sessionmaker(bind=engine)
    g.db_session = session_factory()
    return g.db_session




# Save survey response to the database
def save_survey(agent, run_number, question_index, section, factor_1, factor_2, answer, agent_reasoning):
    db_session = g.db_session  # Ensure this session exists in Flask context

    new_survey = Survey(
        agent=agent,
        run_number=run_number,
        question_index=question_index,
        section=section,
        comparison=f"{factor_1} vs {factor_2}",  # Fixed formatting
        answer=int(answer),  # Ensure it's saved as an integer
        agent_reasoning=agent_reasoning,  # Use passed variable directly
        timestamp=datetime.datetime.now(datetime.UTC)
    )

    db_session.add(new_survey)
    db_session.commit()


def fetch_agent_responses(agent):
    """Collects all responses for the selected agent from all .db files."""
    responses = []
    for db_file in os.listdir(SURVEY_FOLDER):
        if db_file.endswith(".db") and agent.replace(' ', '_').lower() in db_file:
            db_path = os.path.join(SURVEY_FOLDER, db_file)
            try:
                conn = sqlite3.connect(db_path)
                cursor = conn.cursor()
                cursor.execute("SELECT section, comparison, answer FROM survey")
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='survey'")
                if not cursor.fetchone():
                    print(f"⚠️ Skipping {db_file} — no 'survey' table found.")
                    conn.close()
                    continue
                rows = cursor.fetchall()
                conn.close()
                for row in rows:
                    responses.append({
                        "section": row[0],
                        "comparison": row[1],
                        "answer": row[2]
                    })
            except Exception as e:
                print(f"❌ Failed to read {db_file}: {e}")
    return responses

@app.route("/run_ahp", methods=["POST"])
def run_ahp():
    data = request.json
    mode = data.get("mode")
    agent = data.get("agent")
    survey_file = data.get("survey_file")

    if mode == "single" and survey_file:
        db_path = os.path.join(SURVEY_FOLDER, survey_file)
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='survey'")
        if not cursor.fetchone():
            conn.close()
            return jsonify({"error": f"Selected file does not contain a 'survey' table."})
        cursor.execute("SELECT section, comparison, answer FROM survey")
        rows = cursor.fetchall()
        conn.close()
        responses = [{"section": r[0], "comparison": r[1], "answer": r[2]} for r in rows]
        return jsonify(run_analysis_from_data(responses))

    elif mode in ["agent", "all"]:
        all_comparisons = []

        for db_file in os.listdir(SURVEY_FOLDER):
            if not db_file.endswith(".db"):
                continue

            # Filter by agent if needed
            if mode == "agent" and normalize_name(agent) not in normalize_name(db_file):
                continue
            
            db_path = os.path.join(SURVEY_FOLDER, db_file)
            try:
                conn = sqlite3.connect(db_path)
                cursor = conn.cursor()
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='survey'")
                if not cursor.fetchone():
                    conn.close()
                    print(f"⚠️ Skipping {db_file} — no 'survey' table.")
                    continue

                cursor.execute("SELECT section, comparison, answer FROM survey")
                rows = cursor.fetchall()
                conn.close()

                comparisons = [{"section": r[0], "comparison": r[1], "answer": r[2]} for r in rows]
                if comparisons:
                    all_comparisons.append(comparisons)

            except Exception as e:
                print(f"❌ Failed to read {db_file}: {e}")
                continue

        if not all_comparisons:
            return jsonify({"error": "No survey data found."})

        return jsonify(run_analysis_from_data(all_comparisons, is_multiple=True))

    else:
        return jsonify({"error": "Invalid request. Please provide proper mode and arguments."})



@app.route("/plot_ahp", methods=["POST"])
def plot_ahp():
    data = request.json
    factors = data.get("factors")
    priority_vector = data.get("priority_vector")

    if not factors or not priority_vector:
        return jsonify({"error": "Missing data for plot."}), 400

    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(factors, priority_vector, color="skyblue")
    ax.set_xlabel("Priority")
    ax.set_title("AHP Priority Vector")
    ax.invert_yaxis()

    # Send as image
    img = io.BytesIO()
    plt.tight_layout()
    plt.savefig(img, format="png")
    img.seek(0)
    plt.close()

    return send_file(img, mimetype="image/png")


# API Route to get next AI response
@app.route("/get_next_question", methods=["GET"])
def get_next_question():
    agent = request.args.get("agent", "WAAM Expert")
    num_runs = int(request.args.get("num_runs", 1))

    # ✅ Store memory toggle from frontend (default: true)
    use_memory = request.args.get("memory", "true").lower() == "true"
    session["use_memory"] = use_memory

    # Initialize session vars if not set
    if "current_run" not in session:
        session["current_run"] = 1
    if "question_index" not in session:
        session["question_index"] = 0
    if "db_paths" not in session:
        session["db_paths"] = []

    run_index = session["current_run"]
    question_index = session["question_index"]

    # ✅ Force new .db at the START of each run
    if question_index == 0:
        use_memory = session.get("use_memory", True)
        db_path = generate_db_name(agent, run_index, model_name, use_memory)
        session["db_path"] = db_path
        session["db_paths"].append(db_path)
        print(f"📁 Creating DB for run {run_index}: {db_path}")

        g.db_session = get_db_session(agent, run_index)  # only once
    else:
        db_path = session["db_path"]
        g.db_session = get_engine_and_session(db_path)



    form_data = load_form()
    if not form_data:
        return jsonify({"error": "Survey form data missing."})

    all_questions = []
    for section in form_data["sections"]:
        for comparison in section["comparisons"]:
            all_questions.append({
                "section": section["title"],
                "factor_1": comparison["factor_1"],
                "factor_2": comparison["factor_2"],
                "main_question": section.get("main_question", ""),
                "instructions": section.get("instructions", "")
            })


    # If all questions are answered
    if question_index >= len(all_questions):
        if run_index < num_runs:  # More runs are left
            print(f"🔄 Completing Run {run_index}, Starting Run {run_index + 1}")

            # Store previous run database path
            session["db_paths"].append(session["db_path"])

            # Move to the next run
            session["current_run"] = run_index + 1
            session["question_index"] = 0  # Reset question index for new run

            return get_next_question()  # Automatically start next run

        print(f"✅ Survey Completed: {num_runs} Runs")
        return jsonify({"completed": True, "message": f"Survey completed {num_runs} times!", "db_paths": session.get("db_paths", [])})

    # Get next question
    question = all_questions[question_index]
    session["question_index"] += 1  # Update session before returning response

    ai_response = generate_ai_response(
        agent, 
        question["factor_1"], 
        question["factor_2"], 
        question["main_question"], 
        question["instructions"],
        question_index
    )

    answer = int(ai_response["answer"])  # Ensure number

    save_survey(
    agent, run_index, question_index + 1,
    question["section"], question["factor_1"], question["factor_2"],  # FIXED
    ai_response["answer"], ai_response["agent_reasoning"]
    
    )
    

    return jsonify({
        "run_number": run_index,
        "question_index": question_index + 1,
        "total_questions": len(all_questions),
        "total_runs": num_runs,
        "section": question["section"],
        "comparison": f"{question['factor_1']} vs {question['factor_2']}",
        "answer": ai_response["answer"],
        "agent_reasoning": ai_response["agent_reasoning"],
        "db_path": session["db_path"],
        "db_paths": session.get("db_paths", [])
    })



# Cleanup database sessions after request
@app.teardown_appcontext
def close_db_session(exception=None):
    db_session = g.pop("db_session", None)
    if db_session:
        db_session.close()

# Route for home page
@app.route("/")
def index():
    session.clear()  # Reset session for a new survey
    agent_list = list(AGENT_PROFILES.keys())  # ['3D Designer', 'WAAM Expert', ...]
    return render_template("index.html", agents=agent_list)


if __name__ == "__main__":
    app.run(debug=True)
