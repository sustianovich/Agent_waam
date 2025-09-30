import sqlite3
import pandas as pd

# === Configuration ===
db_path = "instance/survey_dbs/3D_Designer_llama3_Run1_memory_on_20250522_092205.db"
output_md = "survey_data_output.md"

# === Load data ===
try:
    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query("SELECT * FROM survey", conn)
    conn.close()
except Exception as e:
    print(f"Failed to read database: {e}")
    exit()

# === Write to Markdown ===
with open(output_md, "w", encoding="utf-8") as f:
    f.write("# Survey Data Output\n\n")
    f.write(f"## Database: {db_path}\n\n")

    if df.empty:
        f.write("_No entries found._\n")
    else:
        for i, row in df.iterrows():
            f.write(f"### Entry {i+1}\n\n")
            for col in df.columns:
                if col == "agent_reasoning":
                    f.write(f"**{col.replace('_', ' ').title()}:**\n> {row[col]}\n\n")
                else:
                    f.write(f"- **{col.replace('_', ' ').title()}**: {row[col]}\n")
            f.write("\n---\n\n")
