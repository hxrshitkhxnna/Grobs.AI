import sqlite3
import os

db_path = "grobs.db"
if not os.path.exists(db_path):
    print(f"Database {db_path} not found")
    exit(1)

conn = sqlite3.connect(db_path)
cursor = conn.cursor()

alterations = [
    # User fields
    ("ALTER TABLE users ADD COLUMN hireability_index FLOAT DEFAULT 0.0", "hireability_index in users"),
    ("ALTER TABLE users ADD COLUMN ghost_job_detected BOOLEAN DEFAULT 0", "ghost_job_detected in users"),
    
    # Skill fields
    ("ALTER TABLE skills ADD COLUMN verification_status VARCHAR DEFAULT 'unverified'", "verification_status in skills"),
    ("ALTER TABLE skills ADD COLUMN confidence_score INTEGER DEFAULT 0", "confidence_score in skills"),
    
    # Job Application fields
    ("ALTER TABLE job_applications ADD COLUMN success_probability FLOAT DEFAULT 0.0", "success_probability in job_applications"),
    ("ALTER TABLE job_applications ADD COLUMN match_breakdown JSON", "match_breakdown in job_applications"),
    
    # Job fields
    ("ALTER TABLE jobs ADD COLUMN is_ghost_job BOOLEAN DEFAULT 0", "is_ghost_job in jobs")
]

for sql, description in alterations:
    try:
        cursor.execute(sql)
        print(f"Successfully added {description}")
    except sqlite3.OperationalError as e:
        if "duplicate column name" in str(e).lower():
            print(f"Column {description} already exists")
        else:
            print(f"Error adding {description}: {e}")

conn.commit()
conn.close()
print("Database migration script completed.")
