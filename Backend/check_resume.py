
from app.database.session import SessionLocal
from app.models import Resume, Experience, Skill, Education, Project
import json

def check():
    db = SessionLocal()
    try:
        r = db.query(Resume).filter(Resume.id == 2).first()
        if not r:
            print("Resume 2 not found")
            return
            
        print(f"ID: {r.id}")
        print(f"Title: {r.title}")
        print(f"Target Role: {r.target_role}")
        print(f"ATS Score: {r.ats_score}")
        print(f"Experience: {len(r.experience)} items")
        for e in r.experience:
            print(f"  - {e.role} at {e.company} ({e.start_date} - {e.end_date})")
        print(f"Skills: {len(r.skills)} items")
        for s in r.skills:
            print(f"  - {s.name}")
        print(f"Education: {len(r.education)} items")
        print(f"Projects: {len(r.projects)} items")
        
        # Try re-parsing
        print("\n--- RE-PARSING ---")
        from app.services.resume_service.parser import parse_resume
        import os
        from app.core.config import settings
        
        full_path = r.file_path
        if not os.path.isabs(full_path):
            full_path = os.path.join(settings.UPLOAD_DIR, full_path)
            
        print(f"File path: {full_path}")
        if os.path.exists(full_path):
            from app.services.resume_service.resume_manager import ResumeManager
            manager = ResumeManager(db)
            print("Running manager.parse_resume_file...")
            import asyncio
            asyncio.run(manager.parse_resume_file(r.id, r.user_id))
            
            # Now trigger ATS check
            print("Running manager.get_ats_score...")
            ats_result = asyncio.run(manager.get_ats_score(r.id, r.user_id))
            print(f"New Final ATS Score: {ats_result.get('overall_score')}")
            
            db.refresh(r)
            print(f"Resume Updated Score in DB: {r.ats_score}")
        else:
            print("File not found on disk, cannot re-parse")
        if r.content:
            print(f"Raw Text length: {len(r.content.raw_text) if r.content.raw_text else 0}")
            if r.content.raw_text:
                print("--- RAW TEXT START ---")
                print(r.content.raw_text)
                print("--- RAW TEXT END ---")
        else:
            print("No content record")
        if r.parsed_data:
            try:
                pd = json.loads(r.parsed_data)
                print(f"Parsed Data Summary: {pd.get('summary', 'No summary')}")
            except:
                print("Parsed data is not valid JSON")
        else:
            print("No parsed data")
            
    finally:
        db.close()

if __name__ == "__main__":
    check()
