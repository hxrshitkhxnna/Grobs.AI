"""
Resume Pipeline Service - Implementation
=======================================
Orchestrates the resume processing flow:
Parsing -> Embedding -> ATS Analysis -> Job Matching
"""
from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

from sqlalchemy import func
from sqlalchemy.orm import Session

from app.models import (
    Education, Experience, Project, Resume, ResumeAnalysis,
    ResumeContent, ResumeEmbedding, ResumeVersion, Skill,
)
from app.utils.encryption import decrypt, encrypt
from app.services.resume_service.parser import EnsembleParser, extract_text_from_file
from app.services.resume_service.multi_stage_parser import multi_stage_parser
from app.services.prediction_engine import prediction_engine
from app.services.resume_service.ats_analyzer import calculate_ats_score as calculate_ats
from app.services.resume_service.performance_optimizer import PerformanceOptimizer
from app.services.resume_service.embedding_service import get_embedding_service
from app.services.job_service.job_matcher import JobMatcher
from app.core.config import settings

logger = logging.getLogger(__name__)

def _ok(stage: str, **extra) -> Dict[str, Any]:
    return {"success": True, "stage": stage, **extra}

def _err(stage: str, error: str, **extra) -> Dict[str, Any]:
    return {"success": False, "stage": stage, "error": error, **extra}

class ResumePipelineService:
    def __init__(self, db: Session) -> None:
        self.db = db
        self.ensemble_parser = EnsembleParser(use_llm=True)
        self.optimizer = PerformanceOptimizer(db)

    async def process_resume_upload(self, resume_id: int, file_path: str, user_id: int) -> Dict[str, Any]:
        """Runs the full processing pipeline for an uploaded resume."""
        import asyncio
        results: Dict[str, Any] = {
            "resume_id": resume_id,
            "stages_completed": [],
            "errors": [],
        }

        resume = self._get_resume(resume_id, user_id)
        if not resume:
            return _err("upload", f"Resume {resume_id} not found.")

        resume.pipeline_status = "uploading"
        resume.updated_at = datetime.utcnow()
        self.db.commit()

        try:
            # 1. Parsing with timeout and robust error handling
            logger.info(f"[Pipeline] Starting parsing for resume {resume_id}, file_path={file_path}")
            try:
                parse_result = await asyncio.wait_for(
                    self.parse_resume(resume_id, file_path, user_id), timeout=60
                )
            except asyncio.TimeoutError:
                logger.error(f"[Pipeline] Parsing timed out for resume {resume_id}, file_path={file_path}")
                resume.pipeline_status = "failed"
                resume.pipeline_error = "Parsing timed out."
                resume.updated_at = datetime.utcnow()
                self.db.commit()
                results["errors"].append("Parsing timed out.")
                results["success"] = False
                return results
            except Exception as exc:
                logger.error(f"[Pipeline] Parsing crashed for resume {resume_id}, file_path={file_path}: {exc}")
                resume.pipeline_status = "failed"
                resume.pipeline_error = f"Parsing crashed: {exc}"
                resume.updated_at = datetime.utcnow()
                self.db.commit()
                results["errors"].append(f"Parsing crashed: {exc}")
                results["success"] = False
                return results

            if parse_result.get("success"):
                results["stages_completed"].append("parsing")
            else:
                logger.error(f"[Pipeline] Parsing failed for resume {resume_id}: {parse_result.get('error')}")
                resume.pipeline_status = "failed"
                resume.pipeline_error = f"Parsing failed: {parse_result.get('error')}"
                resume.updated_at = datetime.utcnow()
                self.db.commit()
                results["errors"].append(f"Parsing failed: {parse_result.get('error')}")
                results["success"] = False
                return results

            # 2. Embeddings
            embed_result = self.generate_resume_embeddings(resume_id, user_id)
            if embed_result.get("success"):
                results["stages_completed"].append("embeddings")
            else:
                logger.error(f"[Pipeline] Embedding failed for resume {resume_id}: {embed_result.get('error')}")
                resume.pipeline_status = "failed"
                resume.pipeline_error = f"Embedding failed: {embed_result.get('error')}"
                resume.updated_at = datetime.utcnow()
                self.db.commit()
                results["errors"].append(f"Embedding failed: {embed_result.get('error')}")
                results["success"] = False
                return results

            # 3. Enhanced ATS Analysis (ML + Semantic)
            ats_result = await calculate_ats(resume, db=self.db)
            if ats_result and "overall_score" in ats_result:
                results["stages_completed"].append("ats_analysis")
                results["ats_score"] = ats_result.get("overall_score")
            else:
                logger.error(f"[Pipeline] ATS analysis failed for resume {resume_id}")
                results["errors"].append("ATS analysis failed to return scores")

            # 4. Job Matching
            match_result = await self.match_jobs(resume_id, user_id)
            if match_result.get("success"):
                results["stages_completed"].append("matched")
            else:
                logger.error(f"[Pipeline] Job matching failed for resume {resume_id}: {match_result.get('error')}")
                results["errors"].append(f"Job matching failed: {match_result.get('error')}")

            results["success"] = True
            resume.pipeline_status = "completed"
            resume.pipeline_error = None
            resume.updated_at = datetime.utcnow()
            self.db.commit()

        except Exception as exc:
            logger.exception("Pipeline error for resume %d", resume_id)
            results["errors"].append(str(exc))
            results["success"] = False
            resume.pipeline_status = "failed"
            resume.pipeline_error = str(exc)
            resume.updated_at = datetime.utcnow()
            self.db.commit()

        return results

    async def parse_resume(self, resume_id: int, file_path: str, user_id: int) -> Dict[str, Any]:
        """Parses the resume file and updates the database."""
        resume = self._get_resume(resume_id, user_id)
        if not resume:
            return _err("parsing", f"Resume {resume_id} not found.")

        try:
            resume.pipeline_status = "parsing"
            self.db.commit()

            full_path = self._resolve_file_path(file_path)
            if full_path is None or not os.path.exists(full_path):
                return _err("parsing", f"File not found: '{file_path}'")

            # 1. Extract raw text from file
            raw_text = extract_text_from_file(full_path)
            if not raw_text:
                return _err("parsing", "Could not extract text from file.")

            # 2. Use MultiStageParser for 95%+ accuracy (Extractor -> Verifier -> Standardizer)
            # Wrap with performance optimizer for caching and rate limiting
            async def _parse():
                return await multi_stage_parser.parse(full_path)
            
            parsed_data = await self.optimizer.execute_with_rate_limit(
                _parse, resource_name="multi_stage_parsing"
            )
            
            # Ensure raw_text is included in the parsed data
            parsed_data["raw_text"] = raw_text

            # Update resume fields
            resume.parsed_data = json.dumps(parsed_data)
            resume.full_name = encrypt(parsed_data.get("full_name", ""))
            resume.email = encrypt(parsed_data.get("email", ""))
            resume.phone = encrypt(parsed_data.get("phone", ""))
            resume.linkedin_url = encrypt(parsed_data.get("linkedin_url", ""))

            # Update ResumeContent with raw text
            if not resume.content:
                resume.content = ResumeContent(resume_id=resume.id)
            resume.content.raw_text = parsed_data.get("raw_text", "")
            # Ensure it's stored as a JSON-serializable dict if needed, 
            # though ResumeContent.parsed_json is Column(JSON)
            resume.content.parsed_json = parsed_data
            
            # Update nested data (Education, Experience, etc.)
            self._update_nested_data(resume, parsed_data)

            resume.pipeline_status = "parsed"
            self.db.commit()
            return _ok("parsing", parsed_data=parsed_data)

        except Exception as exc:
            logger.exception("Parsing error for resume %d", resume_id)
            resume.pipeline_status = "failed"
            resume.pipeline_error = f"Parsing error: {str(exc)}"
            self.db.commit()
            return _err("parsing", str(exc))

    def generate_resume_embeddings(self, resume_id: int, user_id: int) -> Dict[str, Any]:
        """Generates semantic embeddings for the resume."""
        resume = self._get_resume(resume_id, user_id)
        if not resume:
            return _err("embeddings", f"Resume {resume_id} not found.")

        try:
            resume.pipeline_status = "embedding"
            self.db.commit()

            parsed_data = {}
            if resume.parsed_data:
                parsed_data = json.loads(resume.parsed_data)

            # ✅ FIX: Always include raw_text fallback
            sections = []

            if parsed_data.get("raw_text"):
                sections.append(parsed_data["raw_text"])

            if parsed_data.get("summary"):
                sections.append(parsed_data["summary"])

            if parsed_data.get("experience"):
                for exp in parsed_data["experience"]:
                    sections.append(
                        f"{exp.get('role')} at {exp.get('company')}: {exp.get('description')}"
                    )

            if parsed_data.get("skills"):
                sections.append(
                    "Skills: " + ", ".join([
                        s.get("name") if isinstance(s, dict) else str(s)
                        for s in parsed_data["skills"]
                    ])
                )

            holistic_text = "\n".join(sections)

            # ❌ STOP EMPTY TEXT BUG
            if not holistic_text.strip():
                logger.error("[Embedding] Empty text")
                return _err("embeddings", "No content to embed")

            logger.info(f"[Embedding] Text length: {len(holistic_text)}")

            emb_service = get_embedding_service()
            vector = emb_service.get_embedding(holistic_text)

            if not vector:
                logger.error("[Embedding] Failed to generate vector")
                return _err("embeddings", "Embedding failed")

            # ✅ UPSERT embedding
            embedding_obj = self.db.query(ResumeEmbedding).filter(
                ResumeEmbedding.resume_id == resume_id
            ).first()

            if not embedding_obj:
                embedding_obj = ResumeEmbedding(resume_id=resume_id)
                self.db.add(embedding_obj)

            embedding_obj.embedding_vector = vector
            embedding_obj.model_name = emb_service.model_name
            embedding_obj.updated_at = datetime.utcnow()

            resume.pipeline_status = "embedded"
            self.db.commit()

            return _ok("embeddings")

        except Exception as exc:
            logger.exception("Embedding error for resume %d", resume_id)
            resume.pipeline_status = "failed"
            resume.pipeline_error = f"Embedding error: {str(exc)}"
            self.db.commit()
            return _err("embeddings", str(exc))

    async def run_ats_analysis(self, resume_id: int, user_id: int) -> Dict[str, Any]:
        """Runs ATS analysis and scores the resume."""
        resume = self._get_resume(resume_id, user_id)
        if not resume:
            return _err("analyzing", f"Resume {resume_id} not found.")

        try:
            resume.pipeline_status = "analyzing"
            self.db.commit()

            # Use calculate_ats_score (aliased as calculate_ats)
            ats_result = await calculate_ats(resume)
            
            if ats_result:
                resume.ats_score = ats_result.get("overall_score")
                resume.analysis_score = ats_result.get("overall_score")
                
                # Calculate Hire-ability Index (XGBoost + Heuristics)
                hireability = await prediction_engine.get_hireability_index(resume_id, self.db)
                resume.user.hireability_index = hireability
                
                # Save as ResumeAnalysis record
                analysis = ResumeAnalysis(
                    resume_id=resume_id,
                    analysis_type="ats",
                    score=ats_result.get("overall_score"),
                    feedback=json.dumps(ats_result.get("category_scores", {})),
                    missing_keywords=json.dumps(ats_result.get("issues", [])),
                    suggestions=json.dumps(ats_result.get("recommendations", [])),
                    created_at=datetime.utcnow()
                )
                self.db.add(analysis)
                
                resume.pipeline_status = "analyzed"
                self.db.commit()
                return _ok("analyzing", ats_score=resume.ats_score)
            else:
                return _err("analyzing", "ATS analysis returned no results.")

        except Exception as exc:
            logger.exception("ATS analysis error for resume %d", resume_id)
            resume.pipeline_status = "failed"
            resume.pipeline_error = f"Analysis error: {str(exc)}"
            self.db.commit()
            return _err("analyzing", str(exc))

    async def match_jobs(self, resume_id: int, user_id: int) -> Dict[str, Any]:
        """Matches the resume against available jobs."""
        resume = self._get_resume(resume_id, user_id)
        if not resume:
            return _err("matching", f"Resume {resume_id} not found.")

        try:
            resume.pipeline_status = "matching"
            self.db.commit()

            matcher = JobMatcher(self.db)
            # Run matching
            matches = await matcher.match_resume_to_jobs(resume_id, user_id, limit=5)
            
            resume.pipeline_status = "completed"
            self.db.commit()
            return _ok("matching", match_count=len(matches))

        except Exception as exc:
            logger.exception("Matching error for resume %d", resume_id)
            resume.pipeline_status = "failed"
            resume.pipeline_error = f"Matching error: {str(exc)}"
            self.db.commit()
            return _err("matching", str(exc))

    def get_pipeline_status(self, resume_id: int, user_id: int) -> Dict[str, Any]:
        """Returns the current status of the resume pipeline."""
        resume = self._get_resume(resume_id, user_id)
        if not resume:
            return {"error": f"Resume {resume_id} not found"}

        status_map = {
            "pending": 0,
            "uploading": 10,
            "parsing": 20,
            "parsed": 40,
            "embedding": 50,
            "embedded": 60,
            "analyzing": 70,
            "analyzed": 80,
            "matching": 90,
            "completed": 100,
            "failed": 0
        }

        current_status = resume.pipeline_status or "pending"
        progress = status_map.get(current_status, 0)

        return {
            "resume_id": resume_id,
            "current_status": current_status,
            "progress": progress,
            "error": resume.pipeline_error,
            "updated_at": resume.updated_at.isoformat() if resume.updated_at else None
        }

    def _get_resume(self, resume_id: int, user_id: int) -> Optional[Resume]:
        return self.db.query(Resume).filter(Resume.id == resume_id, Resume.user_id == user_id).first()

    def _resolve_file_path(self, file_path: str) -> Optional[str]:
        """Resolves relative file path to absolute path for local storage."""
        if not file_path:
            return None
            
        if os.path.isabs(file_path):
            return file_path if os.path.exists(file_path) else None
        
        # 1. Try relative to settings.upload_path (the standardized absolute path)
        path1 = os.path.join(settings.upload_path, file_path)
        if os.path.exists(path1):
            return path1
            
        # 2. Try direct relative to settings.UPLOAD_DIR (from CWD)
        path2 = os.path.join(settings.UPLOAD_DIR, file_path)
        if os.path.exists(path2):
            return os.path.abspath(path2)
            
        # 3. Try relative to project root
        path3 = os.path.join(settings.BASE_DIR, "uploads", file_path)
        if os.path.exists(path3):
            return path3
            
        logger.error(f"Could not resolve file path: {file_path}. Tried: {path1}, {path2}, {path3}")
        return None

    def _update_nested_data(self, resume: Resume, data: Dict[str, Any]):
        """Helper to update structured sections of the resume."""

        # Clear existing
        self.db.query(Education).filter(Education.resume_id == resume.id).delete()
        self.db.query(Experience).filter(Experience.resume_id == resume.id).delete()
        self.db.query(Project).filter(Project.resume_id == resume.id).delete()

        # Education
        for edu_data in data.get("education", []):
            edu = Education(
                resume_id=resume.id,
                school=edu_data.get("school", ""),
                degree=edu_data.get("degree", ""),
                major=edu_data.get("major"),
                gpa=edu_data.get("gpa"),
                start_date=edu_data.get("start_date") or edu_data.get("year", ""),
                end_date=edu_data.get("end_date") or edu_data.get("year", ""),
                description=edu_data.get("description")
            )
            self.db.add(edu)

        # Experience
        for exp_data in data.get("experience", []):
            exp = Experience(
                resume_id=resume.id,
                company=exp_data.get("company", ""),
                role=exp_data.get("role", ""),
                location=exp_data.get("location"),
                start_date=exp_data.get("start_date") or (
                    exp_data.get("duration", "").split(" - ")[0]
                    if exp_data.get("duration") else ""
                ),
                end_date=exp_data.get("end_date") or (
                    exp_data.get("duration", "").split(" - ")[1]
                    if exp_data.get("duration") and " - " in exp_data.get("duration") else None
                ),
                current=exp_data.get("current", False),
                description=exp_data.get("description") or exp_data.get("desc")
            )
            self.db.add(exp)

        # Projects
        for proj_data in data.get("projects", []):
            proj = Project(
                resume_id=resume.id,
                project_name=proj_data.get("project_name", ""),
                description=proj_data.get("description") or proj_data.get("desc"),
                project_url=proj_data.get("project_url"),
                github_url=proj_data.get("github_url"),
                technologies=proj_data.get("technologies"),
                points=proj_data.get("points", [])
            )
            self.db.add(proj)

        # ✅ Skills (FIXED PROPERLY)
        for skill_data in data.get("skills", []):
            name = skill_data.get("name") if isinstance(skill_data, dict) else str(skill_data)
            category = skill_data.get("category", "Technical") if isinstance(skill_data, dict) else "Technical"

            if not name:
                continue

            name = name.strip()

            # 🔥 IMPORTANT: Check globally
            existing_skill = self.db.query(Skill).filter(Skill.name == name).first()

            if existing_skill:
                continue

            new_skill = Skill(
                resume_id=resume.id,
                name=name,
                category=category
            )
            self.db.add(new_skill)

        # ✅ single commit
        self.db.commit()