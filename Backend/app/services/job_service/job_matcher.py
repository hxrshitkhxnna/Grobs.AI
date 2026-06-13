"""
Job Matcher Service - Vector-based scalable job matching.

Uses vector similarity search instead of looping through all jobs:
1. Generate embeddings for resumes after parsing
2. Generate embeddings for jobs during ingestion
3. Store embeddings using vector storage
4. Use cosine similarity for matching
"""
import json
import logging
from typing import List, Dict, Any, Optional
from sqlalchemy.orm import Session, joinedload, selectinload
import numpy as np

from app.models import Resume, ResumeContent, ResumeEmbedding, Job, JobSkill, JobEmbedding
from app.services.scoring_engine import scoring_engine
from app.services.hybrid_search import HybridSearchService
from app.services.prediction_engine import prediction_engine

logger = logging.getLogger(__name__)


class JobMatcher:
    """
    Scalable job matching using vector similarity search.
    """
    
    def __init__(self, db: Session):
        self.db = db
    
    async def match_resume_to_jobs(
        self,
        resume_id: int,
        user_id: int,
        limit: int = 10,
        min_score: float = 0.3
    ) -> List[Dict[str, Any]]:
        """
        Match a resume to jobs using Hybrid Search and XGBoost Success Prediction.
        """
        try:
            # 1. Get Resume Context
            resume = self.db.query(Resume).filter(
                Resume.id == resume_id,
                Resume.user_id == user_id
            ).first()
            
            if not resume:
                return []
                
            # 2. Execute Hybrid Search (BM25 + Vector)
            hybrid_search = HybridSearchService(self.db)
            query = f"{resume.title} {resume.summary}"
            search_results = await hybrid_search.search(query, limit=limit * 2)
            
            # 3. Apply Prediction Engine for Job Success Probability
            matches = []
            now = datetime.utcnow()
            for res in search_results:
                job_id = res["id"]
                job = self.db.query(Job).filter(Job.id == job_id).first()
                if not job:
                    continue
                    
                # Calculate job age
                days_old = 0
                if job.created_at:
                    days_old = (now - job.created_at).days
                    
                # Calculate Success Probability
                success_prob = prediction_engine.calculate_job_success_probability(
                    resume_data={"match_score": res["hybrid_score"]},
                    job_data={
                        "posted_days_ago": days_old,
                        "is_ghost_job": job.is_ghost_job
                    }
                )
                
                matches.append({
                    "job": job,
                    "match_score": int(res["hybrid_score"] * 100),
                    "success_probability": success_prob,
                    "hybrid_score": res["hybrid_score"]
                })
                
            # Sort by success probability and return top N
            matches.sort(key=lambda x: x["success_probability"], reverse=True)
            return matches[:limit]
            
        except Exception as e:
            logger.error(f"Error in Hybrid Match: {e}")
            return []
    
    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors."""
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return float(dot_product / (norm1 * norm2))
    
    def _keyword_match_resume_to_jobs(
        self,
        resume_id: int,
        user_id: int,
        limit: int,
        resume_skills: Optional[set] = None
    ) -> List[Dict[str, Any]]:
        """
        Fallback: Keyword-based job matching when embeddings unavailable.
        Optimized to avoid N+1 queries.
        """
        if resume_skills is None:
            resume = self.db.query(Resume).options(
                selectinload(Resume.skills)
            ).filter(
                Resume.id == resume_id,
                Resume.user_id == user_id
            ).first()
            
            if not resume:
                return []
            resume_skills = set(s.name.lower() for s in resume.skills)
        
        # Get all jobs with their skills in ONE query
        # EXCLUDE mock/sample data
        jobs = self.db.query(Job).options(
            selectinload(Job.skills)
        ).filter(Job.source != "Sample").all()
        
        matches = []
        for job in jobs:
            job_skill_names = set(s.skill_name.lower() for s in job.skills)
            
            if not job_skill_names and job.skills_required:
                try:
                    if isinstance(job.skills_required, str):
                        skills_list = json.loads(job.skills_required)
                    else:
                        skills_list = job.skills_required
                    
                    if skills_list:
                        job_skill_names = set(s.lower() for s in skills_list)
                except:
                    pass
            
            if not job_skill_names:
                continue
            
            # Calculate match score
            if resume_skills:
                overlap = len(resume_skills.intersection(job_skill_names))
                total = len(job_skill_names)
                score = overlap / total if total > 0 else 0
            else:
                score = 0
            
            if score >= 0.1:
                # Find missing keywords
                missing = list(job_skill_names - resume_skills)[:8]
                
                matches.append({
                    "job": job,
                    "match_score": int(score * 100),
                    "similarity": score,
                    "missing_keywords": missing
                })
        
        # Sort by score
        matches.sort(key=lambda x: x["match_score"], reverse=True)
        return matches[:limit]
    
    def _get_missing_keywords(self, resume_id: int, job_id: int) -> List[str]:
        """Get missing keywords between resume and job."""
        # Get resume skills
        resume = self.db.query(Resume).filter(Resume.id == resume_id).first()
        if not resume:
            return []
        
        resume_skills = set(s.name.lower() for s in resume.skills)
        
        # Get job skills
        job_skills = self.db.query(JobSkill).filter(
            JobSkill.job_id == job_id
        ).all()
        
        job_skill_names = set(s.skill_name.lower() for s in job_skills)
        
        # Find missing
        missing = list(job_skill_names - resume_skills)
        
        return missing[:8]
    
    async def generate_job_embedding(self, job_id: int) -> Dict[str, Any]:
        """
        Generate and store embedding for a job.
        
        Args:
            job_id: ID of the job
            
        Returns:
            Result dictionary
        """
        try:
            from app.services.llm_service import llm_service
            
            # Get job
            job = self.db.query(Job).filter(Job.id == job_id).first()
            if not job:
                return {"success": False, "error": "Job not found"}
            
            # Build job text
            job_text = self._build_job_text(job)
            
            # Generate embedding asynchronously
            embeddings = await llm_service.generate_embeddings_async(job_text)
            
            if not embeddings:
                return {"success": False, "error": "Failed to generate embedding"}
            
            embedding = embeddings[0]
            
            # Store or update embedding
            existing = self.db.query(JobEmbedding).filter(
                JobEmbedding.job_id == job_id
            ).first()
            
            if not existing:
                existing = JobEmbedding(job_id=job_id)
                self.db.add(existing)
            
            existing.embedding_vector = embedding.embedding
            existing.model_name = embedding.model
            
            self.db.commit()
            
            return {"success": True}
            
        except Exception as e:
            logger.error(f"Error generating job embedding: {e}")
            self.db.rollback()
            return {"success": False, "error": str(e)}
    
    def _build_job_text(self, job: Job) -> str:
        """Build text representation of job for embedding."""
        parts = [
            f"Job Title: {job.job_title}",
            f"Company: {job.company_name}",
            f"Location: {job.location or 'Not specified'}",
            f"Job Type: {job.job_type or 'Not specified'}",
            f"Experience Required: {job.experience_required or 'Not specified'}",
        ]
        
        if job.job_description:
            parts.append(f"Description: {job.job_description}")
        
        # Add skills
        skills = self.db.query(JobSkill).filter(JobSkill.job_id == job.id).all()
        if skills:
            skill_names = [s.skill_name for s in skills]
            parts.append(f"Skills: {', '.join(skill_names)}")
        
        return "\n".join(parts)


# Backward compatibility alias
class JobRecommender(JobMatcher):
    """Legacy name for backward compatibility."""
    pass

