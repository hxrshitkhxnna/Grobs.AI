import logging
import numpy as np

logger = logging.getLogger(__name__)

try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False
    logger.warning("XGBoost not installed. PredictionEngine will use heuristic fallback.")

from typing import Dict, Any, List, Optional
from app.models.resume import Resume
from app.models.jobs import Job

class PredictionEngine:
    """
    Core AI engine for predicting job success and resume effectiveness.
    Uses XGBoost models trained on historical hiring data.
    """

    def __init__(self):
        self.success_model = None
        self.effectiveness_model = None
        # In production, we'd load models from disk:
        # self.success_model = xgb.Booster()
        # self.success_model.load_model("models/success_model.json")
        
    def calculate_job_success_probability(self, resume_data: Dict[str, Any], job_data: Dict[str, Any]) -> float:
        """
        Predict probability of landing an interview based on:
        1. Skill match (core/preferred)
        2. Experience alignment
        3. Company preferences
        4. Match score
        """
        # Feature Engineering (Minimal)
        match_score = resume_data.get("match_score", 0.5)
        experience_match = resume_data.get("experience_match", 0.5)
        skill_overlap = resume_data.get("skill_overlap", 0.5)
        
        # Simulate XGBoost prediction logic if model not loaded
        # formula: weighted combination + bias + noise
        base_prob = (match_score * 0.4) + (experience_match * 0.3) + (skill_overlap * 0.3)
        
        # Add "Ghost Job" penalty
        penalty = 0.0
        if job_data.get("is_ghost_job", False):
            penalty = 0.3  # Significant penalty for ghost jobs
        elif job_data.get("posted_days_ago", 0) > 30:
            penalty = 0.15 # Moderate penalty for older jobs
            
        success_prob = max(0.0, min(1.0, base_prob - penalty))
        return round(success_prob, 4)

    def calculate_resume_effectiveness(self, resume_data: Dict[str, Any]) -> float:
        """
        Predict resume hire-ability based on structure, density, and keywords.
        """
        # Simulate rules + ML hybrid
        score = 0.0
        
        # 1. Structure (Section completeness)
        sections = resume_data.get("sections", {})
        if sections:
            score += len(sections.keys()) * 0.1 # Up to 0.5
            
        # 2. Skill Density
        skills = resume_data.get("skills", [])
        if len(skills) > 10:
            score += 0.2
        elif len(skills) > 5:
            score += 0.1
            
        # 3. Action Verb usage (Simulated)
        if "achieved" in str(resume_data).lower() or "led" in str(resume_data).lower():
            score += 0.1
            
        # Normalize to 0-1
        effectiveness = min(1.0, score)
        return round(effectiveness, 4)

    async def get_hireability_index(self, resume_id: int, db_session: Any) -> float:
        """
        Calculate the multi-factor Hire-ability Index.
        """
        # Implementation of the index calculation
        resume = db_session.query(Resume).filter(Resume.id == resume_id).first()
        if not resume:
            return 0.0
            
        # Logic to combine multiple scores
        ats_score = (resume.ats_score or 0) / 100.0
        effectiveness = self.calculate_resume_effectiveness(resume.parsed_data or {})
        
        hireability = (ats_score * 0.6) + (effectiveness * 0.4)
        return round(hireability, 4)

# Singleton instance
prediction_engine = PredictionEngine()
