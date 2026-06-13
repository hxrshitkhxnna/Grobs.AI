"""
Enhanced ATS Analyzer — v4
==========================
Next-generation ATS scoring with ML, embeddings, and ensemble parsing.

Features:
- ML-based scoring with fallback to rule-based
- Semantic skill matching using embeddings
- Ensemble parsing with confidence scoring
- Feedback loop integration
- Performance optimization
- Backward compatibility
"""

from __future__ import annotations

import json
import logging
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from sqlalchemy.orm import Session

from app.core.config import settings
from app.models import Resume
from app.services.resume_service.embedding_service import get_embedding_service
from app.services.resume_service.parser import get_ensemble_parser, parse_resume_ensemble
from app.services.resume_service.feedback_service import get_feedback_service
from app.services.resume_service.ml_scorer import get_ml_scorer
from app.services.resume_service.performance_optimizer import PerformanceOptimizer
from app.services.resume_service.heuristic_ats_analyzer import _prepare_resume_text
from app.services.llm_service import llm_service

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Enhanced ATS Analyzer
# ─────────────────────────────────────────────────────────────────────────────

class EnhancedATSAnalyzer:
    """
    Enhanced ATS analyzer with ML, embeddings, and ensemble parsing.
    
    Features:
    - ML-based scoring with fallback
    - Semantic skill matching
    - Ensemble parsing with confidence
    - Feedback loop integration
    - Performance optimization
    """
    
    def __init__(self, db: Optional[Session] = None):
        """
        Initialize enhanced ATS analyzer.
        
        Args:
            db: Database session for feedback integration
        """
        from app.services.resume_service.resume_manager import ResumeManager
        self.db = db
        self.embedding_service = get_embedding_service()
        self.ensemble_parser = get_ensemble_parser()
        self.ml_scorer = get_ml_scorer(use_ml=True)
        self.feedback_service = get_feedback_service(db=db)
        self.resume_manager = ResumeManager(db) if db else None
        self.optimizer = PerformanceOptimizer(db)
    
    async def analyze_resume(
        self,
        resume: Resume,
        job_description: str = "",
        use_ml: bool = True,
        use_embeddings: bool = True,
        use_ensemble: bool = True,
    ) -> Dict[str, Any]:
        """
        Analyze resume with enhanced ATS scoring.
        """
        # Define cache key
        cache_key = f"ats_analysis_{resume.id}_{hash(job_description)}_{resume.updated_at.timestamp()}"
        
        async def _analyze():
            return await self._run_analysis_pipeline(
                resume, job_description, use_ml, use_embeddings, use_ensemble
            )
        
        # Use optimizer for caching
        return await self.optimizer.get_cached_result(
            cache_key, _analyze, ttl=settings.CACHE_TTL
        )

    async def _run_analysis_pipeline(
        self,
        resume: Resume,
        job_description: str = "",
        use_ml: bool = True,
        use_embeddings: bool = True,
        use_ensemble: bool = True,
    ) -> Dict[str, Any]:
        """Internal analysis pipeline."""
        start_time = time.perf_counter()
        
        # Step 1: Parse resume with ensemble approach
        resume_text = _prepare_resume_text(resume)
        parsing_result = self._parse_resume_enhanced(
            resume_text, use_ensemble=use_ensemble
        )
        
        # Step 2: Extract skills and perform semantic matching
        skill_analysis = self._analyze_skills_semantic(
            resume, job_description, use_embeddings=use_embeddings
        )
        
        # Step 3: Calculate ML-based ATS score
        ml_score = await self._calculate_ml_score(
            resume, job_description, parsing_result, skill_analysis, use_ml=use_ml
        )
        
        # Step 4: Generate recommendations
        recommendations = self._generate_recommendations(
            resume, job_description, parsing_result, skill_analysis, ml_score
        )
        
        # Step 5: Calculate confidence
        confidence = self._calculate_confidence(parsing_result, skill_analysis, ml_score)
        
        # Step 6: Generate feedback entry
        if self.feedback_service:
            self.feedback_service.record_feedback(
                resume_id=resume.id,
                user_id=resume.user_id,
                ats_score=ml_score,
                action="analyzed",
                metadata={
                    "job_description": job_description[:500] if job_description else None,
                    "parsing_method": parsing_result.get("parsing_method", "heuristic"),
                    "confidence": confidence,
                    "features": self._extract_features_for_feedback(resume, job_description),
                },
            )
        
        # Step 7: Update resume with enhanced analysis
        if self.resume_manager:
            await self.resume_manager.update_resume_analysis(
                resume.id,
                ml_score,
                parsing_result,
                skill_analysis,
                recommendations,
                confidence,
            )
        
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        
        return {
            "overall_score": ml_score,
            "confidence": confidence,
            "parsing_result": parsing_result,
            "skill_analysis": skill_analysis,
            "recommendations": recommendations,
            "analysis_time_ms": round(elapsed_ms, 2),
            "job_description": job_description,
            "analysis_timestamp": datetime.now().isoformat(),
            "components": {
                "semantic_skill_match": skill_analysis.get("semantic_match_rate", 0),
                "keyword_match": skill_analysis.get("keyword_match_rate", 0),
                "experience_score": self._calculate_experience_score(resume),
                "content_quality": self._calculate_content_quality(resume_text),
                "ats_parseability": self._calculate_ats_parseability(resume),
            },
        }
    
    def _parse_resume_enhanced(
        self,
        resume_text: str,
        use_ensemble: bool = True,
    ) -> Dict[str, Any]:
        """Parse resume using enhanced ensemble approach."""
        if use_ensemble:
            try:
                result = parse_resume_ensemble(resume_text, use_llm=True)
                return {
                    "method": "ensemble",
                    "confidence": result.get("confidence", 0.7),
                    "parsing_time_ms": result.get("parsing_time_ms", 0),
                    "data": result,
                }
            except Exception as e:
                logger.warning(f"Ensemble parsing failed: {e}. Falling back to heuristic.")
        
        # Fallback to original parser
        from .parser import parse_resume
        result = parse_resume(resume_text)
        return {
            "method": "heuristic",
            "confidence": 0.6,
            "parsing_time_ms": 0,
            "data": result,
        }
    
    def _analyze_skills_semantic(
        self,
        resume: Resume,
        job_description: str,
        use_embeddings: bool = True,
    ) -> Dict[str, Any]:
        """Analyze skills using semantic matching."""
        if not job_description or not resume.skills:
            return {
                "semantic_match_rate": 0.0,
                "keyword_match_rate": 0.0,
                "matched_skills": [],
                "missing_skills": [],
                "method": "fallback",
            }
        
        # Extract skill names
        resume_skills = [skill.name for skill in resume.skills]
        jd_skills = self._extract_skills_from_jd(job_description)
        
        if not jd_skills:
            return {
                "semantic_match_rate": 0.0,
                "keyword_match_rate": 0.0,
                "matched_skills": [],
                "missing_skills": [],
                "method": "no_jd_skills",
            }
        
        # Traditional matching
        traditional_result = self._match_skills_traditional(resume_skills, jd_skills)
        
        # Semantic matching
        semantic_result = {"matched": [], "missing": [], "match_rate": 0.0}
        if use_embeddings:
            try:
                semantic_result = self.embedding_service.match_skills_semantic(
                    resume_skills, jd_skills, threshold=0.75
                )
            except Exception as e:
                logger.warning(f"Semantic matching failed: {e}")
        
        # Combine results
        combined_matched = list(set(traditional_result["matched"] + semantic_result["matched"]))
        combined_missing = list(set(traditional_result["missing"] + semantic_result["missing"]))
        combined_match_rate = (len(combined_matched) / len(jd_skills)) * 100 if jd_skills else 0.0
        
        return {
            "semantic_match_rate": semantic_result["match_rate"],
            "keyword_match_rate": traditional_result["match_rate"],
            "combined_match_rate": combined_match_rate,
            "matched_skills": combined_matched,
            "missing_skills": combined_missing,
            "method": "semantic" if use_embeddings else "traditional",
            "resume_skills_count": len(resume_skills),
            "jd_skills_count": len(jd_skills),
        }
    
    async def _calculate_ml_score(
        self,
        resume: Resume,
        job_description: str,
        parsing_result: Dict[str, Any],
        skill_analysis: Dict[str, Any],
        use_ml: bool = True,
    ) -> int:
        """Calculate ML-based ATS score."""
        if not use_ml or not self.ml_scorer:
            return await self._calculate_rule_based_score(resume, job_description, skill_analysis)
        
        try:
            # Extract features for ML model
            features = self._extract_ml_features(resume, job_description, parsing_result, skill_analysis)
            
            # Get ML prediction
            ml_score = self.ml_scorer.predict_score(
                resume_data=parsing_result["data"],
                job_description=job_description,
                semantic_match_score=skill_analysis.get("semantic_match_rate", 0),
                keyword_match_rate=skill_analysis.get("keyword_match_rate", 0),
            )
            
            # Apply confidence adjustment
            confidence = parsing_result.get("confidence", 0.7)
            adjusted_score = int(ml_score * confidence + (100 * (1 - confidence)) * 0.5)
            
            return max(0, min(100, adjusted_score))
            
        except Exception as e:
            logger.error(f"ML scoring failed: {e}. Using rule-based fallback.")
            return await self._calculate_rule_based_score(resume, job_description, skill_analysis)
    
    async def _calculate_rule_based_score(
        self,
        resume: Resume,
        job_description: str,
        skill_analysis: Dict[str, Any],
    ) -> int:
        """Calculate rule-based ATS score (original logic)."""
        from .heuristic_ats_analyzer import calculate_ats_score
        
        # Use original ATS analyzer as fallback
        try:
            result = await calculate_ats_score(resume, job_description, provider="heuristic")
            return result.get("overall_score", 50)
        except Exception:
            # Ultimate fallback
            base_score = 50
            if resume.skills:
                base_score += min(30, len(resume.skills) * 2)
            if resume.experience:
                base_score += min(20, len(resume.experience) * 3)
            return max(0, min(100, base_score))
    
    def _generate_recommendations(
        self,
        resume: Resume,
        job_description: str,
        parsing_result: Dict[str, Any],
        skill_analysis: Dict[str, Any],
        ml_score: int,
    ) -> List[str]:
        """Generate personalized recommendations."""
        recommendations = []
        
        # Skill gap recommendations
        missing_skills = skill_analysis.get("missing_skills", [])
        if missing_skills:
            recommendations.append(f"Add skills: {', '.join(missing_skills[:3])}")
        
        # Content quality recommendations
        content_quality = self._calculate_content_quality(_prepare_resume_text(resume))
        if content_quality < 70:
            recommendations.append("Improve content quality with more quantifiable achievements")
        
        # Experience recommendations
        experience_score = self._calculate_experience_score(resume)
        if experience_score < 60:
            recommendations.append("Consider adding more detailed experience descriptions")
        
        # ATS parseability recommendations
        parseability = self._calculate_ats_parseability(resume)
        if parseability < 80:
            recommendations.append("Improve ATS parseability by adding missing contact information")
        
        # ML-specific recommendations
        if ml_score < 65:
            recommendations.append("Consider optimizing for ATS keywords and formatting")
        
        return recommendations[:5]  # Limit to 5 recommendations
    
    def _calculate_confidence(
        self,
        parsing_result: Dict[str, Any],
        skill_analysis: Dict[str, Any],
        ml_score: int,
    ) -> float:
        """Calculate overall confidence score."""
        parsing_confidence = parsing_result.get("confidence", 0.6)
        semantic_confidence = skill_analysis.get("semantic_match_rate", 0) / 100
        ml_confidence = abs(ml_score - 50) / 50  # Higher confidence for extreme scores
        
        # Weighted average
        confidence = (
            parsing_confidence * 0.4 +
            semantic_confidence * 0.3 +
            ml_confidence * 0.3
        )
        
        return max(0.0, min(1.0, confidence))
    
    def _extract_skills_from_jd(self, job_description: str) -> List[str]:
        """Extract skills from job description."""
        # Simple skill extraction - could be enhanced with ML
        skill_keywords = [
            "Python", "Java", "JavaScript", "TypeScript", "React", "Angular", "Vue",
            "Node.js", "Django", "Flask", "Spring", "AWS", "Azure", "GCP",
            "Docker", "Kubernetes", "Git", "SQL", "NoSQL", "MongoDB", "PostgreSQL",
            "Redis", "Elasticsearch", "Kafka", "RabbitMQ", "Linux", "Bash"
        ]
        
        found_skills = []
        jd_lower = job_description.lower()
        for skill in skill_keywords:
            if skill.lower() in jd_lower:
                found_skills.append(skill)
        
        return found_skills
    
    def _match_skills_traditional(
        self,
        resume_skills: List[str],
        jd_skills: List[str],
    ) -> Dict[str, Any]:
        """Traditional skill matching."""
        matched = []
        missing = []
        
        resume_lower = [s.lower() for s in resume_skills]
        
        for skill in jd_skills:
            if skill.lower() in resume_lower:
                matched.append(skill)
            else:
                missing.append(skill)
        
        match_rate = (len(matched) / len(jd_skills)) * 100 if jd_skills else 0.0
        
        return {
            "matched": matched,
            "missing": missing,
            "match_rate": match_rate,
        }
    
    def _extract_ml_features(
        self,
        resume: Resume,
        job_description: str,
        parsing_result: Dict[str, Any],
        skill_analysis: Dict[str, Any],
    ) -> List[float]:
        """Extract features for ML model."""
        # This would extract the same features as ML scorer
        # For now, return a placeholder
        return [50.0] * 15  # 15 features as defined in ML scorer
    
    def _calculate_experience_score(self, resume: Resume) -> int:
        """Calculate experience score."""
        if not resume.experience:
            return 30
        
        total_years = 0
        for exp in resume.experience:
            try:
                if exp.start_date and exp.end_date:
                    # Simple year calculation
                    start_year = int(exp.start_date[:4])
                    end_year = int(exp.end_date[:4])
                    total_years += max(0, end_year - start_year)
            except:
                pass
        
        # Logistic curve for experience scoring
        import math
        return int(100 / (1 + math.exp(-0.45 * (total_years - 5))))
    
    def _calculate_content_quality(self, resume_text: str) -> int:
        """Calculate content quality score."""
        # Count quantifiable metrics
        import re
        quantifiable_patterns = [
            r"\d+\s*%",
            r"\$\s*[\d,]+",
            r"\d+\s*x\b",
            r"increased\s+by\s+\d+",
            r"reduced\s+by\s+\d+",
        ]
        
        quantifiable_count = 0
        for pattern in quantifiable_patterns:
            quantifiable_count += len(re.findall(pattern, resume_text, re.IGNORECASE))
        
        # Count action verbs
        action_verbs = {
            "managed", "developed", "led", "created", "increased", "reduced",
            "spearheaded", "implemented", "designed", "achieved", "orchestrated",
        }
        action_count = sum(1 for verb in action_verbs if verb in resume_text.lower())
        
        # Calculate score
        quality_score = min(100, quantifiable_count * 10 + action_count * 2)
        return quality_score
    
    def _calculate_ats_parseability(self, resume: Resume) -> int:
        """Calculate ATS parseability score."""
        score = 100
        if not resume.email: score -= 25
        if not resume.phone: score -= 15
        if not resume.linkedin_url: score -= 10
        if not resume.education: score -= 15
        if not resume.experience: score -= 20
        if not resume.skills: score -= 15
        return max(0, score)
    
    def _extract_features_for_feedback(
        self,
        resume: Resume,
        job_description: str,
    ) -> Dict[str, Any]:
        """Extract features for feedback storage."""
        return {
            "resume_length": len(_prepare_resume_text(resume)),
            "skill_count": len(resume.skills),
            "experience_count": len(resume.experience),
            "education_count": len(resume.education),
            "jd_length": len(job_description),
        }
    
    def should_retrain_model(self) -> bool:
        """Check if model should be retrained based on feedback."""
        return self.feedback_service.should_retrain() if self.feedback_service else False
    
    def retrain_model(self) -> Dict[str, float]:
        """Retrain ML model with accumulated feedback."""
        if not self.feedback_service or not self.ml_scorer:
            return {"error": "Feedback service or ML scorer not available"}
        
        training_data = self.feedback_service.get_training_data()
        if not training_data:
            return {"error": "No training data available"}
        
        # Convert feedback data to ML training format
        ml_training_data = []
        for entry in training_data:
            features = self.ml_scorer.extract_features(
                entry.get("resume_data", {}),
                entry.get("job_description", ""),
                entry.get("semantic_match_score", 0),
                entry.get("keyword_match_rate", 0),
            )
            ml_training_data.append({
                "features": features,
                "label": entry["label"],
            })
        
        # Train model
        metrics = self.ml_scorer.train(ml_training_data, model_type="xgboost")
        return metrics


# ─────────────────────────────────────────────────────────────────────────────
# Backward compatibility wrapper
# ─────────────────────────────────────────────────────────────────────────────

async def calculate_enhanced_ats_score(
    resume: Resume,
    job_description: str = "",
    provider: Optional[str] = None,
    db: Optional[Session] = None,
) -> Dict[str, Any]:
    """
    Enhanced ATS scoring function with backward compatibility.
    
    Args:
        resume: Resume object
        job_description: Job description text
        provider: Scoring provider (heuristic, openai, anthropic, google)
        db: Database session
        
    Returns:
        Enhanced ATS analysis result
    """
    analyzer = EnhancedATSAnalyzer(db=db)
    
    # Determine which features to use based on provider
    use_ml = provider != "heuristic"
    use_embeddings = provider != "heuristic"
    use_ensemble = provider != "heuristic"
    
    result = await analyzer.analyze_resume(
        resume=resume,
        job_description=job_description,
        use_ml=use_ml,
        use_embeddings=use_embeddings,
        use_ensemble=use_ensemble,
    )
    
    # Determine if it was actually a fallback (either forced or due to error)
    parsing_data = result.get("parsing_result", {}).get("data", {})
    is_actually_fallback = (
        result.get("parsing_result", {}).get("method") == "heuristic" or
        parsing_data.get("parsing_method") == "heuristic"
    )
    
    # Convert to format compatible with original ATS analyzer
    return {
        "overall_score": result["overall_score"],
        "category_scores": result.get("components", {}),
        "skill_analysis": result["skill_analysis"],
        "keyword_gap": result["skill_analysis"].get("keyword_gap", {
            "matched": result["skill_analysis"].get("matched_skills", []),
            "missing": result["skill_analysis"].get("missing_skills", []),
            "optional": []
        }),
        "confidence": result["confidence"],
        "analysis_time_ms": result["analysis_time_ms"],
        "job_description": result["job_description"],
        "analysis_timestamp": result["analysis_timestamp"],
        "components": result["components"],
        "recommendations": result["recommendations"],
        "issues": result.get("issues", ["Advanced AI analysis currently unavailable"] if not use_ml or is_actually_fallback else []),
        "parsing_result": result["parsing_result"],
        "llm_powered": use_ml and not is_actually_fallback,
        "is_fallback": not use_ml or is_actually_fallback,
        "status": "Complete" if use_ml and not is_actually_fallback else "Partial",
    }  
# Alias for backward compatibility  
calculate_ats_score = calculate_enhanced_ats_score 


def _parse_jd_requirements(jd: str) -> Dict:
    from .heuristic_ats_analyzer import _parse_jd_requirements as _parse
    return _parse(jd)


def _split_jd_sections(jd: str) -> Tuple[str, str]:
    from .heuristic_ats_analyzer import _split_jd_sections as _split
    return _split(jd)


def _normalise_skill(skill: str) -> str:
    from .heuristic_ats_analyzer import _normalise_skill as _norm
    return _norm(skill)


def _match_skills(resume_skills: List[str], jd_skills: List[str]) -> Dict:
    from .heuristic_ats_analyzer import _match_skills as _match
    return _match(resume_skills, jd_skills)
