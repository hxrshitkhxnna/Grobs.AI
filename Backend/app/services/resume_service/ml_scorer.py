"""
ML-Based ATS Scorer — v4
=========================
Trainable machine learning model for ATS score prediction.

Features:
- XGBoost/Logistic Regression model with feature engineering
- Trainable on real user feedback data
- Fallback to rule-based scoring when model unavailable
- Feature importance analysis
- Model persistence and versioning
"""

from __future__ import annotations

import json
import logging
import os
import pickle
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler

from app.core.config import settings

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

MODEL_DIR = os.path.join(settings.CHROMA_PERSIST_DIR, "ml_models")
os.makedirs(MODEL_DIR, exist_ok=True)

MODEL_PATH = os.path.join(MODEL_DIR, "ats_scorer_v1.pkl")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler_v1.pkl")
FEATURE_CONFIG_PATH = os.path.join(MODEL_DIR, "feature_config.json")

# Feature definitions
FEATURE_NAMES = [
    "semantic_skill_match",      # Semantic similarity score (0-100)
    "keyword_match_rate",        # Keyword match rate (0-100)
    "years_of_experience",       # Years of experience (0-30)
    "project_count",             # Number of projects (0-20)
    "education_level",           # Education level (0-4 scale)
    "skill_breadth",             # Number of unique skills (0-50)
    "content_quality_score",     # Content quality metrics (0-100)
    "ats_parseability",          # ATS formatting score (0-100)
    "quantifiable_metrics",      # Count of quantifiable achievements (0-20)
    "action_verb_density",       # Action verb usage (0-100)
    "star_formula_usage",        # STAR formula usage (0-100)
    "contact_completeness",      # Contact info completeness (0-100)
    "section_completeness",      # Resume section completeness (0-100)
    "role_alignment",            # Alignment with target role (0-100)
    "seniority_match",           # Seniority level match (0-100)
]

# Education level encoding
EDUCATION_LEVELS = {
    "high school": 1,
    "diploma": 1,
    "associate": 2,
    "bachelor": 3,
    "b.tech": 3,
    "b.e.": 3,
    "b.sc": 3,
    "b.a.": 3,
    "bca": 3,
    "master": 4,
    "m.tech": 4,
    "m.e.": 4,
    "m.sc": 4,
    "m.a.": 4,
    "mca": 4,
    "mba": 4,
    "phd": 5,
    "ph.d.": 5,
    "doctorate": 5,
}


class MLATSscorer:
    """
    Machine learning-based ATS score predictor.
    Uses gradient boosting for accurate score prediction.
    """
    
    def __init__(self, use_ml: bool = True):
        """
        Initialize ML scorer.
        
        Args:
            use_ml: Whether to use ML model (fallback to rule-based if False)
        """
        self.use_ml = use_ml
        self.model = None
        self.scaler = None
        self.feature_config = {}
        self.is_trained = False
        
        # Try to load pre-trained model
        if use_ml:
            self._load_model()
    
    def _load_model(self):
        """Load pre-trained model from disk."""
        try:
            if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):
                with open(MODEL_PATH, 'rb') as f:
                    self.model = pickle.load(f)
                with open(SCALER_PATH, 'rb') as f:
                    self.scaler = pickle.load(f)
                
                if os.path.exists(FEATURE_CONFIG_PATH):
                    with open(FEATURE_CONFIG_PATH, 'r') as f:
                        self.feature_config = json.load(f)
                
                self.is_trained = True
                logger.info("ML ATS scorer model loaded successfully")
            else:
                logger.info("No pre-trained ML model found. Will use fallback or require training.")
                self._initialize_default_model()
        except Exception as e:
            logger.warning(f"Failed to load ML model: {e}. Using fallback.")
            self._initialize_default_model()
    
    def _initialize_default_model(self):
        """Initialize a default model for when no trained model exists."""
        # Use logistic regression as default (simpler, requires less data)
        self.model = LogisticRegression(
            max_iter=1000,
            class_weight='balanced',
            random_state=42
        )
        self.scaler = StandardScaler()
        self.is_trained = False
    
    def extract_features(
        self,
        resume_data: Dict[str, Any],
        job_description: str = "",
        semantic_match_score: float = 0.0,
        keyword_match_rate: float = 0.0,
    ) -> np.ndarray:
        """
        Extract numerical features from resume data for ML model.
        
        Args:
            resume_data: Parsed resume data dictionary
            job_description: Job description text (optional)
            semantic_match_score: Semantic skill match score (0-100)
            keyword_match_rate: Keyword match rate (0-100)
            
        Returns:
            Numpy array of features
        """
        # Education level
        education_level = 3  # Default bachelor's
        if resume_data.get("education"):
            max_edu = 0
            for edu in resume_data["education"]:
                degree = (edu.get("degree") or "").lower()
                for key, level in EDUCATION_LEVELS.items():
                    if key in degree:
                        max_edu = max(max_edu, level)
            education_level = max_edu if max_edu > 0 else 3
        
        # Years of experience
        years_exp = 0.0
        if resume_data.get("experience"):
            from datetime import datetime
            current_year = datetime.now().year
            for exp in resume_data["experience"]:
                start_date = exp.get("start_date", "")
                end_date = exp.get("end_date", "")
                try:
                    if start_date:
                        start_year = int(start_date[:4])
                        if end_date and end_date not in ("Present", "Current", "Now"):
                            end_year = int(end_date[:4])
                        else:
                            end_year = current_year
                        years_exp += max(0, end_year - start_year)
                except (ValueError, IndexError):
                    pass
        
        # Project count
        project_count = len(resume_data.get("projects", []))
        
        # Skill breadth
        skills = resume_data.get("skills", [])
        skill_breadth = len(set(s.get("name", "").lower() for s in skills))
        
        # Content quality metrics
        raw_text = resume_data.get("raw_text", "")
        quantifiable_count = self._count_quantifiable_metrics(raw_text)
        action_verb_density = self._calculate_action_verb_density(raw_text)
        star_formula_usage = self._calculate_star_formula_usage(raw_text)
        content_quality = min(100, (quantifiable_count * 5) + (action_verb_density * 0.5) + (star_formula_usage * 0.3))
        
        # ATS parseability
        ats_parseability = self._calculate_ats_parseability(resume_data)
        
        # Contact completeness
        contact_score = 0
        if resume_data.get("full_name"): contact_score += 25
        if resume_data.get("email"): contact_score += 25
        if resume_data.get("phone"): contact_score += 25
        if resume_data.get("linkedin_url"): contact_score += 25
        
        # Section completeness
        section_score = 0
        if resume_data.get("education"): section_score += 20
        if resume_data.get("experience"): section_score += 30
        if resume_data.get("skills"): section_score += 20
        if resume_data.get("projects"): section_score += 15
        if resume_data.get("summary"): section_score += 15
        
        # Role alignment (simplified)
        target_role = (resume_data.get("target_role") or "").lower()
        role_alignment = 70  # Default
        if target_role and job_description:
            jd_lower = job_description.lower()
            if target_role in jd_lower:
                role_alignment = 95
            elif any(word in jd_lower for word in target_role.split()):
                role_alignment = 80
        
        # Seniority match (simplified)
        seniority_match = 75  # Default
        if years_exp < 2:
            seniority_match = 60 if "senior" in (job_description or "").lower() else 85
        elif years_exp < 5:
            seniority_match = 75 if "mid" in (job_description or "").lower() else 75
        elif years_exp >= 5:
            seniority_match = 90 if "senior" in (job_description or "").lower() else 70
        
        # Build feature vector
        features = np.array([
            semantic_match_score,
            keyword_match_rate,
            min(years_exp, 30),
            min(project_count, 20),
            education_level,
            min(skill_breadth, 50),
            min(content_quality, 100),
            min(ats_parseability, 100),
            min(quantifiable_count, 20),
            min(action_verb_density, 100),
            min(star_formula_usage, 100),
            contact_score,
            section_score,
            role_alignment,
            seniority_match,
        ])
        
        return features
    
    def predict_score(self, resume_data: Dict[str, Any], job_description: str = "", **kwargs) -> int:
        """
        Predict ATS score for a resume.
        
        Args:
            resume_data: Parsed resume data
            job_description: Job description text
            **kwargs: Additional scoring parameters (semantic_match_score, keyword_match_rate)
            
        Returns:
            Predicted ATS score (0-100)
        """
        if not self.use_ml or not self.is_trained:
            return self._rule_based_score(resume_data, job_description, **kwargs)
        
        try:
            # Extract features
            features = self.extract_features(
                resume_data,
                job_description,
                kwargs.get("semantic_match_score", 50.0),
                kwargs.get("keyword_match_rate", 50.0),
            )
            
            # Scale features
            features_scaled = self.scaler.transform([features])
            
            # Predict probability (for binary classification) or score
            if hasattr(self.model, 'predict_proba'):
                # For classification models, get probability of positive class
                proba = self.model.predict_proba(features_scaled)[0]
                # Scale to 0-100
                score = int(proba[1] * 100) if len(proba) > 1 else int(proba[0] * 100)
            else:
                # For regression models, use direct prediction
                prediction = self.model.predict(features_scaled)[0]
                score = int(max(0, min(100, prediction)))
            
            return score
            
        except Exception as e:
            logger.error(f"ML prediction failed: {e}. Using rule-based fallback.")
            return self._rule_based_score(resume_data, job_description, **kwargs)
    
    def _rule_based_score(
        self,
        resume_data: Dict[str, Any],
        job_description: str = "",
        **kwargs
    ) -> int:
        """
        Fallback rule-based scoring when ML model unavailable.
        Uses weighted components similar to original system.
        """
        semantic_match = kwargs.get("semantic_match_score", 50.0)
        keyword_match = kwargs.get("keyword_match_rate", 50.0)
        
        # Education score
        education_level = 3
        if resume_data.get("education"):
            max_edu = 0
            for edu in resume_data["education"]:
                degree = (edu.get("degree") or "").lower()
                for key, level in EDUCATION_LEVELS.items():
                    if key in degree:
                        max_edu = max(max_edu, level)
            education_level = max_edu if max_edu > 0 else 3
        education_score = min(100, education_level * 20)
        
        # Experience score
        years_exp = 0.0
        if resume_data.get("experience"):
            from datetime import datetime
            current_year = datetime.now().year
            for exp in resume_data["experience"]:
                start_date = exp.get("start_date", "")
                end_date = exp.get("end_date", "")
                try:
                    if start_date:
                        start_year = int(start_date[:4])
                        if end_date and end_date not in ("Present", "Current", "Now"):
                            end_year = int(end_date[:4])
                        else:
                            end_year = current_year
                        years_exp += max(0, end_year - start_year)
                except (ValueError, IndexError):
                    pass
        
        import math
        # Higher base for students/juniors
        experience_score = 100 / (1 + math.exp(-0.35 * (years_exp - 3))) if years_exp > 0 else 50
        
        # Skills score
        skills = resume_data.get("skills", [])
        skill_breadth = len(set(s.get("name", "").lower() for s in skills))
        skills_score = min(100, skill_breadth * 6 + 20)
        
        # Content quality
        raw_text = resume_data.get("raw_text", "")
        quantifiable_count = self._count_quantifiable_metrics(raw_text)
        content_quality = min(100, quantifiable_count * 12 + 40)
        
        # Weighted combination (Balanced for juniors)
        score = (
            keyword_match * 0.25 +
            experience_score * 0.25 +
            skills_score * 0.20 +
            education_score * 0.10 +
            content_quality * 0.10 +
            semantic_match * 0.10
        )
        
        return int(max(0, min(100, score)))
    
    def _count_quantifiable_metrics(self, text: str) -> int:
        """Count quantifiable metrics in text."""
        import re
        patterns = [
            r"\d+\s*%",
            r"\$\s*[\d,]+",
            r"\d+\s*x\b",
            r"\d[\d,]*\s*(?:users|customers|clients|requests)",
            r"increased\s+by\s+\d+",
            r"reduced\s+by\s+\d+",
        ]
        count = 0
        for pattern in patterns:
            count += len(re.findall(pattern, text, re.IGNORECASE))
        return count
    
    def _calculate_action_verb_density(self, text: str) -> float:
        """Calculate action verb density."""
        action_verbs = {
            "managed", "developed", "led", "created", "increased", "reduced",
            "spearheaded", "implemented", "designed", "achieved", "orchestrated",
            "engineered", "facilitated", "mentored", "optimized", "streamlined",
            "pioneered", "generated", "maximized", "negotiated", "delivered",
            "launched", "automated", "refactored", "migrated", "integrated",
            "architected", "transformed", "drove", "scaled", "built", "deployed",
        }
        text_lower = text.lower()
        found = sum(1 for verb in action_verbs if verb in text_lower)
        return min(100, found * 5)
    
    def _calculate_star_formula_usage(self, text: str) -> float:
        """Calculate STAR formula usage."""
        import re
        star_pattern = re.compile(
            r"(?:(?:achiev|deliver|result|impact|increas|decreas|reduc|improv|sav|earn|"
            r"generated?|drove?|grew?|scaled?)[a-z]*)\s+[^.]{5,50}"
            r"(?:\s+(?:by|to|from|of|in)\s+[\d%$\[\]X]+)?",
            re.IGNORECASE,
        )
        hits = len(star_pattern.findall(text))
        return min(100, hits * 10)
    
    def _calculate_ats_parseability(self, resume_data: Dict[str, Any]) -> float:
        """Calculate ATS parseability score."""
        score = 100
        if not resume_data.get("email"): score -= 25
        if not resume_data.get("phone"): score -= 15
        if not resume_data.get("linkedin_url"): score -= 10
        if not resume_data.get("education"): score -= 15
        if not resume_data.get("experience"): score -= 20
        if not resume_data.get("skills"): score -= 15
        return max(0, score)
    
    def train(
        self,
        training_data: List[Dict[str, Any]],
        test_size: float = 0.2,
        model_type: str = "xgboost"
    ) -> Dict[str, float]:
        """
        Train the ML model on provided data.
        
        Args:
            training_data: List of dictionaries with 'features' and 'label' keys
            test_size: Fraction of data for testing
            model_type: Type of model to train ('xgboost' or 'logistic')
            
        Returns:
            Dictionary with training metrics
        """
        if not training_data:
            logger.error("No training data provided")
            return {"error": "No training data"}
        
        try:
            # Extract features and labels
            X = np.array([item["features"] for item in training_data])
            y = np.array([item["label"] for item in training_data])
            
            # Convert to binary classification (shortlist or not)
            # Threshold at 65 (typical ATS cutoff)
            y_binary = (y >= 65).astype(int)
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y_binary, test_size=test_size, random_state=42, stratify=y_binary
            )
            
            # Scale features
            self.scaler = StandardScaler()
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            # Train model
            if model_type == "xgboost":
                try:
                    from xgboost import XGBClassifier
                    self.model = XGBClassifier(
                        n_estimators=100,
                        max_depth=6,
                        learning_rate=0.1,
                        random_state=42,
                        use_label_encoder=False,
                        eval_metric='logloss'
                    )
                except ImportError:
                    logger.warning("XGBoost not available, using Logistic Regression")
                    self.model = LogisticRegression(
                        max_iter=1000,
                        class_weight='balanced',
                        random_state=42
                    )
            else:
                self.model = LogisticRegression(
                    max_iter=1000,
                    class_weight='balanced',
                    random_state=42
                )
            
            self.model.fit(X_train_scaled, y_train)
            
            # Evaluate
            y_pred = self.model.predict(X_test_scaled)
            metrics = {
                "accuracy": accuracy_score(y_test, y_pred),
                "precision": precision_score(y_test, y_pred, zero_division=0),
                "recall": recall_score(y_test, y_pred, zero_division=0),
                "f1": f1_score(y_test, y_pred, zero_division=0),
            }
            
            # Cross-validation
            cv_scores = cross_val_score(self.model, X_train_scaled, y_train, cv=5)
            metrics["cv_mean"] = cv_scores.mean()
            metrics["cv_std"] = cv_scores.std()
            
            # Save model
            self._save_model()
            self.is_trained = True
            
            logger.info(f"ML model trained successfully. Accuracy: {metrics['accuracy']:.2%}")
            return metrics
            
        except Exception as e:
            logger.error(f"Model training failed: {e}")
            return {"error": str(e)}
    
    def _save_model(self):
        """Save trained model to disk."""
        try:
            with open(MODEL_PATH, 'wb') as f:
                pickle.dump(self.model, f)
            with open(SCALER_PATH, 'wb') as f:
                pickle.dump(self.scaler, f)
            with open(FEATURE_CONFIG_PATH, 'w') as f:
                json.dump({
                    "feature_names": FEATURE_NAMES,
                    "trained_at": datetime.now().isoformat(),
                    "model_type": type(self.model).__name__,
                }, f, indent=2)
            logger.info("ML model saved successfully")
        except Exception as e:
            logger.error(f"Failed to save model: {e}")
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance from trained model."""
        if not self.is_trained or not hasattr(self.model, 'feature_importances_'):
            return {}
        
        importances = self.model.feature_importances_
        return {
            name: float(importance)
            for name, importance in zip(FEATURE_NAMES, importances)
        }


# ─────────────────────────────────────────────────────────────────────────────
# Singleton instance
# ─────────────────────────────────────────────────────────────────────────────

_ml_scorer = None


def get_ml_scorer(use_ml: bool = True) -> MLATSscorer:
    """Get or create the singleton ML scorer instance."""
    global _ml_scorer
    if _ml_scorer is None:
        _ml_scorer = MLATSscorer(use_ml=use_ml)
    return _ml_scorer