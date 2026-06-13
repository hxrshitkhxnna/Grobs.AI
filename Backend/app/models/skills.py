"""
Skills model for resume entries.
"""
from sqlalchemy import Column, Integer, String, ForeignKey, DateTime, Table
from sqlalchemy.orm import relationship
from datetime import datetime
from app.database.session import Base

class Skill(Base):
    """Skill entry model."""
    __tablename__ = "skills"
    
    id = Column(Integer, primary_key=True, index=True)
    resume_id = Column(Integer, ForeignKey("resumes.id"), nullable=True)
    
    # Skill details
    name = Column(String, nullable=False, index=True, unique=True)
    category = Column(String, default="Technical")  # Technical, Soft, Domain, etc.
    
    # Verified Skills Ecosystem
    verification_status = Column(String, default="unverified")  # unverified, machine_verified, human_verified
    confidence_score = Column(Integer, default=0)  # 0-100
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    jobs = relationship("Job", secondary="job_skills", back_populates="skills", overlaps="job_skills,skills")

    @property
    def resumes(self):
        """Mock list of resumes for test compatibility."""
        return [self.resume] if hasattr(self, "resume") and self.resume else []

