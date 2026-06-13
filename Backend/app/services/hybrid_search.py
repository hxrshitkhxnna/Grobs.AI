import logging
from typing import List, Dict, Any, Optional
import numpy as np
from sqlalchemy.orm import Session
from sqlalchemy import text
from app.models.jobs import Job
from app.integrations.vector_db import vector_db_service
from app.services.llm_service import LLMService

logger = logging.getLogger(__name__)

class HybridSearchService:
    """
    Hybrid Search Engine combining BM25 (via Postgres Full-Text Search) 
    and Vector Similarity (via ChromaDB) using Reciprocal Rank Fusion (RRF).
    """

    def __init__(self, db: Session):
        self.db = db
        self.llm_service = LLMService()

    async def search(self, query: str, limit: int = 20, k: int = 60) -> List[Dict[str, Any]]:
        """
        Execute hybrid search using RRF.
        score = sum(1 / (k + rank))
        """
        # 1. Get Vector Search Results
        vector_results = await self._get_vector_results(query, limit * 2)
        
        # 2. Get BM25 Results (Postgres FTS)
        bm25_results = self._get_bm25_results(query, limit * 2)
        
        # 3. Apply Reciprocal Rank Fusion
        scores = {}
        
        # Process vector results
        for rank, res in enumerate(vector_results):
            job_id = res["id"]
            scores[job_id] = scores.get(job_id, 0) + 1 / (k + rank + 1)
            
        # Process BM25 results
        for rank, res in enumerate(bm25_results):
            job_id = res["id"]
            scores[job_id] = scores.get(job_id, 0) + 1 / (k + rank + 1)
            
        # 4. Sort and get top jobs
        sorted_job_ids = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:limit]
        
        # 5. Hydrate jobs from DB
        final_jobs = []
        for job_id, score in sorted_job_ids:
            job = self.db.query(Job).filter(Job.id == job_id).first()
            if job:
                job_dict = self._to_dict(job)
                job_dict["hybrid_score"] = score
                final_jobs.append(job_dict)
                
        return final_jobs

    async def _get_vector_results(self, query: str, limit: int) -> List[Dict[str, Any]]:
        """Get semantic search results from ChromaDB."""
        try:
            results = await vector_db_service.search_jobs_async(query, limit=limit)
            return [{"id": int(res["id"]), "score": res.get("score", 0)} for res in results]
        except Exception as e:
            logger.error(f"Vector search failed: {e}")
            return []

    def _get_bm25_results(self, query: str, limit: int) -> List[Dict[str, Any]]:
        """Get keyword search results from Postgres using Full-Text Search."""
        try:
            # Simple keyword matching for now, can be upgraded to tsvector
            sql = text("""
                SELECT id, ts_rank_cd(to_tsvector('english', title || ' ' || description), plainto_tsquery('english', :query)) AS rank
                FROM jobs
                WHERE to_tsvector('english', title || ' ' || description) @@ plainto_tsquery('english', :query)
                ORDER BY rank DESC
                LIMIT :limit
            """)
            results = self.db.execute(sql, {"query": query, "limit": limit}).fetchall()
            return [{"id": res.id, "score": res.rank} for res in results]
        except Exception as e:
            logger.error(f"BM25 search failed: {e}")
            return []

    def _to_dict(self, job: Job) -> Dict[str, Any]:
        return {
            "id": job.id,
            "title": job.title,
            "company": job.company,
            "location": job.location,
            "match_score": job.match_score,
            "job_type": job.job_type,
            "description": job.description,
            "is_ghost_job": job.is_ghost_job,
            "created_at": job.created_at
        }
