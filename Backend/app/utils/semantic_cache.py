import logging
import json
import redis.asyncio as redis
from typing import Optional, Any
from app.core.config import settings
from app.integrations.vector_db import vector_db_service

logger = logging.getLogger(__name__)

class SemanticCache:
    """
    Semantic Cache for LLM responses.
    Uses Redis for exact matches and Vector DB for semantic similarity matches.
    """

    def __init__(self):
        # Redis connection (using settings from base worker config)
        self.redis = redis.from_url(settings.REDIS_URL or "redis://localhost:6379/1")
        self.similarity_threshold = 0.95

    async def get(self, prompt: str) -> Optional[str]:
        """
        Retrieve cached response for a given prompt.
        1. Try exact match in Redis
        2. Try semantic match in Vector DB
        """
        # Exact Match
        exact_cache = await self.redis.get(f"exact:{prompt}")
        if exact_cache:
            logger.info("SemanticCache: Exact match found in Redis")
            return exact_cache.decode("utf-8")
            
        # Semantic Match
        try:
            results = await vector_db_service.search_async(
                collection_name="semantic_cache",
                query=prompt,
                limit=1
            )
            
            if results and results[0].get("score", 0) > self.similarity_threshold:
                logger.info(f"SemanticCache: Semantic match found (Score: {results[0]['score']})")
                return results[0].get("metadata", {}).get("response")
        except Exception as e:
            logger.error(f"Semantic cache retrieval failed: {e}")
            
        return None

    async def set(self, prompt: str, response: str, ttl: int = 3600):
        """
        Store response in cache.
        1. Store exact match in Redis
        2. Store embedding + response in Vector DB
        """
        # Store in Redis (exact)
        await self.redis.set(f"exact:{prompt}", response, ex=ttl)
        
        # Store in Vector DB (semantic)
        try:
            await vector_db_service.add_document_async(
                collection_name="semantic_cache",
                document=prompt,
                metadata={"response": response},
                id=str(hash(prompt))
            )
        except Exception as e:
            logger.error(f"Semantic cache storage failed: {e}")

# Singleton instance
semantic_cache = SemanticCache()
