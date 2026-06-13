"""
GrobsAI Backend - Main Application Entry Point

This is the main FastAPI application with:
- FastAPI app initialization
- Middleware configuration (CORS, rate limiting, logging)
- Router registration
- Database setup
- Request logging
- Error handling

Architecture:
- API Layer: routers/
- Service Layer: services/
- AI Services: ai_services/
- Pipelines: pipelines/
- Workers: workers/
- Core: core/ (config, database, security, logging, exceptions)
"""
import os
import time
from contextlib import asynccontextmanager
from collections import defaultdict
from datetime import datetime, timedelta
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from typing import Dict, List

# Import from new core module
from app.core.config import settings
from app.database.session import engine, Base
import app.models  # Import all models to ensure they are registered with Base.metadata
from app.core.logging import get_logger, setup_logging
from app.core.exceptions import register_exception_handlers
from app.routers import (
    auth_router,
    resume_router,
    jobs_router,
    applications_router,
    users_router,
    interview_router,
    subscription_router,
    admin_router,
    calendar_router,
    websocket_router,
)
from app.routers.analytics_router import router as analytics_router
from app.routers.notifications_router import router as notifications_router
from app.routers.evaluation_router import router as evaluation_router

# Configure logging
logger = setup_logging()


# ==================== Rate Limiter ====================

class SimpleRateLimiter:
    """Simple in-memory rate limiter."""
    
    def __init__(self, requests_per_minute: int = 60):
        self.requests_per_minute = requests_per_minute
        self.requests: Dict[str, List[float]] = defaultdict(list)
    
    def is_allowed(self, client_ip: str) -> bool:
        """Check if request is allowed for client IP."""
        now = time.time()
        minute_ago = now - 60
        
        # Clean old requests
        self.requests[client_ip] = [
            req_time for req_time in self.requests[client_ip] 
            if req_time > minute_ago
        ]
        
        # Check if limit exceeded
        if len(self.requests[client_ip]) >= self.requests_per_minute:
            return False
        
        # Add current request
        self.requests[client_ip].append(now)
        return True


# Initialize rate limiter
# Increase limit for development to avoid blocking polling
rate_limiter = SimpleRateLimiter(requests_per_minute=settings.RATE_LIMIT_PER_MINUTE * 5)


# ==================== Lifespan ====================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    # Startup
    logger.info("Starting GrobsAI Backend...")
    
    # Validate LLM provider configuration
    from app.services.llm_service import llm_service
    llm_status = llm_service.validate_api_keys()
    
    if llm_status["is_valid"]:
        logger.info(f"LLM Provider: {llm_status['active_provider']} configured successfully")
    else:
        logger.warning("No LLM provider configured - using heuristic fallback mode")
        for warning in llm_status.get("warnings", []):
            logger.warning(warning)
    
    # Create database tables in development only
    if settings.ENVIRONMENT == "development":
        logger.info("Initializing database tables...")
        Base.metadata.create_all(bind=engine)
        logger.info("Database initialization complete")
    
    yield
    
    # Shutdown
    logger.info("Shutting down GrobsAI Backend...")


# ==================== App Initialization ====================

app = FastAPI(
    title=settings.APP_NAME,
    description="AI-powered career platform backend",
    version=settings.APP_VERSION,
    lifespan=lifespan
)


# ==================== Exception Handlers ====================

register_exception_handlers(app)


# ==================== Custom Middleware ====================

# Rate limit exception handler
@app.exception_handler(429)
async def rate_limit_handler(request: Request, exc: Exception):
    """Handle rate limit exceeded errors."""
    return JSONResponse(
        status_code=429,
        content={"detail": "Rate limit exceeded. Please try again later."}
    )


# Request logging and rate limiting middleware
@app.middleware("http")
async def log_requests_and_rate_limit(request: Request, call_next):
    """Log all HTTP requests and apply rate limiting."""
    # Handle CORS preflight requests - let CORS middleware handle them
    if request.method == "OPTIONS":
        response = await call_next(request)
        return response
    
    start_time = time.time()
    
    # Get client IP
    client_ip = request.client.host if request.client else "unknown"
    
    # Apply rate limiting
    if settings.ENVIRONMENT != "testing" and not rate_limiter.is_allowed(client_ip):
        logger.warning(f"Rate limit exceeded for IP: {client_ip}")
        return JSONResponse(
            status_code=429,
            content={"detail": "Rate limit exceeded. Please try again later."}
        )
    
    # Process request
    response = await call_next(request)
    
    # Calculate processing time
    process_time = time.time() - start_time
    
    # Log request details
    logger.info(
        f"{request.method} {request.url.path} - "
        f"Status: {response.status_code} - "
        f"Time: {process_time:.3f}s - "
        f"IP: {client_ip}"
    )
    
    # Add processing time header
    if hasattr(response, "headers"):
        response.headers["X-Process-Time"] = str(process_time)
    
    return response


# ==================== CORS Middleware (Must be outermost) ====================
# Move CORS middleware AFTER other middleware to ensure it wraps all responses (including errors from Rate Limiter).
# In FastAPI/Starlette, the LAST middleware added is the OUTERMOST.

# Parse CORS origins from settings
cors_origins = settings.CORS_ORIGINS
if isinstance(cors_origins, str):
    try:
        import json
        cors_origins = json.loads(cors_origins)
    except Exception:
        cors_origins = [o.strip() for o in cors_origins.split(",") if o.strip()]

# Ensure frontend origins are always allowed in development
if settings.ENVIRONMENT == "development":
    dev_origins = ["http://localhost:5173", "http://127.0.0.1:5173", "http://localhost:3000", "http://127.0.0.1:3000", "http://localhost:5174", "http://127.0.0.1:5174"]
    for origin in dev_origins:
        if origin not in cors_origins:
            cors_origins.append(origin)

logger.info(f"CORS enabled for origins: {cors_origins}")

app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==================== Routers ====================

app.include_router(auth_router.router)
app.include_router(resume_router.router)
app.include_router(jobs_router.router)
app.include_router(applications_router.router)
app.include_router(users_router.router)
app.include_router(interview_router.router)
app.include_router(subscription_router.router)
app.include_router(admin_router.router)
app.include_router(calendar_router.router)
app.include_router(websocket_router.router)
app.include_router(analytics_router)
app.include_router(notifications_router)
app.include_router(evaluation_router)

# ==================== Static Files (Uploads) ====================

# NOTE: Static mount of uploads directory is removed for security.
# Files should be served through protected endpoints (e.g., /api/resumes/{id}/preview)
# to ensure only authorized users can access their files.


# ==================== Health Check ====================

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    from app.services.llm_service import llm_service
    
    # Get LLM provider status
    llm_status = llm_service.validate_api_keys()
    
    return {
        "status": "healthy",
        "service": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "llm_provider": llm_status
    }


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": f"Welcome to {settings.APP_NAME} API",
        "docs": "/docs",
        "version": settings.APP_VERSION
    }


# ==================== Run ====================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app, 
        host=settings.HOST, 
        port=settings.PORT,
        log_level=settings.LOG_LEVEL.lower()
    )

