"""
FastAPI Application

Main application entry point for the Janus3 Nervous System.
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routes import router
from api.dependencies import startup, shutdown


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan handler.

    Manages startup and shutdown of services.
    """
    # Startup
    await startup()
    yield
    # Shutdown
    await shutdown()


# Create FastAPI application
app = FastAPI(
    title="Janus3 Memory System",
    description="Production-ready conversational memory system with episodic memory and knowledge graph.",
    version="3.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes
app.include_router(router, prefix="/api/v1")


# ============================================
# Root and Health Endpoints
# ============================================

@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "name": "Janus3 Memory System",
        "version": "3.0.0",
        "status": "running",
        "docs_url": "/docs"
    }


@app.get("/health")
async def health():
    """
    Health check endpoint.

    Returns:
        Health status with component states.
    """
    from db.connection import get_db_pool
    from api.dependencies import get_redis

    health_status = {
        "status": "healthy",
        "components": {}
    }

    # Check database
    try:
        db_pool = await get_db_pool()
        async with db_pool.acquire() as conn:
            await conn.fetchval("SELECT 1")
        health_status["components"]["database"] = "healthy"
    except Exception as e:
        health_status["status"] = "degraded"
        health_status["components"]["database"] = f"unhealthy: {str(e)}"

    # Check Redis
    try:
        redis_client = await get_redis()
        await redis_client.ping()
        health_status["components"]["redis"] = "healthy"
    except Exception as e:
        health_status["status"] = "degraded"
        health_status["components"]["redis"] = f"unhealthy: {str(e)}"

    return health_status


# ============================================
# Run with uvicorn (for development)
# ============================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
