"""
FastAPI Application - Janus4 Memory System
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from janus_app.routes import router
from janus_app.dependencies import startup, shutdown
from janus_app.infra.tracing import configure_tracing


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    configure_tracing()
    await startup()
    yield
    await shutdown()


app = FastAPI(
    title="Janus Memory System",
    description="Conversational memory system with episodic memory and knowledge graph.",
    version="4.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, prefix="/api/v1")


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "name": "Janus Memory System",
        "version": "4.0.0",
        "status": "running",
        "docs_url": "/docs"
    }


@app.get("/health")
async def health():
    """Health check endpoint. Returns 503 if any component is unhealthy."""
    from janus_core.db.connection import get_db_pool
    from janus_app.dependencies import get_redis

    status = "healthy"
    components = {}

    try:
        db_pool = await get_db_pool()
        async with db_pool.acquire() as conn:
            await conn.fetchval("SELECT 1")
        components["database"] = "healthy"
    except Exception as e:
        status = "unhealthy"
        components["database"] = f"error: {e}"

    try:
        redis_client = await get_redis()
        await redis_client.ping()
        components["redis"] = "healthy"
    except Exception as e:
        status = "unhealthy"
        components["redis"] = f"error: {e}"

    response = {"status": status, "components": components}
    status_code = 200 if status == "healthy" else 503
    return JSONResponse(content=response, status_code=status_code)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("janus_app.main:app", host="0.0.0.0", port=8000, reload=True)
