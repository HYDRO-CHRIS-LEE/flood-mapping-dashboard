"""
EarthAI FastAPI application.
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from backend.config import FRONTEND_DIR
from backend.db import init_db
from backend.routers.events import router as events_router
from backend.routers.optical import router as optical_router
from backend.routers.rainfall import router as rainfall_router
from backend.routers.sar import router as sar_router
from backend.routers.classifier import router as classifier_router
from backend.routers.leaderboard import router as leaderboard_router
from backend.routers.flappy import router as flappy_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup / shutdown lifecycle."""
    init_db()
    yield


app = FastAPI(title="EarthAI", version="0.1.0", lifespan=lifespan)

# ── CORS (permissive for dev) ────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Routers ──────────────────────────────────────────────────────
app.include_router(events_router)
app.include_router(rainfall_router)
app.include_router(optical_router)
app.include_router(sar_router)
app.include_router(classifier_router)
app.include_router(leaderboard_router)
app.include_router(flappy_router)


# ── Custom error handler ────────────────────────────────────────
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Return consistent JSON for all HTTP errors."""
    if isinstance(exc.detail, dict):
        return JSONResponse(status_code=exc.status_code, content=exc.detail)
    return JSONResponse(
        status_code=exc.status_code,
        content={"ok": False, "error": str(exc.status_code), "message": str(exc.detail)},
    )


# ── Static files (catch-all, must be LAST) ──────────────────────
app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")
