from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.routers import health, videos, analysis, reports, athletes, ai_analysis, debug
from app.services.runtime_selfcheck import run_startup_selfcheck

app = FastAPI(
    title="SprintForm API",
    version="0.1.0",
    description="Backend for sprint video analysis mini-program (MVP).",
)

# Allow WeChat DevTools (localhost) to call this server during development.
# Tighten allow_origins in production.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router)
app.include_router(videos.router, prefix="/api")
app.include_router(analysis.router, prefix="/api")
app.include_router(reports.router, prefix="/api")
app.include_router(athletes.router, prefix="/api")
app.include_router(ai_analysis.router, prefix="/api")
app.include_router(debug.router, prefix="/api")


@app.on_event("startup")
def _startup_selfcheck() -> None:
    run_startup_selfcheck()
