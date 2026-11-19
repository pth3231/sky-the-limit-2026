"""FastAPI application exposing stateless draft recommendations.

Run locally with ``uv run uvicorn app.api.main:app --reload`` from ``backend/``.
"""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import Depends, FastAPI, Request
from fastapi.responses import JSONResponse

from app.config import Settings
from app.engine.errors import DraftStateError
from app.engine.models import ChampionSummary, DraftRequest, DraftResponse, TeamSummary
from app.engine.service import DraftEngine, load_engine


def _engine(request: Request) -> DraftEngine:
    return request.app.state.engine


EngineDependency = Annotated[DraftEngine, Depends(_engine)]


def create_app(engine: DraftEngine | None = None) -> FastAPI:
    """Create the API; without an ``engine`` it is loaded from settings at startup."""

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        if getattr(app.state, "engine", None) is None:
            settings = Settings()
            app.state.engine = load_engine(
                settings.derived_dir, settings.data_dir, settings.cloud9_team_id
            )
        yield

    app = FastAPI(title="Draft Coach API", version="0.1.0", lifespan=lifespan)
    app.state.engine = engine

    @app.exception_handler(DraftStateError)
    async def draft_state_error(_: Request, error: DraftStateError) -> JSONResponse:
        return JSONResponse(status_code=422, content={"detail": str(error)})

    @app.get("/health")
    def health() -> dict[str, str]:
        """Report that the API is up."""
        return {"status": "ok"}

    @app.get("/teams", response_model=list[TeamSummary])
    def teams(engine: EngineDependency) -> list[TeamSummary]:
        """List opponents with recorded games."""
        return engine.opponents()

    @app.get("/champions", response_model=list[ChampionSummary])
    def champions(engine: EngineDependency) -> list[ChampionSummary]:
        """List draftable champions."""
        return engine.champion_list()

    @app.get("/patches")
    def patches(engine: EngineDependency) -> list[str]:
        """List patches with data, newest first."""
        return engine.patches()

    @app.post("/draft/recommend", response_model=DraftResponse)
    def recommend(request: DraftRequest, engine: EngineDependency) -> DraftResponse:
        """Recommend the next move for the submitted series and draft state."""
        return engine.recommend(request)

    return app


app = create_app()
