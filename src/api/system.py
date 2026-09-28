from fastapi import APIRouter, Request

from core.config import settings
from core.logging import log_route, setup_logging

logger = setup_logging()

router = APIRouter()

@log_route(logger=logger, include_args=True)
@router.get("/health")
async def health(request: Request) -> dict:
    return {
        "status": "ok",
        "service": settings.PROJECT_NAME,
        "version": settings.PROJECT_VERSION,
        "ready": True if getattr(request.app.state, "policy", None) else False,
        }
