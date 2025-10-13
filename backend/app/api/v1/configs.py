"""Configuration endpoints."""

from fastapi import APIRouter

router = APIRouter()


@router.get("/configs")
async def list_configs() -> dict[str, str]:
    """Placeholder endpoint for configuration resources."""

    return {"message": "Configuration service not yet implemented"}
