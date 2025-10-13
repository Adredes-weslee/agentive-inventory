"""Approvals endpoints."""

from fastapi import APIRouter

router = APIRouter()


@router.get("/approvals")
async def list_approvals() -> dict[str, str]:
    """Placeholder endpoint for approvals resources."""

    return {"message": "Approvals service not yet implemented"}
