"""Catalog endpoints."""

from fastapi import APIRouter

router = APIRouter()


@router.get("/catalog")
async def list_catalog_items() -> dict[str, str]:
    """Placeholder endpoint for catalog resources."""

    return {"message": "Catalog service not yet implemented"}
