r"""backend\app\api\v1\health.py

Health check endpoints.

These endpoints can be used by orchestrators and load balancers to verify
that the service is running.  A simple GET request to `/api/v1/health`
returns a JSON payload with status information.
"""

from fastapi import APIRouter

router = APIRouter()


@router.get("/health")
async def health_check() -> dict[str, str]:
    """Return a basic health indicator."""
    return {"status": "ok"}