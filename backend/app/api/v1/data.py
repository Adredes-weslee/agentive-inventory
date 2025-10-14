r"""backend\app\api\v1\data.py"""

from __future__ import annotations

from fastapi import APIRouter

from ...services.validation_service import ValidationService

router = APIRouter()
_validation_service = ValidationService()


@router.get("/data/validate")
def validate() -> dict:
    return _validation_service.run()
