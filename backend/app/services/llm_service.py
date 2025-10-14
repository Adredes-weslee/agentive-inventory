r"""backend/app/services/llm_service.py

Optional integration with Google's Gemini (genai) API.

This service provides functions to generate human‑readable explanations for
procurement recommendations and to handle exception triage.  If the
``GEMINI_API_KEY`` environment variable is not set or the ``google-genai``
package is unavailable, the functions will return fallback messages.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

try:
    import google.genai as genai  # type: ignore
except ImportError:  # pragma: no cover - dependency is optional at runtime
    genai = None  # type: ignore


_client: Optional["genai.Client"] = None  # type: ignore[misc]


def _get_client() -> Optional["genai.Client"]:
    """Return a cached google.genai client if credentials are available."""

    if genai is None:
        return None

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return None

    global _client
    if _client is None:
        try:
            _client = genai.Client(api_key=api_key)  # type: ignore[call-arg]
        except Exception:
            # Do not allow failures when constructing the client to bubble up.
            return None
    return _client


def explain_recommendation(
    context_object: Dict[str, Any],
    recommendations: List[Dict[str, Any]],
) -> str:
    """Return an explanation of procurement recommendations in business terms."""

    client = _get_client()
    if client is None:
        # Return a simple fallback explanation if LLM is unavailable
        return (
            "Based on the forecast and configured thresholds, the system recommends purchasing the specified quantities. "
            "These quantities aim to maintain the target service level and maximise GMROI while respecting cash limits."
        )

    prompt = [
        {
            "role": "user",
            "content": (
                "You are an assistant helping explain inventory procurement recommendations. "
                "Given the following context and recommendations, explain the reasoning in plain language suitable for a CFO or operations manager.\n"
                f"Context: {context_object}\n"
                f"Recommendations: {recommendations}\n"
                "Please highlight service levels, cash constraints and any uncertainty."
            ),
        }
    ]

    try:
        response = client.responses.generate(  # type: ignore[attr-defined]
            model="gemini-2.5-flash",
            contents=prompt,
        )
        return getattr(response, "output_text", None) or str(response)
    except Exception:
        return (
            "An automated explanation could not be generated due to an error with the LLM service. "
            "Please refer to the raw recommendation data for details."
        )
