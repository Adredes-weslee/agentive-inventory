r"""backend/app/services/llm_service.py

Optional integration with Google's Gemini (genai) API.

This service provides functions to generate human-readable explanations for
procurement recommendations and to handle exception triage.  If the
``GEMINI_API_KEY`` environment variable is not set or the ``google-genai``
package is unavailable, the functions will return fallback messages.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

# Prefer `google-generativeai` (current SDK). Fall back to `google.genai` if present.
_GENAI_FLAVOR: str | None = None
try:  # Preferred library
    import google.generativeai as genai  # type: ignore
    _GENAI_FLAVOR = "generativeai"
except Exception:  # pragma: no cover
    try:
        import google.genai as genai  # type: ignore
        _GENAI_FLAVOR = "genai"
    except Exception:
        genai = None  # type: ignore
        _GENAI_FLAVOR = None

_client: Optional[object] = None  # cached client/module depending on flavor


def _get_client() -> tuple[Optional[object], Optional[str]]:
    """Return a configured Gemini client/module and flavor if credentials are available."""

    if genai is None or _GENAI_FLAVOR is None:
        return None, None

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return None, None

    global _client
    if _client is None:
        try:
            if _GENAI_FLAVOR == "generativeai":
                # Module-level configure; the module itself is the "client".
                genai.configure(api_key=api_key)  # type: ignore[attr-defined]
                _client = genai
            else:
                # Older google.genai style
                _client = genai.Client(api_key=api_key)  # type: ignore[attr-defined]
        except Exception:
            return None, None
    return _client, _GENAI_FLAVOR


def explain_recommendation(
    context_object: Dict[str, Any],
    recommendations: List[Dict[str, Any]],
) -> str:
    """Return an explanation of procurement recommendations in business terms."""

    client, flavor = _get_client()
    if client is None or flavor is None:
        # Return a simple fallback explanation if LLM is unavailable
        return (
            "Based on the forecast and configured thresholds, the system recommends purchasing the specified quantities. "
            "These quantities aim to maintain the target service level and maximise GMROI while respecting cash limits."
        )

    # Simple text prompt works with both SDKs.
    prompt_text = (
        "You are an assistant helping explain inventory procurement recommendations. "
        "Given the following context and recommendations, explain the reasoning in plain language suitable for a CFO or "
        "operations manager. Highlight service levels, cash constraints, and uncertainty.\n\n"
        f"Context: {context_object}\n\n"
        f"Recommendations: {recommendations}"
    )

    try:
        # 1.5-flash is widely available; keep your env default if you have 2.5 access
        model_name = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
        if flavor == "generativeai":
            # google-generativeai
            model = client.GenerativeModel(model_name)  # type: ignore[attr-defined]
            resp = model.generate_content(prompt_text)
            text = getattr(resp, "text", None) or getattr(resp, "output_text", None)
            return text.strip() if isinstance(text, str) and text else str(resp)
        else:
            # google.genai (older)
            response = client.responses.generate(  # type: ignore[attr-defined]
                model=model_name,
                contents=[{"role": "user", "content": prompt_text}],
            )
            text = getattr(response, "output_text", None) or getattr(response, "text", None)
            return text.strip() if isinstance(text, str) and text else str(response)
    except Exception:
        return (
            "An automated explanation could not be generated due to an error with the LLM service. "
            "Please refer to the raw recommendation data for details."
        )
