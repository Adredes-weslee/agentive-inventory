"""
Optional integration with Google's Gemini (genai) API.

This service provides functions to generate human‑readable explanations for
procurement recommendations and to handle exception triage.  If the
``GEMINI_API_KEY`` environment variable is not set or the ``google-genai``
package is unavailable, the functions will return fallback messages.
"""

from __future__ import annotations

import os
from typing import Dict, Any, List

try:
    import google.generativeai as genai  # type: ignore
except ImportError:
    genai = None  # type: ignore


def _get_client():
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key or genai is None:
        return None
    # For the genai SDK, you instantiate the model directly rather than a client.
    return genai.GenerativeModel(model_name="gemini-pro", api_key=api_key)


def explain_recommendation(context_object: Dict[str, Any], recommendations: List[Dict[str, Any]]) -> str:
    """Return an explanation of procurement recommendations in business terms.

    Args:
        context_object: A dictionary containing business context, e.g. service
            levels, cash limits and disruption signals.
        recommendations: A list of recommendation dictionaries.

    Returns:
        A string explaining the rationale behind the recommendations.
    """
    model = _get_client()
    if model is None:
        # Return a simple fallback explanation if LLM is unavailable
        return (
            "Based on the forecast and configured thresholds, the system recommends purchasing the specified quantities. "
            "These quantities aim to maintain the target service level and maximise GMROI while respecting cash limits."
        )
    prompt = (
        "You are an assistant helping explain inventory procurement recommendations. "
        "Given the following context and recommendations, explain the reasoning in plain language suitable for a CFO or operations manager.\n"
        f"Context: {context_object}\n"
        f"Recommendations: {recommendations}\n"
        "Please highlight service levels, cash constraints and any uncertainty."
    )
    try:
        response = model.generate_content(prompt)
        # The genai API returns objects with a `text` attribute
        return response.text if hasattr(response, "text") else str(response)
    except Exception:
        return (
            "An automated explanation could not be generated due to an error with the LLM service. "
            "Please refer to the raw recommendation data for details."
        )