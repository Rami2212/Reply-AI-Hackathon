from __future__ import annotations

import json
import logging
from typing import Any, Optional

from openai import OpenAI
from openai.types.chat import ChatCompletionUserMessageParam

from .config import OpenRouterConfig

logger = logging.getLogger(__name__)


class OpenRouterDecisionClient:
    def __init__(self, config: OpenRouterConfig) -> None:
        self.config = config
        self.enabled = config.enabled and bool(config.api_key)
        self.client: Optional[OpenAI] = None

        if self.enabled:
            logger.info("[OpenRouter] LLM enabled — model: %s", config.model)
            self.client = OpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=config.api_key,
            )
        else:
            logger.warning(
                "[OpenRouter] LLM DISABLED — OPENROUTER_API_KEY not set or empty. "
                "Decisions will rely on ML + rules only."
            )

    def review_borderline_case(self, payload: dict[str, Any]) -> Optional[int]:
        """Return 0/1 for uncertain cases, or None when disabled/unavailable."""
        if not self.enabled or self.client is None:
            return None

        headers = {}
        if self.config.site_url:
            headers["HTTP-Referer"] = self.config.site_url
        if self.config.site_name:
            headers["X-OpenRouter-Title"] = self.config.site_name

        prompt = (
            "You are a preventive-risk reviewer. "
            "Given the citizen summary JSON, return ONLY one character: 0 or 1. "
            "Use 1 when support should be activated, otherwise 0.\n"
            f"Citizen summary: {json.dumps(payload, default=str)}"
        )

        try:
            messages: list[ChatCompletionUserMessageParam] = [{"role": "user", "content": prompt}]
            logger.debug("[OpenRouter] Sending review request for CitizenID=%s", payload.get("CitizenID"))
            completion = self.client.chat.completions.create(
                extra_headers=headers,
                model=self.config.model,
                messages=messages,
                temperature=0,
            )
            content = (completion.choices[0].message.content or "").strip()
            logger.info("[OpenRouter] LLM response for CitizenID=%s → %r", payload.get("CitizenID"), content)
            if content.startswith("1"):
                return 1
            if content.startswith("0"):
                return 0
            logger.warning("[OpenRouter] Unexpected LLM response: %r — skipping override", content)
            return None
        except Exception as exc:
            logger.error("[OpenRouter] LLM call failed for CitizenID=%s: %s", payload.get("CitizenID"), exc)
            return None
