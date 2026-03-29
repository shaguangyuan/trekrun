"""
DeepSeek API client wrapper with structured JSON output enforcement.

Uses app.config.settings for configuration.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

import httpx

from app.config import settings

logger = logging.getLogger(__name__)

DEFAULT_BASE_URL = "https://api.deepseek.com"
DEFAULT_MODEL = "deepseek-chat"
DEFAULT_TIMEOUT = 60


class DeepSeekError(Exception):
    """Raised when DeepSeek API returns an error or request fails."""

    def __init__(self, message: str, response_body: Optional[Dict] = None) -> None:
        super().__init__(message)
        self.response_body = response_body


class DeepSeekClient:
    """
    Unified DeepSeek client for sprint analysis AI layer.

    Features:
    - Environment-based configuration
    - Structured JSON output enforcement via system prompt
    - Timeout and retry handling
    - Error logging without exposing API key
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        timeout: Optional[int] = None,
    ) -> None:
        # Read from settings (which loads from .env) or use provided values
        self.api_key = api_key or getattr(settings, "deepseek_api_key", "") or ""
        self.base_url = base_url or getattr(settings, "deepseek_base_url", DEFAULT_BASE_URL) or DEFAULT_BASE_URL
        self.model = model or getattr(settings, "deepseek_model", DEFAULT_MODEL) or DEFAULT_MODEL
        timeout_val = timeout or getattr(settings, "deepseek_timeout", DEFAULT_TIMEOUT)
        self.timeout = int(timeout_val) if timeout_val else DEFAULT_TIMEOUT

        if not self.api_key:
            logger.warning("DeepSeek API key not configured. AI analysis will be unavailable.")

    def _build_headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def chat_completion(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.3,
        max_tokens: int = 4096,
        response_format: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """
        Send chat completion request to DeepSeek API.

        Args:
            messages: List of message dicts with "role" and "content"
            temperature: Sampling temperature (0-2)
            max_tokens: Maximum tokens to generate
            response_format: Optional format spec, e.g., {"type": "json_object"}

        Returns:
            Parsed JSON response from API

        Raises:
            DeepSeekError: If request fails or API returns error
        """
        if not self.api_key:
            raise DeepSeekError("DeepSeek API key not configured")

        url = f"{self.base_url.rstrip('/')}/v1/chat/completions"
        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if response_format:
            payload["response_format"] = response_format

        try:
            with httpx.Client(timeout=self.timeout) as client:
                resp = client.post(url, headers=self._build_headers(), json=payload)
                resp.raise_for_status()
                data = resp.json()
        except httpx.HTTPStatusError as exc:
            body = exc.response.text if exc.response else None
            logger.error("DeepSeek HTTP error: %s - %s", exc.response.status_code if exc.response else "?", body)
            raise DeepSeekError(f"HTTP {exc.response.status_code if exc.response else 'unknown'}: {body}") from exc
        except httpx.RequestError as exc:
            logger.error("DeepSeek request error: %s", exc)
            raise DeepSeekError(f"Request failed: {exc}") from exc
        except Exception as exc:
            logger.error("DeepSeek unexpected error: %s", exc)
            raise DeepSeekError(f"Unexpected error: {exc}") from exc

        return data

    def generate_structured(
        self,
        system_prompt: str,
        user_prompt: str,
        json_schema_hint: Optional[Dict[str, Any]] = None,
        temperature: float = 0.3,
    ) -> Dict[str, Any]:
        """
        Generate structured JSON output using DeepSeek with enforced JSON mode.

        Args:
            system_prompt: System instructions
            user_prompt: User content (analysis input)
            json_schema_hint: Optional schema description for the model
            temperature: Sampling temperature

        Returns:
            Parsed JSON object from model response

        Raises:
            DeepSeekError: If response cannot be parsed as JSON
        """
        # Build enhanced system prompt with JSON enforcement
        enhanced_system = system_prompt.strip()
        if json_schema_hint:
            schema_desc = json.dumps(json_schema_hint, ensure_ascii=False, indent=2)
            enhanced_system += f"\n\nYou must respond with valid JSON matching this structure:\n{schema_desc}"
        enhanced_system += "\n\nIMPORTANT: Your entire response must be a single valid JSON object. Do not include markdown formatting, explanations, or any text outside the JSON."

        messages = [
            {"role": "system", "content": enhanced_system},
            {"role": "user", "content": user_prompt},
        ]

        # Use JSON mode if supported by the model/endpoint
        response_format = {"type": "json_object"}

        logger.debug("DeepSeek request: model=%s, messages_length=%d", self.model, len(messages))

        try:
            result = self.chat_completion(
                messages=messages,
                temperature=temperature,
                response_format=response_format,
            )
        except DeepSeekError:
            # If JSON mode fails, retry without it and parse manually
            logger.warning("JSON mode failed, retrying without format enforcement")
            result = self.chat_completion(
                messages=messages,
                temperature=temperature,
                response_format=None,
            )

        # Extract content from response
        choices = result.get("choices", [])
        if not choices:
            raise DeepSeekError("No choices in response", result)

        content = choices[0].get("message", {}).get("content", "")
        if not content:
            raise DeepSeekError("Empty content in response", result)

        # Parse JSON content
        try:
            # Try to extract JSON from markdown code blocks if present
            json_str = content
            if "```json" in content:
                json_str = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                json_str = content.split("```")[1].split("```")[0].strip()
            parsed = json.loads(json_str)
        except json.JSONDecodeError as exc:
            logger.error("Failed to parse JSON from response: %s", content[:500])
            raise DeepSeekError(f"Invalid JSON in response: {exc}") from exc

        return parsed


def get_client() -> DeepSeekClient:
    """Factory function to get configured DeepSeek client instance."""
    return DeepSeekClient()


def is_configured() -> bool:
    """Check if DeepSeek API is properly configured."""
    api_key = getattr(settings, "deepseek_api_key", "") or ""
    return bool(api_key and api_key.startswith("sk-"))
