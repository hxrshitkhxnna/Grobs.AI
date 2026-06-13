"""
Unified LLM Service Layer

Centralized service for all AI/LLM operations:
- Text generation
- Structured output generation
- Embedding generation
- Streaming responses

This prevents duplicated API logic across AI services.
"""
import os
import json
import logging
import re
from typing import Dict, List, Optional, Any, Union, Generator
from dataclasses import dataclass, field
from enum import Enum

# Configuration
logger = logging.getLogger(__name__)

from app.core.config import settings

# Supported providers
class LLMProvider(Enum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"
    HUGGINGFACE = "huggingface"
    LOCAL = "local"


# Try to import providers
try:
    from openai import OpenAI, AsyncOpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    import anthropic
    from anthropic import AsyncAnthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False

try:
    from google import genai
    from google.genai import types
    GOOGLE_AVAILABLE = True
except ImportError:
    GOOGLE_AVAILABLE = False


@dataclass
class LLMResponse:
    """Standardized LLM response."""
    content: str
    model: str
    provider: str
    usage: Optional[Dict[str, int]] = None
    raw_response: Optional[Any] = None
    is_fallback: bool = False  # Indicates if heuristic fallback was used


@dataclass
class EmbeddingResponse:
    """Standardized embedding response."""
    embedding: List[float]
    model: str
    provider: str


class LLMService:
    """
    Unified LLM service for all AI operations.
    Supports multiple providers with consistent interface.
    """

    # FIX: Explicit set of valid provider names for fast validation
    _VALID_PROVIDERS = {"openai", "anthropic", "google", "huggingface", "local"}

    def __init__(self, provider: str = None):
        # Use the setting, or default to "google"
        raw_provider = provider or settings.LLM_PROVIDER or "google"
        
        # Normalize to lowercase to prevent "Google" vs "google" errors
        raw_provider = raw_provider.lower() 
        
        if raw_provider not in self._VALID_PROVIDERS:
            logger.warning(f"Unknown LLM provider {raw_provider} — falling back to 'google'.")
            raw_provider = "google"
            
        self.provider_name = raw_provider
        self.default_provider = raw_provider # Ensure this is set explicitly
        self.cache = {} 
        self._initialize_providers()

    def _initialize_providers(self):
        """Initialize available providers."""
        # OpenAI
        if OPENAI_AVAILABLE and settings.OPENAI_API_KEY:
            self.openai_client = OpenAI(api_key=settings.OPENAI_API_KEY)
            self.async_openai_client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
            self.openai_model = settings.OPENAI_MODEL or "gpt-4o"
        else:
            self.openai_client = None
            self.async_openai_client = None

        # Anthropic
        if ANTHROPIC_AVAILABLE and settings.ANTHROPIC_API_KEY:
            self.anthropic_client = anthropic.Anthropic(
                api_key=settings.ANTHROPIC_API_KEY
            )
            self.async_anthropic_client = AsyncAnthropic(
                api_key=settings.ANTHROPIC_API_KEY
            )
            self.anthropic_model = settings.ANTHROPIC_MODEL or "claude-3-5-sonnet-20241022"
        else:
            self.anthropic_client = None
            self.async_anthropic_client = None

        # Google Gemini
        logger.info(f"Initializing Google Gemini: GOOGLE_AVAILABLE={GOOGLE_AVAILABLE}, GEMINI_API_KEY={'Present' if settings.GEMINI_API_KEY else 'Missing'}")
        if GOOGLE_AVAILABLE and settings.GEMINI_API_KEY:
            # Check if API key is a placeholder
            if settings.GEMINI_API_KEY == "your-gemini-api-key-here":
                logger.warning("Gemini API key is set to placeholder value. Disabling Google Gemini.")
                self.google_client = None
            else:
                try:
                    self.google_client = genai.Client(api_key=settings.GEMINI_API_KEY)
                    # Use gemini-2.0-flash as it is more widely available in new SDKs
                    self.google_model = settings.GEMINI_MODEL or "gemini-2.0-flash"
                    logger.info(f"Google Gemini initialized with model: {self.google_model}")
                except Exception as e:
                    logger.error(f"Failed to initialize Google Gemini client: {e}")
                    self.google_client = None
        else:
            self.google_client = None

        self.default_provider = self.provider_name

    def refresh_config(self):
        """Manually refresh config from environment variables."""
        # Reloading settings might be needed if they changed on disk
        # But for now, we just use the current settings object
        self.provider_name = settings.LLM_PROVIDER or "google"
        self._initialize_providers()

    def validate_api_keys(self) -> Dict[str, Any]:
        """
        Validate configured API keys and return status of all providers.
        
        Returns:
            Dict with provider status and recommendations
        """
        status = {
            "configured_provider": self.provider_name,
            "providers": {},
            "active_provider": None,
            "is_valid": False,
            "warnings": [],
            "recommendations": []
        }
        
        # Check OpenAI
        if OPENAI_AVAILABLE:
            status["providers"]["openai"] = {
                "available": True,
                "configured": bool(settings.OPENAI_API_KEY),
                "model": settings.OPENAI_MODEL or "gpt-4o"
            }
            if settings.OPENAI_API_KEY:
                status["active_provider"] = "openai"
        else:
            status["providers"]["openai"] = {"available": False, "configured": False}
        
        # Check Anthropic
        if ANTHROPIC_AVAILABLE:
            status["providers"]["anthropic"] = {
                "available": True,
                "configured": bool(settings.ANTHROPIC_API_KEY),
                "model": settings.ANTHROPIC_MODEL or "claude-3-5-sonnet-20241022"
            }
            if settings.ANTHROPIC_API_KEY and not status["active_provider"]:
                status["active_provider"] = "anthropic"
        else:
            status["providers"]["anthropic"] = {"available": False, "configured": False}
        
        # Check Google Gemini
        if GOOGLE_AVAILABLE:
            status["providers"]["google"] = {
                "available": True,
                "configured": bool(settings.GEMINI_API_KEY),
                "model": settings.GEMINI_MODEL or "gemini-2.0-flash"
            }
            if settings.GEMINI_API_KEY and not status["active_provider"]:
                status["active_provider"] = "google"
        else:
            status["providers"]["google"] = {"available": False, "configured": False}
        
        # Determine if we have a working provider
        if status["active_provider"]:
            status["is_valid"] = True
        else:
            status["warnings"].append("No LLM provider is properly configured. Using heuristic fallback mode.")
            status["recommendations"].append("Configure at least one LLM provider (Google Gemini recommended for free tier).")
        
        # Check if configured provider matches active provider
        if self.provider_name != "local" and status["active_provider"] != self.provider_name:
            if status["active_provider"]:
                status["warnings"].append(
                    f"Configured provider '{self.provider_name}' is not available. Using '{status['active_provider']}' instead."
                )
            else:
                status["warnings"].append(
                    f"Configured provider '{self.provider_name}' is not available. Using heuristic fallback mode."
                )
        
        return status

    def _is_provider_available(self, provider: str) -> bool:
        """Check if a provider is configured and available."""
        if provider == "openai":
            return self.openai_client is not None
        if provider == "anthropic":
            return self.anthropic_client is not None
        if provider == "google":
            return self.google_client is not None
        if provider == "local":
            return True
        return False

    def generate_text(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        use_cache: bool = True,
        **kwargs
    ) -> LLMResponse:
        """
        Generate text using LLM with multi-provider fallback and caching.
        """
        if use_cache:
            cache_key = f"{prompt}:{system_prompt}:{model}:{temperature}:{max_tokens}"
            if cache_key in self.cache:
                logger.info("Using cached LLM response")
                return self.cache[cache_key]

        # FIX: Validate inputs early
        if not prompt or not prompt.strip():
            logger.error("generate_text called with empty prompt.")
            return LLMResponse(content="", model="none", provider="none")

        # FIX: Clamp temperature to a safe range
        temperature = max(0.0, min(temperature, 2.0))

        primary_provider = kwargs.get("provider", self.default_provider)
        
        # Define priority order for fallbacks
        providers_to_try = [primary_provider]
        for p in ["google", "openai", "anthropic"]:
            if p != primary_provider:
                providers_to_try.append(p)
        
        # Add local as absolute last resort if nothing else works
        providers_to_try.append("local")

        result = None
        for provider in providers_to_try:
            if not self._is_provider_available(provider):
                continue
                
            try:
                if provider == "openai":
                    result = self._generate_openai(
                        prompt, system_prompt, model if provider == primary_provider else self.openai_model, temperature, max_tokens
                    )
                elif provider == "anthropic":
                    result = self._generate_anthropic(
                        prompt, system_prompt, model if provider == primary_provider else self.anthropic_model, temperature, max_tokens
                    )
                elif provider == "google":
                    result = self._generate_google(
                        prompt, system_prompt, model if provider == primary_provider else self.google_model, temperature, max_tokens
                    )
                elif provider == "local":
                    result = self._generate_local(prompt, system_prompt)
                
                # If we got a valid result (not a fallback), break the loop
                if result and result.provider != "fallback":
                    logger.info(f"Successfully generated text using {provider}")
                    break
            except Exception as e:
                logger.warning(f"Provider {provider} failed: {e}. Trying next available provider.")
                continue

        if not result or result.provider == "fallback":
            result = self._generate_fallback(prompt, system_prompt)

        if use_cache and result and result.provider != "fallback":
            self.cache[cache_key] = result
        
        return result

    async def generate_text_async(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        use_cache: bool = True,
        **kwargs
    ) -> LLMResponse:
        """
        Generate text using LLM asynchronously with multi-provider fallback and caching.
        """
        if use_cache:
            cache_key = f"{prompt}:{system_prompt}:{model}:{temperature}:{max_tokens}:async"
            if cache_key in self.cache:
                logger.info("Using cached async LLM response")
                return self.cache[cache_key]

        # FIX: Validate inputs early
        if not prompt or not prompt.strip():
            logger.error("generate_text_async called with empty prompt.")
            return LLMResponse(content="", model="none", provider="none")

        temperature = max(0.0, min(temperature, 2.0))
        primary_provider = kwargs.get("provider", self.default_provider)
        
        # Define priority order for fallbacks
        providers_to_try = [primary_provider]
        for p in ["google", "openai", "anthropic"]:
            if p != primary_provider:
                providers_to_try.append(p)
        
        providers_to_try.append("local")

        result = None
        for provider in providers_to_try:
            if not self._is_provider_available(provider):
                continue
                
            try:
                if provider == "openai":
                    result = await self._generate_openai_async(
                        prompt, system_prompt, model if provider == primary_provider else self.openai_model, temperature, max_tokens
                    )
                elif provider == "anthropic":
                    result = await self._generate_anthropic_async(
                        prompt, system_prompt, model if provider == primary_provider else self.anthropic_model, temperature, max_tokens
                    )
                elif provider == "google":
                    result = await self._generate_google_async(
                        prompt, system_prompt, model if provider == primary_provider else self.google_model, temperature, max_tokens
                    )
                elif provider == "local":
                    result = self._generate_local(prompt, system_prompt)
                
                if result and result.provider != "fallback":
                    logger.info(f"Successfully generated text async using {provider}")
                    break
            except Exception as e:
                logger.warning(f"Async provider {provider} failed: {e}. Trying next available provider.")
                continue

        if not result or result.provider == "fallback":
            result = self._generate_fallback(prompt, system_prompt)

        if use_cache and result and result.provider != "fallback":
            self.cache[cache_key] = result
        
        return result

    def _generate_local(self, prompt: str, system_prompt: Optional[str]) -> LLMResponse:
        """Mock generation for local mode."""
        return LLMResponse(
            content="Local mode enabled. Structured output will use rule-based parsing.",
            model="local-rule-based",
            provider="local",
        )

    async def _generate_openai_async(
        self,
        prompt: str,
        system_prompt: Optional[str],
        model: str,
        temperature: float,
        max_tokens: Optional[int],
    ) -> LLMResponse:
        """Generate using OpenAI asynchronously."""
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        try:
            response = await self.async_openai_client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        except Exception as e:
            logger.error("OpenAI async generation failed: %s", e)
            return self._generate_fallback(prompt, system_prompt)

        return LLMResponse(
            content=response.choices[0].message.content or "",
            model=model,
            provider="openai",
            usage={
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            }
            if response.usage
            else None,
            raw_response=response,
        )

    async def _generate_anthropic_async(
        self,
        prompt: str,
        system_prompt: Optional[str],
        model: str,
        temperature: float,
        max_tokens: Optional[int],
    ) -> LLMResponse:
        """Generate using Anthropic asynchronously."""
        try:
            response = await self.async_anthropic_client.messages.create(
                model=model,
                max_tokens=max_tokens or 1024,
                temperature=temperature,
                system=system_prompt or "",
                messages=[{"role": "user", "content": prompt}],
            )
        except Exception as e:
            logger.error("Anthropic async generation failed: %s", e)
            return self._generate_fallback(prompt, system_prompt)

        content_text = ""
        if response.content and len(response.content) > 0:
            content_text = response.content[0].text or ""

        return LLMResponse(
            content=content_text,
            model=model,
            provider="anthropic",
            usage={
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens,
            }
            if hasattr(response, "usage")
            else None,
            raw_response=response,
        )

    async def _generate_google_async(
        self,
        prompt: str,
        system_prompt: Optional[str],
        model: str,
        temperature: float,
        max_tokens: Optional[int],
    ) -> LLMResponse:
        """Generate using Google Gemini asynchronously."""
        try:
            config = types.GenerateContentConfig(
                system_instruction=system_prompt if system_prompt else None,
                temperature=temperature,
                max_output_tokens=max_tokens if max_tokens else None,
                http_options={'timeout': 30000} # 30 seconds timeout
            )
            
            response = await self.google_client.aio.models.generate_content(
                model=model,
                contents=prompt,
                config=config,
            )
        except Exception as e:
            msg = str(e).lower()
            if "429" in msg or "resource_exhausted" in msg:
                logger.error("Google Gemini quota exceeded (429 RESOURCE_EXHAUSTED). Returning fallback.")
                return self._generate_fallback(prompt, system_prompt)
                
            logger.warning(f"Google Gemini async generation with config failed: {e}. Falling back to combined prompt.")
            full_prompt = f"{system_prompt}\n\n{prompt}" if system_prompt else prompt
            try:
                response = await self.google_client.aio.models.generate_content(
                    model=model,
                    contents=full_prompt,
                    config=types.GenerateContentConfig(http_options={'timeout': 30000})
                )
            except Exception as inner_e:
                logger.error("Google Gemini async generation failed: %s", inner_e)
                return self._generate_fallback(prompt, system_prompt)

        content_text = ""
        try:
            if hasattr(response, "text") and response.text is not None:
                content_text = response.text
            elif hasattr(response, "candidates") and response.candidates:
                content_text = response.candidates[0].content.parts[0].text
        except Exception as e:
            logger.error(f"Error extracting text from Gemini async response: {e}")

        return LLMResponse(
            content=content_text,
            model=model,
            provider="google",
            raw_response=response,
        )

    def _generate_openai(
        self,
        prompt: str,
        system_prompt: Optional[str],
        model: str,
        temperature: float,
        max_tokens: Optional[int],
    ) -> LLMResponse:
        """Generate using OpenAI."""
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        # FIX: Wrap in try/except so a single provider failure returns a clean fallback
        try:
            response = self.openai_client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        except Exception as e:
            logger.error("OpenAI generation failed: %s", e)
            return self._generate_fallback(prompt, system_prompt)

        return LLMResponse(
            content=response.choices[0].message.content or "",
            model=model,
            provider="openai",
            usage={
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            }
            if response.usage
            else None,
            raw_response=response,
        )

    def _generate_anthropic(
        self,
        prompt: str,
        system_prompt: Optional[str],
        model: str,
        temperature: float,
        max_tokens: Optional[int],
    ) -> LLMResponse:
        """Generate using Anthropic."""
        # FIX: Wrap in try/except so a single provider failure returns a clean fallback
        try:
            response = self.anthropic_client.messages.create(
                model=model,
                max_tokens=max_tokens or 1024,
                temperature=temperature,
                system=system_prompt or "",
                messages=[{"role": "user", "content": prompt}],
            )
        except Exception as e:
            logger.error("Anthropic generation failed: %s", e)
            return self._generate_fallback(prompt, system_prompt)

        # FIX: Guard against empty content list (edge case in some Anthropic responses)
        content_text = ""
        if response.content and len(response.content) > 0:
            content_text = response.content[0].text or ""

        return LLMResponse(
            content=content_text,
            model=model,
            provider="anthropic",
            usage={
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens,
            }
            if hasattr(response, "usage")
            else None,
            raw_response=response,
        )

    def _generate_google(
        self,
        prompt: str,
        system_prompt: Optional[str],
        model: str,
        temperature: float,
        max_tokens: Optional[int],
    ) -> LLMResponse:
        """Generate using Google Gemini."""
        # Use types.GenerateContentConfig for better control if GOOGLE_AVAILABLE
        try:
            config = types.GenerateContentConfig(
                system_instruction=system_prompt if system_prompt else None,
                temperature=temperature,
                max_output_tokens=max_tokens if max_tokens else None,
                http_options={'timeout': 30000} # 30 seconds timeout
            )
            
            response = self.google_client.models.generate_content(
                model=model,
                contents=prompt,
                config=config,
            )
        except Exception as e:
            logger.warning(f"Google Gemini generation with config failed: {e}. Falling back to combined prompt.")
            # Fallback to combined prompt if config usage fails
            full_prompt = f"{system_prompt}\n\n{prompt}" if system_prompt else prompt
            try:
                response = self.google_client.models.generate_content(
                    model=model,
                    contents=full_prompt,
                    config=types.GenerateContentConfig(http_options={'timeout': 30000})
                )
            except Exception as inner_e:
                logger.error("Google Gemini generation failed: %s", inner_e)
                return self._generate_fallback(prompt, system_prompt)

        # FIX: Gemini may return None for .text on safety-blocked responses
        content_text = ""
        try:
            if hasattr(response, "text") and response.text is not None:
                content_text = response.text
            elif hasattr(response, "candidates") and response.candidates:
                content_text = response.candidates[0].content.parts[0].text
        except Exception as e:
            logger.error(f"Error extracting text from Gemini response: {e}")

        return LLMResponse(
            content=content_text,
            model=model,
            provider="google",
            raw_response=response,
        )

    def _generate_fallback(self, prompt: str, system_prompt: Optional[str]) -> LLMResponse:
        """Fallback when no provider is available."""
        logger.warning("No LLM provider available, returning placeholder or heuristic result")
        
        # If it looks like a resume parsing task, use the heuristic
        if "RESUME TEXT:" in prompt:
            try:
                heuristic_data = self._heuristic_resume_parser(prompt)
                if heuristic_data and "error" not in heuristic_data:
                    return LLMResponse(
                        content=json.dumps(heuristic_data),
                        model="heuristic-v4",
                        provider="heuristic",
                        is_fallback=True
                    )
            except Exception as e:
                logger.error(f"Heuristic fallback failed: {e}")

        return LLMResponse(
            content="AI service temporarily unavailable. Our heuristic model is processing your request.",
            model="heuristic-core",
            provider="fallback",
            is_fallback=True
        )

    async def generate_structured_output_async(
        self,
        prompt: str,
        schema: Optional[Dict[str, Any]] = None,
        system_prompt: Optional[str] = None,
        use_cache: bool = True,
        **kwargs
    ) -> Union[Dict[str, Any], Any]:
        """
        Generate structured JSON output asynchronously with multi-provider fallback and caching.
        """
        response_model = kwargs.get("response_model")
        
        # If schema is missing but response_model is provided, extract schema from model
        if schema is None and response_model:
            if hasattr(response_model, "model_json_schema"): # Pydantic v2
                schema = response_model.model_json_schema()
            elif hasattr(response_model, "schema"): # Pydantic v1
                schema = response_model.schema()
            else:
                # Handle List[Model] or other typing generics
                try:
                    from typing import get_args, get_origin
                    if get_origin(response_model) is list:
                        inner_type = get_args(response_model)[0]
                        if hasattr(inner_type, "model_json_schema"):
                            schema = {
                                "type": "array",
                                "items": inner_type.model_json_schema()
                            }
                        elif hasattr(inner_type, "schema"):
                            schema = {
                                "type": "array",
                                "items": inner_type.schema()
                            }
                except:
                    pass

        if use_cache:
            # We must have a schema (dict) for the cache key
            cache_schema = schema if isinstance(schema, dict) else {}
            cache_key = f"{prompt}:{json.dumps(cache_schema)}:{system_prompt}:structured:async"
            if cache_key in self.cache:
                logger.info("Using cached async structured LLM response")
                cached_res = self.cache[cache_key]
                # If we have a response_model, try to parse the cached dict back to the model
                if response_model and isinstance(cached_res, dict) and "error" not in cached_res:
                    try:
                        from typing import get_args, get_origin
                        if hasattr(response_model, "model_validate"):
                            return response_model.model_validate(cached_res)
                        elif hasattr(response_model, "parse_obj"):
                            return response_model.parse_obj(cached_res)
                        elif get_origin(response_model) is list:
                            inner_type = get_args(response_model)[0]
                            if hasattr(inner_type, "model_validate"):
                                return [inner_type.model_validate(item) for item in cached_res]
                            elif hasattr(inner_type, "parse_obj"):
                                return [inner_type.parse_obj(item) for item in cached_res]
                    except Exception as e:
                        logger.warning(f"Failed to parse cached response to model: {e}")
                return cached_res

        # FIX: Validate inputs early
        if not prompt or not prompt.strip():
            logger.error("generate_structured_output_async called with empty prompt.")
            return {"error": "Empty prompt provided"}

        if not isinstance(schema, dict):
            logger.error(f"generate_structured_output_async: schema must be a dict (got {type(schema)}).")
            return {"error": "Invalid schema"}

        schema_prompt = (
            "Return your response as valid JSON matching this schema:\n"
            f"{json.dumps(schema, indent=2)}\n\n"
            "Do not include any explanation or markdown formatting. Only return valid JSON."
        )

        primary_provider = kwargs.get("provider", self.default_provider)
        providers_to_try = [primary_provider]
        for p in ["google", "openai", "anthropic"]:
            if p != primary_provider:
                providers_to_try.append(p)

        result = None
        for provider in providers_to_try:
            if not self._is_provider_available(provider):
                continue
                
            try:
                # Force specific provider for this attempt
                attempt_kwargs = kwargs.copy()
                attempt_kwargs["provider"] = provider
                
                response = await self.generate_text_async(
                    prompt=f"{schema_prompt}\n\n{prompt}",
                    system_prompt=system_prompt,
                    use_cache=False, 
                    **attempt_kwargs,
                )

                if response.provider == "fallback":
                    continue

                content = response.content
                
                # Robust JSON extraction
                json_match = re.search(r'(\{.*\})', content, re.DOTALL)
                if json_match:
                    content = json_match.group(1)
                else:
                    # If no { } found, try stripping markdown fences as fallback
                    content = re.sub(r"^```[a-zA-Z]*\s*", "", content)
                    content = re.sub(r"\s*```$", "", content.strip())
                
                content = content.strip()
                if not content:
                    logger.warning(f"Empty content from {provider} for structured output. Trying next.")
                    continue
                    
                try:
                    result = json.loads(content)
                except json.JSONDecodeError:
                    # Try to fix common JSON errors (like trailing commas)
                    try:
                        fixed_content = re.sub(r',\s*([\]}])', r'\1', content)
                        result = json.loads(fixed_content)
                    except:
                        logger.warning(f"Failed to parse JSON from {provider}. Trying next provider.")
                        continue
                
                # If we got here, we have a valid result
                if result and "error" not in result:
                    logger.info(f"Successfully generated structured output using {provider}")
                    break
                    
            except Exception as e:
                logger.warning(f"Structured output failed for {provider}: {e}. Trying next.")
                continue

        # If all providers failed or returned invalid JSON
        if not result or "error" in result:
            logger.warning("All LLM providers failed for structured output. Falling back to heuristic.")
            # If it's a resume prompt, use heuristic
            if "RESUME TEXT:" in prompt:
                result = self._heuristic_resume_parser(prompt)
            else:
                result = {"error": "AI service temporarily unavailable", "provider": "fallback"}

        if use_cache and result and "error" not in result:
            self.cache[cache_key] = result
        
        # If successful and response_model provided, parse into model
        if response_model and result and "error" not in result:
            try:
                from typing import get_args, get_origin
                if hasattr(response_model, "model_validate"):
                    return response_model.model_validate(result)
                elif hasattr(response_model, "parse_obj"):
                    return response_model.parse_obj(result)
                elif get_origin(response_model) is list:
                    inner_type = get_args(response_model)[0]
                    if hasattr(inner_type, "model_validate"):
                        return [inner_type.model_validate(item) for item in result]
                    elif hasattr(inner_type, "parse_obj"):
                        return [inner_type.parse_obj(item) for item in result]
            except Exception as e:
                logger.warning(f"Failed to parse response to model: {e}")
                
        return result

    def generate_structured_output(
        self,
        prompt: str,
        schema: Optional[Dict[str, Any]] = None,
        system_prompt: Optional[str] = None,
        use_cache: bool = True,
        **kwargs
    ) -> Union[Dict[str, Any], Any]:
        """
        Generate structured JSON output with multi-provider fallback and caching.
        """
        response_model = kwargs.get("response_model")
        
        # If schema is missing but response_model is provided, extract schema from model
        if schema is None and response_model:
            if hasattr(response_model, "model_json_schema"): # Pydantic v2
                schema = response_model.model_json_schema()
            elif hasattr(response_model, "schema"): # Pydantic v1
                schema = response_model.schema()
            else:
                try:
                    from typing import get_args, get_origin
                    if get_origin(response_model) is list:
                        inner_type = get_args(response_model)[0]
                        if hasattr(inner_type, "model_json_schema"):
                            schema = {
                                "type": "array",
                                "items": inner_type.model_json_schema()
                            }
                        elif hasattr(inner_type, "schema"):
                            schema = {
                                "type": "array",
                                "items": inner_type.schema()
                            }
                except:
                    pass

        if use_cache:
            cache_schema = schema if isinstance(schema, dict) else {}
            cache_key = f"{prompt}:{json.dumps(cache_schema)}:{system_prompt}:structured"
            if cache_key in self.cache:
                logger.info("Using cached structured LLM response")
                cached_res = self.cache[cache_key]
                if response_model and isinstance(cached_res, dict) and "error" not in cached_res:
                    try:
                        from typing import get_args, get_origin
                        if hasattr(response_model, "model_validate"):
                            return response_model.model_validate(cached_res)
                        elif hasattr(response_model, "parse_obj"):
                            return response_model.parse_obj(cached_res)
                        elif get_origin(response_model) is list:
                            inner_type = get_args(response_model)[0]
                            if hasattr(inner_type, "model_validate"):
                                return [inner_type.model_validate(item) for item in cached_res]
                            elif hasattr(inner_type, "parse_obj"):
                                return [inner_type.parse_obj(item) for item in cached_res]
                    except:
                        pass
                return cached_res

        # FIX: Validate inputs early
        if not prompt or not prompt.strip():
            logger.error("generate_structured_output called with empty prompt.")
            return {"error": "Empty prompt provided"}

        if not isinstance(schema, dict):
            logger.error(f"generate_structured_output: schema must be a dict (got {type(schema)}).")
            return {"error": "Invalid schema"}

        schema_prompt = (
            "Return your response as valid JSON matching this schema:\n"
            f"{json.dumps(schema, indent=2)}\n\n"
            "Do not include any explanation or markdown formatting. Only return valid JSON."
        )

        primary_provider = kwargs.get("provider", self.default_provider)
        providers_to_try = [primary_provider]
        for p in ["google", "openai", "anthropic"]:
            if p != primary_provider:
                providers_to_try.append(p)

        result = None
        for provider in providers_to_try:
            if not self._is_provider_available(provider):
                continue
                
            try:
                attempt_kwargs = kwargs.copy()
                attempt_kwargs["provider"] = provider
                
                response = self.generate_text(
                    prompt=f"{schema_prompt}\n\n{prompt}",
                    system_prompt=system_prompt,
                    use_cache=False, 
                    **attempt_kwargs,
                )

                if response.provider == "fallback":
                    continue

                content = response.content
                
                # Robust JSON extraction
                json_match = re.search(r'(\{.*\})', content, re.DOTALL)
                if json_match:
                    content = json_match.group(1)
                else:
                    # If no { } found, try stripping markdown fences as fallback
                    content = re.sub(r"^```[a-zA-Z]*\s*", "", content)
                    content = re.sub(r"\s*```$", "", content.strip())
                
                content = content.strip()
                if not content:
                    continue
                    
                try:
                    result = json.loads(content)
                except json.JSONDecodeError:
                    try:
                        fixed_content = re.sub(r',\s*([\]}])', r'\1', content)
                        result = json.loads(fixed_content)
                    except:
                        continue
                
                if result and "error" not in result:
                    logger.info(f"Successfully generated structured output using {provider}")
                    break
                    
            except Exception as e:
                logger.warning(f"Structured output failed for {provider}: {e}. Trying next.")
                continue

        if not result or "error" in result:
            logger.warning("All LLM providers failed for structured output. Falling back to heuristic.")
            if "RESUME TEXT:" in prompt:
                result = self._heuristic_resume_parser(prompt)
            else:
                result = {"error": "AI service temporarily unavailable", "provider": "fallback"}

        if use_cache and result and "error" not in result:
            self.cache[cache_key] = result
        
        # If successful and response_model provided, parse into model
        if response_model and result and "error" not in result:
            try:
                from typing import get_args, get_origin
                if hasattr(response_model, "model_validate"):
                    return response_model.model_validate(result)
                elif hasattr(response_model, "parse_obj"):
                    return response_model.parse_obj(result)
                elif get_origin(response_model) is list:
                    inner_type = get_args(response_model)[0]
                    if hasattr(inner_type, "model_validate"):
                        return [inner_type.model_validate(item) for item in result]
                    elif hasattr(inner_type, "parse_obj"):
                        return [inner_type.parse_obj(item) for item in result]
            except Exception as e:
                logger.warning(f"Failed to parse response to model: {e}")
                
        return result

    def _heuristic_resume_parser(self, prompt: str) -> Dict[str, Any]:
        """A sophisticated heuristic parser for resumes when LLM is unavailable."""
        parts = prompt.split("RESUME TEXT:")
        if len(parts) < 2:
            return {"error": "Could not find resume text"}

        text = parts[1].strip()
        # FIX: Guard against completely empty resume text
        if not text:
            return {"error": "Resume text is empty"}

        lines = [l.strip() for l in text.split("\n") if l.strip()]

        def extract_email(t):
            m = re.search(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b", t)
            return m.group(0) if m else ""

        def extract_phone(t):
            m = re.search(
                r"\b(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b", t
            )
            return m.group(0) if m else ""

        def extract_linkedin(t):
            m = re.search(r"linkedin\.com/in/[a-zA-Z0-9_-]+", t)
            return m.group(0) if m else ""

        def extract_github(t):
            m = re.search(r"github\.com/[a-zA-Z0-9_-]+", t)
            return m.group(0) if m else ""

        # Smarter name and title extraction
        name = "Unknown"
        title = ""
        
        # Skip common generic headers for name
        start_idx = 0
        while start_idx < len(lines):
            l = lines[start_idx].lower()
            if any(h in l for h in ["faculty profile", "curriculum vitae", "resume", "cv", "bio-data"]):
                start_idx += 1
                continue
            break
            
        if start_idx < len(lines):
            # Clean "Name: " prefix if present (separator is optional)
            name = re.sub(r"^(name|full name)\s*[:\-]?\s*", "", lines[start_idx], flags=re.IGNORECASE).strip()
            
            if start_idx + 1 < len(lines):
                second_line = lines[start_idx + 1]
                if not any(
                    x in second_line.lower()
                    for x in ["@", "phone", "linkedin", "github", "|", "+"]
                ):
                    # Clean "Title: " or "Designation: " prefix if present (separator is optional)
                    title = re.sub(r"^(title|designation|role)\s*[:\-]?\s*", "", second_line, flags=re.IGNORECASE).strip()

        sections: Dict[str, Any] = {
            "full_name": name,
            "title": title,
            "email": extract_email(text),
            "phone": extract_phone(text),
            "linkedin_url": extract_linkedin(text),
            "github_url": extract_github(text),
            "summary": "",
            "education": [],
            "experience": [],
            "projects": [],
            "skills": [],
        }

        current_section = None

        summary_headers = ["professional summary", "summary", "about me", "profile"]
        experience_headers = [
            "professional experience",
            "experience",
            "work history",
            "employment history",
            "academic experience",
            "industrial experience",
            "work experience",
        ]
        education_headers = ["education", "academic background", "qualification", "academic profile"]
        skills_headers = [
            "technical skills",
            "skills",
            "expertise",
            "competencies",
            "languages",
            "skills expertise",
        ]
        project_headers = ["key projects", "projects", "personal projects", "technical projects"]

        all_section_headers = (
            summary_headers
            + experience_headers
            + education_headers
            + skills_headers
            + project_headers
        )

        for i, line in enumerate(lines):
            lower_line = line.lower()
            # Remove leading numbers/dots and trailing colons/dots
            clean_line = re.sub(r"^[0-9.]+\s*", "", lower_line).strip()
            clean_line = re.sub(r"[:.]$", "", clean_line).strip()
            # Final header match (no punctuation)
            clean_header = re.sub(r"[^\w\s]", "", clean_line).strip()

            def matches(headers):
                return any(h == clean_header for h in headers) or any(h in clean_header and len(h) > 5 for h in headers)

            # Determine which section this header belongs to
            is_header = False
            if matches(all_section_headers) or (
                line.isupper() and len(line.split()) < 5 and len(line) > 3
            ):
                if matches(summary_headers):
                    current_section = "summary"
                    is_header = True
                elif matches(experience_headers):
                    current_section = "experience"
                    is_header = True
                elif matches(education_headers):
                    current_section = "education"
                    is_header = True
                elif matches(skills_headers):
                    current_section = "skills"
                    is_header = True
                elif matches(project_headers):
                    current_section = "projects"
                    is_header = True

            if is_header:
                continue

            if current_section == "summary":
                sections["summary"] += " " + line

            elif current_section == "experience":
                # FIX: Avoid IndexError — only use lines[i-1] when i > 0
                prev_line_upper = (i > 0) and lines[i - 1].isupper()
                if len(line.split()) < 8 and (
                    re.search(r"\d{4}", line) or i == 0 or prev_line_upper
                ):
                    sections["experience"].append(
                        {
                            "company": line,
                            "role": "Professional",
                            "description": "",
                            "location": "Remote" if "remote" in lower_line else "",
                            "start_date": "",
                            "end_date": "",
                        }
                    )
                    dates = re.findall(
                        r"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{4}|\d{4}",
                        line,
                    )
                    if dates:
                        sections["experience"][-1]["start_date"] = dates[0]
                        if len(dates) > 1:
                            sections["experience"][-1]["end_date"] = dates[1]
                        elif "present" in lower_line:
                            sections["experience"][-1]["end_date"] = "Present"
                elif sections["experience"]:
                    entry = sections["experience"][-1]
                    if not entry["description"] and len(line.split()) < 5:
                        entry["role"] = line
                    else:
                        sep = "\n- " if (line.startswith("-") or line.startswith("•")) else " "
                        entry["description"] += sep + line

            elif current_section == "education":
                # More lenient education entry detection
                is_new_entry = False
                if len(line.split()) < 10 and (re.search(r"\d{4}", line) or i == 0 or line.isupper()):
                    is_new_entry = True
                elif not sections["education"] and len(line.split()) < 8:
                    is_new_entry = True
                
                if is_new_entry:
                    sections["education"].append(
                        {"school": line, "degree": "", "start_date": "", "end_date": ""}
                    )
                    dates = re.findall(r"\d{4}", line)
                    if dates:
                        sections["education"][-1]["start_date"] = dates[0]
                        if len(dates) > 1:
                            sections["education"][-1]["end_date"] = dates[1]
                elif sections["education"]:
                    edu = sections["education"][-1]
                    if not edu["degree"]:
                        edu["degree"] = line
                    else:
                        edu["degree"] += " " + line

            elif current_section == "skills":
                clean_line = re.sub(r"^[•\-\*]\s*", "", line)
                if ":" in clean_line:
                    clean_line = clean_line.split(":", 1)[1]
                skill_parts = re.split(r"[,|•\t]", clean_line)
                for s in skill_parts:
                    s = s.strip()
                    if s and len(s) < 40:
                        sections["skills"].append({"name": s, "category": "Technical"})

            elif current_section == "projects":
                if len(line.split()) < 6 and not line.startswith("-"):
                    sections["projects"].append({"project_name": line, "description": ""})
                elif sections["projects"]:
                    sections["projects"][-1]["description"] += " " + line

        # Cleanup whitespace
        sections["summary"] = sections["summary"].strip()
        for exp in sections["experience"]:
            exp["description"] = exp["description"].strip()
            # FIX: Only attempt role/company swap when description has content
            if (
                exp["role"] == exp["company"]
                and exp["description"]
                and "\n" in exp["description"]
            ):
                desc_lines = exp["description"].split("\n")
                exp["role"] = desc_lines[0].strip("- ")
                exp["description"] = "\n".join(desc_lines[1:]).strip()

        # Validate if the result is meaningful
        meaningful_sections = sum(1 for k in ["summary", "education", "experience", "projects", "skills"] if sections[k])
        if sections["full_name"] == "Unknown" and meaningful_sections == 0:
            return {"error": "Heuristic parser failed to extract meaningful data"}

        return sections

    def _heuristic_resume_optimizer(self, prompt: str) -> Dict[str, Any]:
        """A heuristic optimizer for resumes when LLM is unavailable."""
        logger.warning("Using heuristic resume optimizer fallback - results will be generic.")
        try:
            # Extract resume data from prompt - handle multiple format variations
            data_str = None
            
            # Try format 1: "RESUME DATA:\n..."
            data_str = re.search(r"RESUME DATA:\n(.*?)\n\nReturn a", prompt, re.DOTALL)
            
            # Try format 2: "RESUME JSON:\n..." (used by current optimizer)
            if not data_str:
                data_str = re.search(r"RESUME JSON:\n(\{.*?\})\n\n", prompt, re.DOTALL)
            
            # Try format 3: Look for the JSON object directly after common prompts
            if not data_str:
                # Find the JSON that starts with { after "RESUME" keywords
                json_match = re.search(r'(?:RESUME[^:]*:\s*\n)?(\{[^{}]*"full_name"[^{}]*\})', prompt, re.DOTALL)
                if json_match:
                    # Try to parse it to verify it's valid JSON
                    try:
                        test_parse = json.loads(json_match.group(1))
                        data_str = type('obj', (object,), {'group': lambda: json_match.group(1)})()
                    except json.JSONDecodeError:
                        pass
            
            if not data_str:
                logger.warning("Could not extract resume data from prompt for heuristic optimization")
                # Return a graceful fallback response instead of error
                return {
                    "optimized_resume": {
                        "title": "Optimized Resume",
                        "summary": "Results-driven professional with a proven track record of delivering high-quality solutions. Expert in leveraging industry-standard tools and methodologies to optimize performance and achieve organizational goals.",
                        "experience": [],
                        "skills": ["Project Management", "Team Collaboration", "Problem Solving"]
                    },
                    "optimization_summary": "The resume was optimized for better ATS readability and professional tone (Heuristic Mode).",
                    "improvements_made": ["Generated a professional summary focused on core competencies.", "Added essential industry keywords to skills section."],
                    "projected_ats_score": 75,
                    "compatibility_score": 70,
                    "compatibility_feedback": "Using rule-based optimization. Configure LLM API keys for personalized AI enhancements.",
                    "status": "Fallback",
                    "is_fallback": True
                }
            
            resume_data = json.loads(data_str.group(1))
            optimized_resume = json.loads(json.dumps(resume_data)) # Deep copy
            
            # Extract job description if present
            job_desc = ""
            jd_match = re.search(r"TARGET JOB DESCRIPTION:\n(.*?)\n\nGOALS:", prompt, re.DOTALL)
            if jd_match:
                job_desc = jd_match.group(1).strip()
            
            improvements = []
            
            # 1. Optimize Summary - More realistic improvement
            summary = optimized_resume.get("summary", "")
            if summary:
                # Add professional tone and action verbs to existing summary
                optimized_resume["summary"] = f"Accomplished professional with experience in {resume_data.get('target_role', 'their field')}. {summary}"
                improvements.append("Enhanced professional summary with stronger action verbs (Heuristic).")
            else:
                optimized_resume["summary"] = f"Results-driven professional with a proven track record of delivering high-quality solutions in {resume_data.get('target_role', 'their field')}."
                improvements.append("Generated a new professional summary focused on core competencies (Heuristic).")
            
            # 2. Optimize Experience
            if "experience" in optimized_resume:
                for exp in optimized_resume["experience"]:
                    # Basic language improvement
                    if exp.get("description"):
                        desc = exp["description"]
                        # Replace weak verbs with strong ones
                        verb_map = {
                            "responsible for": "Spearheaded",
                            "worked on": "Engineered",
                            "helped with": "Collaborated on",
                            "managed": "Orchestrated",
                            "did": "Executed",
                            "made": "Developed"
                        }
                        
                        for weak, strong in verb_map.items():
                            if weak in desc.lower():
                                desc = re.sub(rf"\b{weak}\b", strong, desc, flags=re.IGNORECASE)
                        
                        exp["description"] = desc
                
                improvements.append("Refined experience bullet points to emphasize achievements (Heuristic).")

            # 3. Optimize Skills
            if "skills" in optimized_resume:
                essential_skills = ["Project Management", "Team Collaboration", "Problem Solving"]
                for s in essential_skills:
                    if s.lower() not in [str(sk).lower() for sk in optimized_resume["skills"]]:
                        optimized_resume["skills"].append(s)
                
                improvements.append("Expanded skills section with essential industry competencies (Heuristic).")

            return {
                "optimized_resume": optimized_resume,
                "optimization_summary": "The resume was optimized using rule-based heuristics for better readability and tone.",
                "improvements_made": list(dict.fromkeys(improvements))[:5],
                "projected_ats_score": 75,
                "compatibility_score": 70,
                "compatibility_feedback": "Rule-based optimization applied. Configure API keys for full AI power.",
                "status": "Fallback",
                "is_fallback": True
            }
        except Exception as e:
            logger.error(f"Heuristic optimization failed: {e}")
            return {"error": str(e)}

    def _heuristic_ats_analyzer(self, prompt: str) -> Dict[str, Any]:
        """A heuristic analyzer for ATS when LLM is unavailable."""
        logger.warning("Using heuristic ATS analyzer fallback - results will be generic.")
        return {
            "overall_score": 65,
            "keyword_optimization_score": 60,
            "semantic_relevance_score": 65,
            "industry_alignment_score": 70,
            "formatting_score": 80,
            "structure_score": 75,
            "readability_score": 70,
            "contact_info_score": 90,
            "presence_score": 60,
            "education_score": 80,
            "experience_score": 70,
            "skills_score": 75,
            "issues": [
                "Using heuristic analysis. Deep AI insights unavailable without API keys.",
                "Content could benefit from more quantifiable achievements.",
                "Standard section headers detected, but semantic depth is limited."
            ],
            "recommendations": [
                "Configure an LLM provider (Gemini/OpenAI) for personalized analysis.",
                "Add more metrics and percentages to your experience bullet points.",
                "Ensure all technical skills are explicitly listed in the skills section."
            ],
            "skill_analysis": {
                "hard_skills": ["Python", "JavaScript", "SQL", "Git"],
                "soft_skills": ["Communication", "Teamwork", "Problem Solving"],
                "tools": ["Git", "VS Code", "Linux"]
            },
            "keyword_gap": {
                "matched": [],
                "missing": ["Configure API Keys for keyword analysis"],
                "optional": []
            },
            "industry_tips": [
                "Tailor your resume keywords to match job descriptions.",
                "Include quantifiable achievements that demonstrate business impact.",
                "Add relevant certifications and continuous learning initiatives."
            ],
            "llm_powered": False,
            "is_fallback": True,
            "status": "Fallback"
        }

    def generate_embeddings(
        self,
        texts: Union[str, List[str]],
        model: Optional[str] = None,
        **kwargs
    ) -> List[EmbeddingResponse]:
        """
        Generate embeddings for text(s).

        Args:
            texts: Single text or list of texts
            model: Embedding model name

        Returns:
            List of embedding responses
        """
        if isinstance(texts, str):
            texts = [texts]

        # FIX: Guard against empty input list
        if not texts:
            logger.warning("generate_embeddings called with empty text list.")
            return []

        # FIX: Filter out blank strings to avoid wasted API calls / crashes
        texts = [t for t in texts if t and t.strip()]
        if not texts:
            logger.warning("generate_embeddings: all texts were empty after filtering.")
            return []

        provider = kwargs.get("provider", "huggingface")

        if provider == "openai" and self.openai_client:
            return self._embeddings_openai(texts, model or "text-embedding-3-small")
        else:
            return self._embeddings_huggingface(
                texts, model or "sentence-transformers/all-MiniLM-L6-v2"
            )

    async def generate_embeddings_async(
        self,
        texts: Union[str, List[str]],
        model: Optional[str] = None,
        **kwargs
    ) -> List[EmbeddingResponse]:
        """
        Generate embeddings for text(s) asynchronously.
        """
        if isinstance(texts, str):
            texts = [texts]

        if not texts:
            return []

        texts = [t for t in texts if t and t.strip()]
        if not texts:
            return []

        provider = kwargs.get("provider", "huggingface")

        if provider == "openai" and self.async_openai_client:
            return await self._embeddings_openai_async(texts, model or "text-embedding-3-small")
        else:
            # Run CPU intensive HF embedding in a thread pool
            import asyncio
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(
                None, 
                lambda: self._embeddings_huggingface(texts, model or "sentence-transformers/all-MiniLM-L6-v2")
            )

    async def _embeddings_openai_async(self, texts: List[str], model: str) -> List[EmbeddingResponse]:
        """Generate embeddings using OpenAI asynchronously."""
        try:
            response = await self.async_openai_client.embeddings.create(
                model=model,
                input=texts,
            )
        except Exception as e:
            logger.error("OpenAI async embeddings failed: %s", e)
            return []

        return [
            EmbeddingResponse(embedding=data.embedding, model=model, provider="openai")
            for data in response.data
        ]

    def _embeddings_openai(self, texts: List[str], model: str) -> List[EmbeddingResponse]:
        """Generate embeddings using OpenAI."""
        # FIX: Wrap in try/except — API errors should not propagate as uncaught exceptions
        try:
            response = self.openai_client.embeddings.create(
                model=model,
                input=texts,
            )
        except Exception as e:
            logger.error("OpenAI embeddings failed: %s", e)
            return []

        return [
            EmbeddingResponse(embedding=data.embedding, model=model, provider="openai")
            for data in response.data
        ]

    def _embeddings_huggingface(self, texts: List[str], model: str) -> List[EmbeddingResponse]:
        """Generate embeddings using HuggingFace."""
        try:
            from sentence_transformers import SentenceTransformer
            
            # Use singleton for model to avoid reloading
            if not hasattr(self, "_hf_model_cache"):
                self._hf_model_cache = {}
            
            if model not in self._hf_model_cache:
                logger.info(f"Loading HuggingFace model: {model}")
                self._hf_model_cache[model] = SentenceTransformer(model)
            
            hf_model = self._hf_model_cache[model]
            embeddings = hf_model.encode(texts, convert_to_numpy=True)

            return [
                EmbeddingResponse(
                    embedding=embedding.tolist(), model=model, provider="huggingface"
                )
                for embedding in embeddings
            ]
        except Exception as e:
            logger.error("Failed to generate HuggingFace embeddings: %s", e)
            return []

    def stream_response(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> Generator[str, None, None]:
        """
        Stream LLM response token by token.

        Args:
            prompt: User prompt
            system_prompt: System instructions

        Yields:
            Text chunks
        """
        # FIX: Validate prompt before streaming
        if not prompt or not prompt.strip():
            logger.error("stream_response called with empty prompt.")
            return

        provider = kwargs.get("provider", self.default_provider)

        if provider == "openai" and self.openai_client:
            yield from self._stream_openai(prompt, system_prompt, **kwargs)
        elif provider == "google" and self.google_client:
            yield from self._stream_google(prompt, system_prompt, **kwargs)
        else:
            yield "Streaming not available. Please configure an LLM provider."

    def _stream_openai(
        self,
        prompt: str,
        system_prompt: Optional[str],
        **kwargs
    ) -> Generator[str, None, None]:
        """Stream using OpenAI."""
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        # FIX: Wrap streaming in try/except — mid-stream errors should be surfaced cleanly
        try:
            response = self.openai_client.chat.completions.create(
                model=kwargs.get("model", self.openai_model),
                messages=messages,
                temperature=kwargs.get("temperature", 0.7),
                stream=True,
            )
            for chunk in response:
                delta_content = chunk.choices[0].delta.content
                if delta_content:
                    yield delta_content
        except Exception as e:
            logger.error("OpenAI streaming failed: %s", e)
            yield "[Streaming error. Please try again.]"

    def _stream_google(
        self,
        prompt: str,
        system_prompt: Optional[str],
        **kwargs
    ) -> Generator[str, None, None]:
        """Stream using Google Gemini."""
        full_prompt = f"{system_prompt}\n\n{prompt}" if system_prompt else prompt

        # FIX: Wrap streaming in try/except — mid-stream errors should be surfaced cleanly
        try:
            response = self.google_client.models.generate_content_stream(
                model=kwargs.get("model", self.google_model),
                contents=full_prompt,
            )
            for chunk in response:
                # FIX: Guard against None .text on safety-blocked chunks
                if chunk.text:
                    yield chunk.text
        except Exception as e:
            logger.error("Google Gemini streaming failed: %s", e)
            yield "[Streaming error. Please try again.]"


# Singleton instance
llm_service = LLMService()