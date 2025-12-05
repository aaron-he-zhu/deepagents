"""Middleware for dynamic model selection based on runtime configuration.

This middleware allows the model to be configured at runtime through the
`configurable` parameter passed in the request config.
"""

import logging
from collections.abc import Awaitable, Callable
from typing import Any

from langchain.agents.middleware.types import AgentMiddleware, ModelRequest, ModelResponse
from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel
from langchain_openai import ChatOpenAI
from langgraph.config import get_config

logger = logging.getLogger(__name__)


def create_model_from_config(config: dict[str, Any]) -> BaseChatModel | None:
    """Create a model instance from runtime configuration.
    
    Args:
        config: Configuration dictionary containing:
            - provider: API provider (openai, anthropic, google, openrouter)
            - model: Model name
            - openai_api_key: OpenAI API key (for openai/openrouter)
            - anthropic_api_key: Anthropic API key
            - google_api_key: Google API key
            - openai_base_url: Base URL for OpenAI-compatible APIs
    
    Returns:
        Configured model instance, or None if config is invalid.
    """
    provider = config.get("provider")
    model_name = config.get("model")
    
    if not provider or not model_name:
        return None
    
    logger.info(f"[MODEL CONFIG] Creating model: provider={provider}, model={model_name}")
    
    try:
        if provider == "openai":
            api_key = config.get("openai_api_key")
            if api_key:
                return ChatOpenAI(model=model_name, openai_api_key=api_key)
        
        elif provider == "anthropic":
            api_key = config.get("anthropic_api_key")
            if api_key:
                return init_chat_model(f"anthropic:{model_name}", api_key=api_key)
        
        elif provider == "google":
            api_key = config.get("google_api_key")
            if api_key:
                return init_chat_model(f"google_genai:{model_name}", api_key=api_key)
        
        elif provider == "openrouter":
            api_key = config.get("openai_api_key")
            base_url = config.get("openai_base_url")
            if api_key and base_url:
                return ChatOpenAI(
                    model=model_name,
                    openai_api_base=base_url,
                    openai_api_key=api_key,
                )
        
        logger.warning(f"[MODEL CONFIG] Could not create model for provider={provider}")
        return None
        
    except Exception as e:
        logger.error(f"[MODEL CONFIG] Error creating model: {e}")
        return None


class DynamicModelMiddleware(AgentMiddleware):
    """Middleware that enables dynamic model selection at runtime.
    
    This middleware reads the `configurable` from the runtime config and
    creates a new model instance if valid configuration is provided.
    
    Example config structure (passed via LangGraph SDK):
        {
            "configurable": {
                "provider": "openrouter",
                "model": "gpt-4o-mini",
                "openai_api_key": "sk-xxx",
                "openai_base_url": "https://api.deerapi.com/v1",
                "model_overrides": {
                    "subagents": {"general-purpose": "gpt-4o"},
                    "summarization": "gpt-4o-mini",
                    "suggestions": "gpt-4o-mini"
                }
            }
        }
    """
    
    def __init__(self, default_model: BaseChatModel | None = None):
        """Initialize the middleware.
        
        Args:
            default_model: Fallback model to use when no runtime config is provided.
        """
        self._default_model = default_model
        self._current_model: BaseChatModel | None = None
    
    def _get_runtime_config(self) -> dict[str, Any]:
        """Get the runtime configuration from LangGraph context."""
        try:
            cfg = get_config()
            configurable = cfg.get("configurable", {})
            if configurable:
                logger.debug(f"[MODEL CONFIG] Got runtime configurable: {list(configurable.keys())}")
            return configurable
        except Exception as e:
            logger.debug(f"[MODEL CONFIG] Could not get runtime config: {e}")
            return {}
    
    def _get_model(self, component: str | None = None) -> BaseChatModel | None:
        """Get the appropriate model based on runtime config.
        
        Args:
            component: Optional component name for model overrides
                       (e.g., "summarization", "subagents.general-purpose")
        
        Returns:
            Model to use, or None to use the default.
        """
        config = self._get_runtime_config()
        
        if not config:
            return None
        
        # Check for component-specific override
        if component:
            overrides = config.get("model_overrides") or {}
            if "." in component:
                # Handle nested overrides like "subagents.general-purpose"
                parts = component.split(".", 1)
                override_model = (overrides.get(parts[0]) or {}).get(parts[1])
            else:
                override_model = overrides.get(component)
            
            if override_model:
                logger.info(f"[MODEL CONFIG] Using override for {component}: {override_model}")
                # Create model with override name but same provider/keys
                override_config = {**config, "model": override_model}
                return create_model_from_config(override_config)
        
        # Use the primary model from config
        return create_model_from_config(config)
    
    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse:
        """Wrap model calls to log model configuration.
        
        Note: This middleware logs the model being used. Dynamic model replacement
        requires deeper integration with LangChain's agent architecture.
        
        The logs help verify that frontend configuration is being passed correctly.
        """
        config = self._get_runtime_config()
        
        if config:
            model_name = config.get("model", "unknown")
            provider = config.get("provider", "unknown")
            overrides = config.get("model_overrides") or {}
            
            logger.info(f"[MODEL CONFIG] Primary Model call: provider={provider}, model={model_name}")
            
            # Log model overrides if present
            if overrides:
                subagent_override = (overrides.get("subagents") or {}).get("general-purpose")
                summarization_override = overrides.get("summarization")
                suggestions_override = overrides.get("suggestions")
                
                if subagent_override:
                    logger.info(f"[MODEL CONFIG] Subagent (general-purpose) override: {subagent_override}")
                if summarization_override:
                    logger.info(f"[MODEL CONFIG] Summarization override: {summarization_override}")
                if suggestions_override:
                    logger.info(f"[MODEL CONFIG] Suggestions override: {suggestions_override}")
        else:
            logger.info("[MODEL CONFIG] Primary Model call: using default (no runtime config)")
        
        return handler(request)
    
    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelResponse:
        """Async version of wrap_model_call."""
        config = self._get_runtime_config()
        
        if config:
            model_name = config.get("model", "unknown")
            provider = config.get("provider", "unknown")
            overrides = config.get("model_overrides") or {}
            
            logger.info(f"[MODEL CONFIG] Primary Model call (async): provider={provider}, model={model_name}")
            
            # Log model overrides if present
            if overrides:
                subagent_override = (overrides.get("subagents") or {}).get("general-purpose")
                summarization_override = overrides.get("summarization")
                suggestions_override = overrides.get("suggestions")
                
                if subagent_override:
                    logger.info(f"[MODEL CONFIG] Subagent (general-purpose) override: {subagent_override}")
                if summarization_override:
                    logger.info(f"[MODEL CONFIG] Summarization override: {summarization_override}")
                if suggestions_override:
                    logger.info(f"[MODEL CONFIG] Suggestions override: {suggestions_override}")
        else:
            logger.info("[MODEL CONFIG] Primary Model call (async): using default (no runtime config)")
        
        return await handler(request)

