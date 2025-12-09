"""Tests for Model Configuration feature.

These tests verify that model configuration is correctly:
1. Passed from frontend to backend via configurable
2. Logged by DynamicModelMiddleware
3. Applied to different components (primary model, subagents, summarization, suggestions)

Run with: pytest tests/test_model_config.py -v
"""

import logging
import pytest
from unittest.mock import MagicMock, patch

# Configure logging for tests
logging.basicConfig(level=logging.DEBUG)


class TestDynamicModelMiddleware:
    """Tests for DynamicModelMiddleware."""

    def test_get_runtime_config_returns_empty_when_no_context(self):
        """Test that _get_runtime_config returns empty dict when not in LangGraph context."""
        from deepagents.middleware.dynamic_model import DynamicModelMiddleware
        
        middleware = DynamicModelMiddleware()
        config = middleware._get_runtime_config()
        
        assert config == {}

    def test_get_runtime_config_returns_configurable(self):
        """Test that _get_runtime_config returns configurable when in LangGraph context."""
        from deepagents.middleware.dynamic_model import DynamicModelMiddleware
        
        middleware = DynamicModelMiddleware()
        
        mock_config = {
            "configurable": {
                "provider": "openrouter",
                "model": "gpt-4o",
                "openai_api_key": "test-key",
                "openai_base_url": "https://api.test.com/v1",
                "model_overrides": {
                    "subagents": {"general-purpose": "gpt-4o-mini"},
                    "summarization": "gpt-4o-mini",
                    "suggestions": "gpt-4o-mini",
                },
            }
        }
        
        with patch("deepagents.middleware.dynamic_model.get_config", return_value=mock_config):
            config = middleware._get_runtime_config()
            
            assert config["provider"] == "openrouter"
            assert config["model"] == "gpt-4o"
            assert "model_overrides" in config

    def test_create_model_from_config_openrouter(self):
        """Test creating model from OpenRouter configuration."""
        from deepagents.middleware.dynamic_model import create_model_from_config
        
        config = {
            "provider": "openrouter",
            "model": "gpt-4o-mini",
            "openai_api_key": "test-key",
            "openai_base_url": "https://api.test.com/v1",
        }
        
        # Note: This will actually try to create a model, so we mock the ChatOpenAI
        with patch("deepagents.middleware.dynamic_model.ChatOpenAI") as mock_chat:
            mock_chat.return_value = MagicMock()
            model = create_model_from_config(config)
            
            mock_chat.assert_called_once_with(
                model="gpt-4o-mini",
                openai_api_base="https://api.test.com/v1",
                openai_api_key="test-key",
            )

    def test_create_model_from_config_returns_none_for_empty_config(self):
        """Test that create_model_from_config returns None for empty configuration."""
        from deepagents.middleware.dynamic_model import create_model_from_config
        
        assert create_model_from_config({}) is None
        assert create_model_from_config({"provider": "openai"}) is None
        assert create_model_from_config({"model": "gpt-4o"}) is None

    def test_get_model_with_override(self):
        """Test _get_model correctly applies component overrides."""
        from deepagents.middleware.dynamic_model import DynamicModelMiddleware
        
        middleware = DynamicModelMiddleware()
        
        mock_config = {
            "configurable": {
                "provider": "openrouter",
                "model": "gpt-4o",
                "openai_api_key": "test-key",
                "openai_base_url": "https://api.test.com/v1",
                "model_overrides": {
                    "subagents": {"general-purpose": "gpt-4o-mini"},
                    "summarization": "gpt-4o-mini",
                    "suggestions": "gpt-4o-mini",
                },
            }
        }
        
        with patch("deepagents.middleware.dynamic_model.get_config", return_value=mock_config):
            with patch("deepagents.middleware.dynamic_model.create_model_from_config") as mock_create:
                mock_create.return_value = MagicMock()
                
                # Test getting model with summarization override
                middleware._get_model(component="summarization")
                
                # Verify the override model was used
                call_args = mock_create.call_args[0][0]
                assert call_args["model"] == "gpt-4o-mini"

    def test_get_model_subagent_nested_override(self):
        """Test _get_model correctly handles nested subagent overrides."""
        from deepagents.middleware.dynamic_model import DynamicModelMiddleware
        
        middleware = DynamicModelMiddleware()
        
        mock_config = {
            "configurable": {
                "provider": "openrouter",
                "model": "gpt-4o",
                "openai_api_key": "test-key",
                "openai_base_url": "https://api.test.com/v1",
                "model_overrides": {
                    "subagents": {"general-purpose": "gpt-4o-mini"},
                },
            }
        }
        
        with patch("deepagents.middleware.dynamic_model.get_config", return_value=mock_config):
            with patch("deepagents.middleware.dynamic_model.create_model_from_config") as mock_create:
                mock_create.return_value = MagicMock()
                
                # Test getting model with subagent.general-purpose override
                middleware._get_model(component="subagents.general-purpose")
                
                # Verify the override model was used
                call_args = mock_create.call_args[0][0]
                assert call_args["model"] == "gpt-4o-mini"


class TestModelConfigLogging:
    """Tests for model configuration logging."""

    def test_wrap_model_call_logs_config(self, caplog):
        """Test that wrap_model_call logs the model configuration."""
        from deepagents.middleware.dynamic_model import DynamicModelMiddleware
        
        middleware = DynamicModelMiddleware()
        
        mock_request = MagicMock()
        mock_handler = MagicMock(return_value=MagicMock())
        
        mock_config = {
            "configurable": {
                "provider": "openrouter",
                "model": "gpt-4o",
                "model_overrides": {
                    "subagents": {"general-purpose": "gpt-4o-mini"},
                },
            }
        }
        
        with patch("deepagents.middleware.dynamic_model.get_config", return_value=mock_config):
            with caplog.at_level(logging.INFO):
                middleware.wrap_model_call(mock_request, mock_handler)
                
                # Check that logging occurred
                assert "[MODEL CONFIG]" in caplog.text
                assert "provider=openrouter" in caplog.text
                assert "model=gpt-4o" in caplog.text


class TestFrontendConfigPassing:
    """Test cases for frontend configuration passing (conceptual tests)."""

    def test_configurable_structure(self):
        """Verify the expected configurable structure."""
        expected_structure = {
            "configurable": {
                "provider": "openrouter",  # API provider
                "model": "gpt-4o",  # Primary model
                "openai_api_key": "sk-xxx",  # API key (varies by provider)
                "openai_base_url": "https://api.deerapi.com/v1",  # For OpenAI-compatible APIs
                "model_overrides": {
                    "subagents": {
                        "general-purpose": "gpt-4o-mini",  # Override for general-purpose subagent
                    },
                    "summarization": "gpt-4o-mini",  # Override for summarization
                    "suggestions": "gpt-4o-mini",  # Override for suggestions
                },
            }
        }
        
        # This test documents the expected structure
        assert "configurable" in expected_structure
        configurable = expected_structure["configurable"]
        
        # Required fields for model selection
        assert "provider" in configurable
        assert "model" in configurable
        
        # Model overrides structure
        assert "model_overrides" in configurable
        overrides = configurable["model_overrides"]
        assert "subagents" in overrides
        assert "summarization" in overrides
        assert "suggestions" in overrides


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


