#!/usr/bin/env python3
"""Unit tests for all model servers without requiring Docker or model downloads"""

import pytest
from unittest.mock import patch, MagicMock
import sys

# Test Qwen3-4B Server
def test_qwen3_completions_response_format():
    """Test that /v1/completions returns correct format for Qwen3-4B"""
    with patch("qwen3_4b_server.model"), \
         patch("qwen3_4b_server.tokenizer") as mock_tokenizer:

        mock_tokenizer.return_value = MagicMock(input_ids=[[1, 2, 3]])
        mock_tokenizer.decode.return_value = "test response"

        from fastapi.testclient import TestClient
        from qwen3_4b_server import app

        client = TestClient(app)
        response = client.post("/v1/completions", json={
            "prompt": "test prompt",
            "max_tokens": 50
        })

        assert response.status_code == 200
        data = response.json()
        assert "choices" in data, "Missing 'choices' field"
        assert len(data["choices"]) > 0, "Empty choices array"
        assert "text" in data["choices"][0], "Missing 'text' in choices[0]"
        assert data["choices"][0]["text"] == "test response"


def test_qwen3_chat_completions_response_format():
    """Test that /v1/chat/completions returns correct format for Qwen3-4B"""
    with patch("qwen3_4b_server.model"), \
         patch("qwen3_4b_server.tokenizer") as mock_tokenizer:

        mock_tokenizer.return_value = MagicMock(input_ids=[[1, 2, 3]])
        mock_tokenizer.decode.return_value = "chat response"

        from fastapi.testclient import TestClient
        from qwen3_4b_server import app

        client = TestClient(app)
        response = client.post("/v1/chat/completions", json={
            "messages": [{"role": "user", "content": "hello"}],
            "max_tokens": 50
        })

        assert response.status_code == 200
        data = response.json()
        assert "choices" in data
        assert "message" in data["choices"][0]
        assert "content" in data["choices"][0]["message"]


# Test Mistral-7B Server
def test_mistral_completions_response_format():
    """Test that /v1/completions returns correct format for Mistral-7B"""
    with patch("mistral_server.model"), \
         patch("mistral_server.tokenizer") as mock_tokenizer:

        mock_tokenizer.return_value = MagicMock(input_ids=[[1, 2, 3]])
        mock_tokenizer.decode.return_value = "mistral response"

        from fastapi.testclient import TestClient
        from mistral_server import app

        client = TestClient(app)
        response = client.post("/v1/completions", json={
            "prompt": "test prompt",
            "max_tokens": 50
        })

        assert response.status_code == 200
        data = response.json()
        assert "choices" in data, "Missing 'choices' field"
        assert "text" in data["choices"][0], "Missing 'text' in choices[0]"
        assert data["choices"][0]["text"] == "mistral response"


def test_mistral_chat_completions_response_format():
    """Test that /v1/chat/completions returns correct format for Mistral-7B"""
    with patch("mistral_server.model"), \
         patch("mistral_server.tokenizer") as mock_tokenizer:

        mock_tokenizer.return_value = MagicMock(input_ids=[[1, 2, 3]])
        mock_tokenizer.decode.return_value = "mistral chat response"

        from fastapi.testclient import TestClient
        from mistral_server import app

        client = TestClient(app)
        response = client.post("/v1/chat/completions", json={
            "messages": [{"role": "user", "content": "hello"}],
            "max_tokens": 50
        })

        assert response.status_code == 200
        data = response.json()
        assert "choices" in data
        assert "message" in data["choices"][0]
        assert "content" in data["choices"][0]["message"]


# Test Gemma Server
def test_gemma_completions_response_format():
    """Test that /v1/completions returns correct format for Gemma"""
    with patch("gemma_server.model"), \
         patch("gemma_server.processor") as mock_processor:

        mock_processor.apply_chat_template.return_value = MagicMock(
            input_ids=[[1, 2, 3]],
            to=MagicMock(return_value=MagicMock(input_ids=[[1, 2, 3]]))
        )
        mock_processor.decode.return_value = "gemma response"

        from fastapi.testclient import TestClient
        from gemma_server import app

        client = TestClient(app)
        response = client.post("/v1/completions", json={
            "prompt": "test prompt",
            "max_tokens": 50
        })

        assert response.status_code == 200
        data = response.json()
        assert "choices" in data, "Missing 'choices' field"
        assert "text" in data["choices"][0], "Missing 'text' in choices[0]"


# Test Llama Server
def test_llama_completions_response_format():
    """Test that /v1/completions returns correct format for Llama"""
    with patch("llama_server.model"), \
         patch("llama_server.tokenizer") as mock_tokenizer:

        mock_tokenizer.return_value = MagicMock(input_ids=[[1, 2, 3]])
        mock_tokenizer.decode.return_value = "llama response"

        from fastapi.testclient import TestClient
        from llama_server import app

        client = TestClient(app)
        response = client.post("/v1/completions", json={
            "prompt": "test prompt",
            "max_tokens": 50
        })

        assert response.status_code == 200
        data = response.json()
        assert "choices" in data, "Missing 'choices' field"
        assert "text" in data["choices"][0], "Missing 'text' in choices[0]"


def test_llama_chat_completions_response_format():
    """Test that /v1/chat/completions returns correct format for Llama"""
    with patch("llama_server.model"), \
         patch("llama_server.tokenizer") as mock_tokenizer:

        mock_tokenizer.return_value = MagicMock(input_ids=[[1, 2, 3]])
        mock_tokenizer.decode.return_value = "llama chat response"

        from fastapi.testclient import TestClient
        from llama_server import app

        client = TestClient(app)
        response = client.post("/v1/chat/completions", json={
            "messages": [{"role": "user", "content": "hello"}],
            "max_tokens": 50
        })

        assert response.status_code == 200
        data = response.json()
        assert "choices" in data
        assert "message" in data["choices"][0]
        assert "content" in data["choices"][0]["message"]


# Test client response parsing
def test_client_response_parsing():
    """Test that client.py can parse both response formats"""
    from sway_harness.client import get_completion

    # Mock the requests.post to return both formats
    with patch("sway_harness.client.requests.post") as mock_post:
        # Test /v1/completions format
        mock_post.return_value.status_code = 200
        mock_post.return_value.json.return_value = {
            "choices": [{"text": "test response"}]
        }

        result = get_completion(
            messages=[{"role": "user", "content": "test"}],
            base_url="http://localhost:8100",
            model_path="test-model"
        )
        assert result == "test response"

        # Test /v1/chat/completions format
        mock_post.return_value.json.return_value = {
            "choices": [{"message": {"content": "chat response"}}]
        }

        result = get_completion(
            messages=[{"role": "user", "content": "test"}],
            base_url="http://localhost:8100",
            model_path="test-model"
        )
        assert result == "chat response"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
