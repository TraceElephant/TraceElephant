#!/usr/bin/env python3
"""
LLM API Middleware Server
Intercepts and logs all OpenAI-compatible API requests and responses.
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import httpx
import yaml
from fastapi import FastAPI, Request, Response
from fastapi.responses import StreamingResponse, JSONResponse
import uvicorn


class LLMMiddleware:
    def __init__(self, config_path: str = "config.yaml"):
        """Initialize the middleware with configuration."""
        self.config = self._load_config(config_path)
        self.upstream_base_url = self.config["upstream"]["base_url"].rstrip("/")
        self.upstream_api_key = self.config["upstream"]["api_key"]
        self.log_dir = Path(self.config["logging"]["log_dir"])
        self.logging_enabled = self.config["logging"]["enabled"]
        self.console_output = self.config["logging"]["console_output"]

        # Create log directory if it doesn't exist
        if self.logging_enabled:
            self.log_dir.mkdir(parents=True, exist_ok=True)

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)

    def _save_log(self, request_data: Dict[str, Any], response_data: Any,
                  endpoint: str, status_code: int):
        """Save request and response to a JSON file."""
        if not self.logging_enabled:
            return

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "endpoint": endpoint,
            "status_code": status_code,
            "input": request_data,
            "output": str(response_data) if not isinstance(response_data, (dict, list)) else response_data
        }

        log_file = self.log_dir / f"{timestamp}.json"
        with open(log_file, 'w', encoding='utf-8') as f:
            json.dump(log_entry, f, indent=2, ensure_ascii=False)

        if self.console_output:
            print(f"[{timestamp}] Logged API call to {log_file}")

    async def proxy_request(self, request: Request, endpoint: str) -> Response:
        """Proxy the request to upstream API and log the interaction."""
        # Parse request body
        try:
            body = await request.body()
            request_data = json.loads(body) if body else {}
        except json.JSONDecodeError:
            request_data = {}

        # Prepare headers
        headers = {
            "Authorization": f"Bearer {self.upstream_api_key}",
            "Content-Type": "application/json",
        }

        # Add other headers from original request (excluding host and authorization)
        for key, value in request.headers.items():
            if key.lower() not in ["host", "authorization", "content-length"]:
                headers[key] = value

        # Construct upstream URL
        upstream_url = f"{self.upstream_base_url}{endpoint}"

        # Check if streaming is requested
        is_streaming = request_data.get("stream", False)

        try:
            async with httpx.AsyncClient(timeout=300.0) as client:
                if is_streaming:
                    # Handle streaming response
                    return await self._handle_streaming(
                        client, upstream_url, headers, request_data, endpoint
                    )
                else:
                    # Handle non-streaming response
                    return await self._handle_non_streaming(
                        client, upstream_url, headers, request_data, endpoint
                    )

        except Exception as e:
            error_response = {
                "error": {
                    "message": str(e),
                    "type": "middleware_error",
                    "code": "internal_error"
                }
            }
            self._save_log(request_data, error_response, endpoint, 500)
            return JSONResponse(content=error_response, status_code=500)

    async def _handle_non_streaming(self, client: httpx.AsyncClient, url: str,
                                     headers: Dict[str, str], request_data: Dict[str, Any],
                                     endpoint: str) -> Response:
        """Handle non-streaming API requests."""
        response = await client.post(url, headers=headers, json=request_data)

        # Parse response
        try:
            response_data = response.json()
        except json.JSONDecodeError:
            response_data = response.text

        # Log the interaction
        self._save_log(request_data, response_data, endpoint, response.status_code)

        # Filter out headers that should not be copied
        filtered_headers = {
            k: v for k, v in response.headers.items()
            if k.lower() not in ['content-length', 'content-encoding', 'transfer-encoding']
        }

        # Return response
        return JSONResponse(
            content=response_data,
            status_code=response.status_code,
            headers=filtered_headers
        )

    async def _handle_streaming(self, client: httpx.AsyncClient, url: str,
                                headers: Dict[str, str], request_data: Dict[str, Any],
                                endpoint: str) -> StreamingResponse:
        """Handle streaming API requests."""
        collected_chunks = []

        async def stream_generator():
            async with client.stream("POST", url, headers=headers, json=request_data) as response:
                async for chunk in response.aiter_bytes():
                    collected_chunks.append(chunk)
                    yield chunk

                # After streaming is complete, save the log
                try:
                    # Reconstruct the full response from chunks
                    full_response = b"".join(collected_chunks).decode('utf-8')
                    self._save_log(request_data, full_response, endpoint, response.status_code)
                except Exception as e:
                    if self.console_output:
                        print(f"Error logging streaming response: {e}")

        return StreamingResponse(
            stream_generator(),
            media_type="text/event-stream"
        )


# Initialize FastAPI app and middleware
app = FastAPI(title="LLM API Middleware", version="1.0.0")
middleware = LLMMiddleware()


@app.get("/")
async def root():
    """Health check endpoint."""
    return {
        "status": "running",
        "service": "LLM API Middleware",
        "version": "1.0.0"
    }


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    """Proxy chat completions endpoint."""
    return await middleware.proxy_request(request, "/v1/chat/completions")


@app.post("/v1/completions")
async def completions(request: Request):
    """Proxy completions endpoint."""
    return await middleware.proxy_request(request, "/v1/completions")


@app.post("/v1/embeddings")
async def embeddings(request: Request):
    """Proxy embeddings endpoint."""
    return await middleware.proxy_request(request, "/v1/embeddings")


@app.get("/v1/models")
async def list_models(request: Request):
    """Proxy list models endpoint."""
    return await middleware.proxy_request(request, "/v1/models")


@app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH"])
async def catch_all(request: Request, path: str):
    """Catch-all endpoint for other API routes."""
    endpoint = f"/{path}"
    return await middleware.proxy_request(request, endpoint)


def main():
    """Run the middleware server."""
    config = middleware.config
    host = config["server"]["host"]
    port = config["server"]["port"]

    print(f"Starting LLM API Middleware on {host}:{port}")
    print(f"Proxying to: {middleware.upstream_base_url}")
    print(f"Logs directory: {middleware.log_dir.absolute()}")

    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
