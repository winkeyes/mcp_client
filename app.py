"""REST API that uses an AI model to call tools on the City Services MCP server."""

from contextlib import AsyncExitStack, asynccontextmanager
import json
import logging
import os
from pathlib import Path
from typing import Annotated, Any

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, StringConstraints
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client


load_dotenv(Path(__file__).resolve().parent / ".env")

MCP_SERVER_URL = os.getenv("MCP_SERVER_URL", "http://127.0.0.1:8090/mcp")
AI_PROVIDER = os.getenv("AI_PROVIDER", "ollama").strip().lower()
AI_BASE_URL = os.getenv(
    "AI_BASE_URL",
    os.getenv("OLLAMA_URL", "http://127.0.0.1:11434"),
).rstrip("/")
AI_MODEL = os.getenv("AI_MODEL", os.getenv("OLLAMA_MODEL", "llama3.2"))
AI_API_KEY = os.getenv("AI_API_KEY", "")
if AI_PROVIDER not in {"ollama", "openai-compatible"}:
    raise ValueError(
        "AI_PROVIDER must be 'ollama' or 'openai-compatible'."
    )
MAX_TOOL_ROUNDS = 5
logger = logging.getLogger(__name__)


class QueryRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2000)


class QueryResponse(BaseModel):
    answer: str
    tool_calls: list[dict[str, Any]]


SchoolName = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=200)]
FAQQuestion = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=2000)]
FAQAnswer = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=5000)]


class SchoolFAQAddRequest(BaseModel):
    school_name: SchoolName
    question: FAQQuestion
    answer: FAQAnswer


class SchoolFAQQueryRequest(BaseModel):
    school_name: SchoolName
    question: FAQQuestion
    limit: int = Field(default=3, ge=1, le=10)


class MCPConnection:
    def __init__(self) -> None:
        self._exit_stack: AsyncExitStack | None = None
        self.session: ClientSession | None = None

    async def connect(self) -> None:
        self._exit_stack = AsyncExitStack()
        try:
            read_stream, write_stream, _ = await self._exit_stack.enter_async_context(
                streamable_http_client(MCP_SERVER_URL)
            )
            self.session = await self._exit_stack.enter_async_context(
                ClientSession(read_stream, write_stream)
            )
            await self.session.initialize()
        except Exception:
            await self._exit_stack.aclose()
            self._exit_stack = None
            self.session = None
            raise

    async def close(self) -> None:
        if self._exit_stack is not None:
            await self._exit_stack.aclose()
            self._exit_stack = None
        self.session = None

    async def list_tools(self) -> list[dict[str, Any]]:
        if self.session is None:
            raise RuntimeError("MCP connection is not initialized")
        response = await self.session.list_tools()
        tools = [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description or "",
                    "parameters": tool.inputSchema,
                },
            }
            for tool in response.tools
        ]
        logger.info("Available MCP tools: %s", tools)
        return tools

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> Any:
        if self.session is None:
            raise RuntimeError("MCP connection is not initialized")
        result = await self.session.call_tool(name, arguments)
        structured_content = getattr(result, "structuredContent", None)
        if structured_content is not None:
            if (
                isinstance(structured_content, dict)
                and set(structured_content) == {"result"}
            ):
                return structured_content["result"]
            return structured_content

        for content in result.content:
            text = getattr(content, "text", None)
            if text is not None:
                try:
                    return json.loads(text)
                except json.JSONDecodeError:
                    return text
        return None


async def ai_chat(
    messages: list[dict[str, Any]], tools: list[dict[str, Any]]
) -> dict[str, Any]:
    headers = {"Authorization": f"Bearer {AI_API_KEY}"} if AI_API_KEY else {}
    async with httpx.AsyncClient(timeout=120) as client:
        if AI_PROVIDER == "ollama":
            response = await client.post(
                f"{AI_BASE_URL}/api/chat",
                json={
                    "model": AI_MODEL,
                    "messages": messages,
                    "tools": tools,
                    "stream": False,
                },
            )
            response.raise_for_status()
            return response.json()["message"]

        response = await client.post(
            f"{AI_BASE_URL}/chat/completions",
            headers=headers,
            json={
                "model": AI_MODEL,
                "messages": messages,
                "tools": tools,
            },
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]


@asynccontextmanager
async def lifespan(app: FastAPI):
    connection = MCPConnection()
    try:
        await connection.connect()
    except Exception:
        await connection.close()
        raise
    app.state.mcp = connection
    yield
    await connection.close()


app = FastAPI(
    title="AI MCP Client",
    description="Ask questions in natural language and use City Services MCP tools.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok", "provider": AI_PROVIDER, "model": AI_MODEL}


@app.post("/school-faqs")
async def add_school_faq(request: SchoolFAQAddRequest) -> Any:
    """Store a school-specific FAQ through the MCP server."""
    connection: MCPConnection = app.state.mcp
    return await connection.call_tool("add_school_faq", request.model_dump())


@app.post("/school-faqs/query")
async def query_school_faq(request: SchoolFAQQueryRequest) -> Any:
    """Retrieve school-specific FAQ matches through the MCP server."""
    connection: MCPConnection = app.state.mcp
    return await connection.call_tool("query_school_faq", request.model_dump())


@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest) -> QueryResponse:
    connection: MCPConnection = app.state.mcp
    tools = await connection.list_tools()
    messages: list[dict[str, Any]] = [
        {
            "role": "system",
            "content": (
                "You are a city services and stored-knowledge assistant. Use the "
                "available tools for live city service, user, or bus data. For "
                "questions asking which users or businesses provide a service in "
                "a city, call get_users_by_service_and_city with service_name and "
                "city. Include landmark only when the user provides one. Do not "
                "use get_services for user or business searches. "
                "questions about a named school's FAQ, call query_school_faq with "
                "that school_name and the user's question; if the school is missing, "
                "ask which school before searching. When the user explicitly asks "
                "to add or save a school FAQ, call add_school_faq with the school "
                "name, question, and answer. Never add an FAQ just to answer a "
                "question. "
                "questions about facts, products, places, or other information that "
                "may be in the knowledge base, call search_knowledge_base before "
                "answering and base your answer only on its results. If it returns "
                "no relevant results, say the information was not found. Never call "
                "add_knowledge to answer a question; use add_knowledge only when the "
                "user explicitly asks to store new information. Do not invent tool "
                "results."
            ),
        },
        {"role": "user", "content": request.query},
    ]
    tool_calls: list[dict[str, Any]] = []

    try:
        for _ in range(MAX_TOOL_ROUNDS):
            message = await ai_chat(messages, tools)
            messages.append(message)
            requested_tools = message.get("tool_calls", [])
            if not requested_tools:
                return QueryResponse(
                    answer=message.get("content", ""), tool_calls=tool_calls
                )

            for requested_tool in requested_tools:
                function = requested_tool["function"]
                name = function["name"]
                arguments = function.get("arguments", {})
                if isinstance(arguments, str):
                    arguments = json.loads(arguments)
                result = await connection.call_tool(name, arguments)
                tool_calls.append({"name": name, "arguments": arguments, "result": result})
                tool_message = {
                    "role": "tool",
                    "content": json.dumps(result),
                }
                if AI_PROVIDER == "openai-compatible":
                    tool_message["tool_call_id"] = requested_tool["id"]
                else:
                    tool_message["tool_name"] = name
                messages.append(tool_message)
    except (httpx.HTTPError, json.JSONDecodeError, KeyError, TypeError) as error:
        raise HTTPException(status_code=502, detail=f"AI tool call failed: {error}") from error

    raise HTTPException(status_code=502, detail="AI provider exceeded the tool-call limit")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="localhost", port=8001)
