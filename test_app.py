from unittest.mock import AsyncMock, Mock, patch

from fastapi.testclient import TestClient

from .app import MCPConnection, MCP_SERVER_URL, app, ai_chat


class FakeConnection:
    async def connect(self):
        return None

    async def close(self):
        return None

    async def list_tools(self):
        return [
            {
                "type": "function",
                "function": {
                    "name": "get_services",
                    "description": "List city services",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ]

    async def call_tool(self, name, arguments):
        assert name == "get_services"
        assert arguments == {}
        return [{"id": "library", "status": "open"}]


def test_mcp_connection_uses_streamable_http_endpoint():
    transport = AsyncMock()
    read_stream = object()
    write_stream = object()
    transport.__aenter__.return_value = (read_stream, write_stream, lambda: None)
    session = AsyncMock()
    session.__aenter__.return_value = session

    with (
        patch(
            "ollama_mcp_client.app.streamable_http_client",
            return_value=transport,
        ) as create_transport,
        patch(
            "ollama_mcp_client.app.ClientSession",
            return_value=session,
        ) as create_session,
    ):
        connection = MCPConnection()
        import asyncio

        asyncio.run(connection.connect())
        create_transport.assert_called_once_with(MCP_SERVER_URL)
        create_session.assert_called_once_with(read_stream, write_stream)
        session.initialize.assert_awaited_once_with()
        asyncio.run(connection.close())

    transport.__aexit__.assert_awaited_once()
    session.__aexit__.assert_awaited_once()


def test_query_executes_mcp_tool_and_returns_answer():
    first_message = {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {"function": {"name": "get_services", "arguments": {}}}
        ],
    }
    final_message = {
        "role": "assistant",
        "content": "The library is open.",
    }

    with (
        patch("ollama_mcp_client.app.MCPConnection", return_value=FakeConnection()),
        patch(
            "ollama_mcp_client.app.ai_chat",
            new=AsyncMock(side_effect=[first_message, final_message]),
        ),
        TestClient(app) as client,
    ):
        response = client.post("/query", json={"query": "Is the library open?"})

    assert response.status_code == 200
    assert response.json()["answer"] == "The library is open."
    assert response.json()["tool_calls"][0]["name"] == "get_services"


def test_openai_compatible_query_returns_tool_result_with_call_id():
    connection = FakeConnection()
    tool_call_message = {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": "call-123",
                "type": "function",
                "function": {"name": "get_services", "arguments": "{}"},
            }
        ],
    }
    answer_message = {"role": "assistant", "content": "The library is open."}
    chat = AsyncMock(side_effect=[tool_call_message, answer_message])

    with (
        patch("ollama_mcp_client.app.MCPConnection", return_value=connection),
        patch("ollama_mcp_client.app.AI_PROVIDER", "openai-compatible"),
        patch("ollama_mcp_client.app.ai_chat", new=chat),
        TestClient(app) as client,
    ):
        response = client.post("/query", json={"query": "Is the library open?"})

    assert response.status_code == 200
    assert response.json()["answer"] == "The library is open."
    assert chat.await_args_list[1].args[0][-2] == {
        "role": "tool",
        "tool_call_id": "call-123",
        "content": '[{"id": "library", "status": "open"}]',
    }


def test_query_routes_service_city_question_to_service_users_tool():
    connection = FakeConnection()
    connection.list_tools = AsyncMock(
        return_value=[
            {
                "type": "function",
                "function": {
                    "name": "get_users_by_service_and_city",
                    "description": "Find users by service, city, and optional landmark.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "service_name": {"type": "string"},
                            "city": {"type": "string"},
                            "landmark": {"type": "string"},
                        },
                        "required": ["service_name", "city"],
                    },
                },
            }
        ]
    )
    arguments = {
        "service_name": "Internet",
        "city": "Sagar",
        "landmark": "BH Road",
    }
    connection.call_tool = AsyncMock(return_value={"users": [{"shopName": "AK automobiles"}]})
    tool_call_message = {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "function": {
                    "name": "get_users_by_service_and_city",
                    "arguments": arguments,
                }
            }
        ],
    }
    answer_message = {
        "role": "assistant",
        "content": "AK automobiles provides Internet service near BH Road in Sagar.",
    }
    chat = AsyncMock(side_effect=[tool_call_message, answer_message])

    with (
        patch("ollama_mcp_client.app.MCPConnection", return_value=connection),
        patch("ollama_mcp_client.app.ai_chat", new=chat),
        TestClient(app) as client,
    ):
        response = client.post(
            "/query",
            json={"query": "Who provides Internet service near BH Road in Sagar?"},
        )

    assert response.status_code == 200
    assert response.json()["tool_calls"][0]["name"] == "get_users_by_service_and_city"
    connection.call_tool.assert_awaited_once_with(
        "get_users_by_service_and_city",
        arguments,
    )
    assert "get_users_by_service_and_city" in chat.await_args_list[0].args[0][0]["content"]
    assert chat.await_args_list[0].args[1][0]["function"]["name"] == (
        "get_users_by_service_and_city"
    )


def test_knowledge_question_uses_search_tool():
    search_connection = FakeConnection()

    async def list_knowledge_tools():
        return [
            {
                "type": "function",
                "function": {
                    "name": "search_knowledge_base",
                    "description": "Search stored knowledge",
                    "parameters": {
                        "type": "object",
                        "properties": {"question": {"type": "string"}},
                    },
                },
            }
        ]

    async def search_knowledge(name, arguments):
        assert name == "search_knowledge_base"
        assert arguments == {"question": "Where can I get carrots?"}
        return {"results": [{"content": "Green Market sells carrots."}]}

    search_connection.list_tools = list_knowledge_tools
    search_connection.call_tool = search_knowledge
    tool_call_message = {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "function": {
                    "name": "search_knowledge_base",
                    "arguments": {"question": "Where can I get carrots?"},
                }
            }
        ],
    }
    answer_message = {
        "role": "assistant",
        "content": "Green Market sells carrots.",
    }
    chat = AsyncMock(side_effect=[tool_call_message, answer_message])

    with (
        patch("ollama_mcp_client.app.MCPConnection", return_value=search_connection),
        patch("ollama_mcp_client.app.ai_chat", new=chat),
        TestClient(app) as client,
    ):
        response = client.post(
            "/query", json={"query": "Where can I get carrots?"}
        )

    assert response.status_code == 200
    assert response.json()["answer"] == "Green Market sells carrots."
    assert response.json()["tool_calls"][0]["name"] == "search_knowledge_base"
    system_prompt = chat.await_args_list[0].args[0][0]["content"]
    assert "search_knowledge_base" in system_prompt
    assert "Never call add_knowledge to answer a question" in system_prompt
    assert "query_school_faq" in system_prompt
    assert "add_school_faq" in system_prompt


def test_add_school_faq_endpoint_sends_typed_fields_to_mcp():
    connection = FakeConnection()
    connection.call_tool = AsyncMock(return_value={"id": 4, "school_name": "North High"})

    with (
        patch("ollama_mcp_client.app.MCPConnection", return_value=connection),
        TestClient(app) as client,
    ):
        response = client.post(
            "/school-faqs",
            json={
                "school_name": " North High ",
                "question": " When does the library close? ",
                "answer": " At 6 PM. ",
            },
        )

    assert response.status_code == 200
    assert response.json()["id"] == 4
    connection.call_tool.assert_awaited_once_with(
        "add_school_faq",
        {
            "school_name": "North High",
            "question": "When does the library close?",
            "answer": "At 6 PM.",
        },
    )


def test_query_school_faq_endpoint_sends_school_and_question_to_mcp():
    connection = FakeConnection()
    connection.call_tool = AsyncMock(
        return_value={"school_name": "North High", "results": [{"answer": "At 6 PM."}]}
    )

    with (
        patch("ollama_mcp_client.app.MCPConnection", return_value=connection),
        TestClient(app) as client,
    ):
        response = client.post(
            "/school-faqs/query",
            json={"school_name": "North High", "question": "When does the library close?"},
        )

    assert response.status_code == 200
    assert response.json()["results"][0]["answer"] == "At 6 PM."
    connection.call_tool.assert_awaited_once_with(
        "query_school_faq",
        {
            "school_name": "North High",
            "question": "When does the library close?",
            "limit": 3,
        },
    )


def test_query_rejects_empty_query():
    with (
        patch("ollama_mcp_client.app.MCPConnection", return_value=FakeConnection()),
        TestClient(app) as client,
    ):
        response = client.post("/query", json={"query": ""})

    assert response.status_code == 422


def test_ai_chat_uses_openai_compatible_endpoint_and_bearer_key():
    import asyncio

    response = Mock()
    response.json.return_value = {
        "choices": [{"message": {"role": "assistant", "content": "Hello"}}]
    }
    response.raise_for_status = Mock()
    client = AsyncMock()
    client.__aenter__.return_value = client
    client.post.return_value = response

    with (
        patch("ollama_mcp_client.app.AI_PROVIDER", "openai-compatible"),
        patch("ollama_mcp_client.app.AI_BASE_URL", "https://api.example.com/v1"),
        patch("ollama_mcp_client.app.AI_MODEL", "paid-model"),
        patch("ollama_mcp_client.app.AI_API_KEY", "test-key"),
        patch("ollama_mcp_client.app.httpx.AsyncClient", return_value=client),
    ):
        message = asyncio.run(ai_chat([{"role": "user", "content": "Hi"}], []))

    assert message == {"role": "assistant", "content": "Hello"}
    client.post.assert_awaited_once_with(
        "https://api.example.com/v1/chat/completions",
        headers={"Authorization": "Bearer test-key"},
        json={
            "model": "paid-model",
            "messages": [{"role": "user", "content": "Hi"}],
            "tools": [],
        },
    )
