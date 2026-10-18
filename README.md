# Ollama MCP Client

A small REST client that connects an AI model to the City Services MCP server.
The client accepts a natural-language query, gives the MCP tool definitions to
the model, executes requested MCP tool calls, and returns the final answer.
It supports Ollama locally and any provider with an OpenAI-compatible
Chat Completions API and tool calling.

## Setup

From the workspace root:

```powershell
cd ollama_mcp_client
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Copy `.env.example` to `.env` and configure the MCP server and AI provider.
The defaults use Ollama with `llama3.2`:

```powershell
Copy-Item .env.example .env
ollama pull llama3.2
```

Start Ollama separately if it is not already running:

```powershell
ollama serve
```

## Run

```powershell
python app.py
```

The client reads `.env` automatically. For running directly on Windows rather
than in Docker, set `MCP_SERVER_URL` to `http://127.0.0.1:8090/mcp` and
`AI_BASE_URL` to `http://127.0.0.1:11434` in `.env`.

## Run in Docker

The MCP server and Ollama must be running separately and reachable from the
container. The default `.env` URLs use `host.docker.internal`, which Docker
Desktop maps to the host machine.

```powershell
docker build -t ollama-mcp-client .
docker run --rm --name ollama-mcp-client -p 8001:8001 --env-file .env ollama-mcp-client
```

The REST API is then available at `http://localhost:8001`; interactive API
documentation is at `http://localhost:8001/docs`.

### Use a paid OpenAI-compatible model

Choose a model that supports tool/function calling, then edit `.env`:

```dotenv
AI_PROVIDER=openai-compatible
AI_BASE_URL=https://api.openai.com/v1
AI_MODEL=your-tool-capable-model
AI_API_KEY=your_api_key_here
```

`AI_BASE_URL` and `AI_MODEL` vary by provider. This mode uses the standard
OpenAI-compatible `/chat/completions` endpoint and sends the configured API key
in the authorization header. It works with providers implementing that API,
but not providers that only offer a different protocol. Keep real API keys in
`.env` and do not commit it.

## API

Health check:

```powershell
Invoke-RestMethod http://localhost:8001/health
```

Ask a question:

```powershell
$body = @{ query = 'When is the next bus to downtown after 09:00?' } | ConvertTo-Json
Invoke-RestMethod -Method Post -Uri http://localhost:8001/query -ContentType 'application/json' -Body $body
```

Example response:

```json
{
  "answer": "The next bus to downtown is at 09:15.",
  "tool_calls": [
    {
      "name": "get_next_bus",
      "arguments": {"destination": "downtown", "current_time": "09:00"},
      "result": {"next_bus": "09:15"}
    }
  ]
}
```

## School FAQs

Create or update a question and answer for a school:

```powershell
$body = @{
  school_name = 'North High'
  question = 'When does the library close?'
  answer = 'The library closes at 6 PM on weekdays.'
} | ConvertTo-Json
Invoke-RestMethod -Method Post -Uri http://localhost:8001/school-faqs -ContentType 'application/json' -Body $body
```

Search FAQs for a specific school:

```powershell
$body = @{
  school_name = 'North High'
  question = 'What time does the library close?'
  limit = 3
} | ConvertTo-Json
Invoke-RestMethod -Method Post -Uri http://localhost:8001/school-faqs/query -ContentType 'application/json' -Body $body
```

Natural-language `/query` requests about a named school's FAQ use the same school-scoped search tool. The assistant adds FAQs only when explicitly asked to save information.

Interactive API documentation is available at `http://localhost:8001/docs`.
