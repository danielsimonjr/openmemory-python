# OpenMemory (Python)

Python implementation of OpenMemory - Long-term memory for AI systems with cognitive architecture.

This is a reimplementation of [OpenMemory](https://github.com/CaviraOSS/OpenMemory) in Python, maintaining the same cognitive memory architecture while enabling integration with Python-based AI systems.

## Features

- **Multi-sector memory** - Episodic, semantic, procedural, emotional, and reflective memory types
- **Automatic decay** - Memories fade naturally unless reinforced
- **Graph associations** - Waypoint-based memory linking
- **Pattern recognition** - Regex-based sector classification
- **User isolation** - Per-user memory spaces
- **Pluggable embeddings** - OpenAI, Sentence-Transformers, or local models
- **MCP Server** - Built-in Model Context Protocol server

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

```python
from openmemory import MemorySystem

# Initialize memory system
memory = MemorySystem(db_path="memory.db")

# Add a memory
result = memory.add_memory(
    content="User prefers dark mode in their IDE",
    user_id="user123"
)

# Query memories
results = memory.query(
    query="What are the user's preferences?",
    user_id="user123",
    k=5
)

for mem in results:
    print(f"Score: {mem.score:.3f} | {mem.content}")
```

## Architecture

OpenMemory uses a cognitive architecture with five memory sectors:

- **Episodic** (decay: 0.015/day) - Time-based experiences
- **Semantic** (decay: 0.005/day) - Facts and knowledge
- **Procedural** (decay: 0.008/day) - How-to instructions
- **Emotional** (decay: 0.020/day) - Feelings and moods
- **Reflective** (decay: 0.001/day) - Insights and wisdom

Each memory is:
- Classified into sectors using pattern matching
- Embedded with sector-specific models
- Connected via waypoint graphs
- Tracked with salience scores (0-1)
- Subject to exponential decay

## MCP Server

Run the MCP server:

```bash
python -m openmemory.mcp.server
```

Or use with Claude Desktop (add to config):

```json
{
  "mcpServers": {
    "openmemory": {
      "command": "python",
      "args": ["-m", "openmemory.mcp.server"]
    }
  }
}
```

## Security: model loading

Hugging Face model repos can ship arbitrary `*.py` files that execute on
load when `trust_remote_code` is enabled, and the legacy
`pytorch_model.bin` format is a Python pickle and therefore an
arbitrary-code-execution channel by itself. OpenMemory loads embedding
models with hardened defaults:

- `trust_remote_code=False` on every load, so no custom Python from the
  model repo is ever executed.
- Every default model is pinned to an audited commit SHA (see
  `DEFAULT_REVISIONS` in `openmemory/embeddings.py`); an upstream tampered
  or reuploaded weight cannot silently swap in.
- `safetensors` weights are preferred via
  `model_kwargs={"use_safetensors": True}`, removing the pickle attack
  surface entirely. We fall back to legacy weights only when a repo has no
  safetensors, and even then keep `trust_remote_code=False` and the pin.

### Overriding the default model

For air-gapped mirrors, vendored forks, or pinning to a different audited
revision, set:

| Env var                          | Effect                                                                 |
| -------------------------------- | ---------------------------------------------------------------------- |
| `OPENMEMORY_EMBEDDING_MODEL`     | Replaces the default model id (applies to every sector).               |
| `OPENMEMORY_EMBEDDING_REVISION`  | Pins all loads to this commit SHA, overriding `DEFAULT_REVISIONS`.     |

Example:

```bash
export OPENMEMORY_EMBEDDING_MODEL="my-org/audited-minilm"
export OPENMEMORY_EMBEDDING_REVISION="c9745ed1d9f207416be6d2e6f8de32d1f16199bf"
python -m openmemory.mcp.server
```

For per-sector control, pass an explicit `models` dict to
`EmbeddingProvider` instead of using the env vars.

## Development

```bash
# Install dev dependencies
pip install -r requirements-dev.txt

# Run tests
pytest tests/

# Format code
black openmemory/
```

## License

MIT License - Copyright (c) 2025

## Credits

Based on [OpenMemory by CaviraOSS](https://github.com/CaviraOSS/OpenMemory)

Python reimplementation by @danielsimonjr
