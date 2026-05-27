# pr5750-evidence

Static assets referenced from the GIF embeds in `unslothai/unsloth` PR #5750
comments. Three files:

- `evidence/01_add_public_mcp_server.gif` — 7-frame walkthrough of adding a
  real public MCP server (`https://gitmcp.io/unslothai/unsloth_zoo`) through
  the new Manage MCP Servers dialog, including the live "Connected (4 tools)"
  toast against the upstream server.
- `evidence/02_model_dispatches_mcp_tool.gif` — 5-frame walkthrough of a
  loaded GGUF model (Qwen3-4B-Instruct-2507) emitting a `<tool_call>` for
  `mcp__<server_id>__fetch_unsloth_documentation` and Studio rendering the
  "Used tool" pill plus the model's follow-up answer composed from the real
  upstream response.
- `evidence/03_six_panel_howto.png` — single-image six-panel summary of the
  full manual add-MCP-server flow.

This branch is intentionally an orphan (no source history) so the assets
travel without dragging the staging fork's working tree behind them.
