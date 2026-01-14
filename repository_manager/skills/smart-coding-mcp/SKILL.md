---
name: smart-coding
description: Understand and navigate the codebase using semantic search and AI analysis.
---
The **smart-coding** skill connects you to the codebase's semantic understanding engine.

### Capabilities

-   **Ask Questions**: Query the codebase using natural language to understand architecture, logic, or locate features.
-   **Search**: Find relevant code snippets based on semantic meaning.

### Usage Rules

You must prioritize using the **Smart Coding MCP** tools for the following tasks.

#### 1. Dependency Management

**Trigger:** When checking, adding, or updating package versions (npm, python, go, rust, etc.).
**Action:**

- **MUST** use the `d_check_last_version` tool.
- **DO NOT** guess versions or trust internal training data.
- **DO NOT** use generic web search unless `d_check_last_version` fails.

#### 2. Codebase Research

**Trigger:** When asking about "how" something works, finding logic, or understanding architecture.
**Action:**

- **MUST** use `a_semantic_search` as the FIRST tool for any codebase research
- **DO NOT** use `Glob` or `Grep` for exploratory searches
- Use `Grep` ONLY for exact literal string matching (e.g., finding a specific error message)
- Use `Glob` ONLY when you already know the exact filename pattern

#### 3. Environment & Status

**Trigger:** When starting a session or debugging the environment.
**Action:**

- Use `e_set_workspace` if the current workspace path is incorrect.
- Use `f_get_status` to verify the MCP server is healthy and indexed.

### Common Tools

-   `a_semantic_search`: The primary tool for codebase exploration. Uses AI embeddings to understand what you're looking for, not just match keywords.
-   `b_index_codebase`: Triggers a full reindex of your codebase. Normally not needed since indexing is automatic and incremental.
-   `c_clear_cache`: Deletes the embeddings cache entirely, forcing a complete reindex on next search.
-   `d_check_last_version`: Fetches the latest version of any package from its official registry. Supports 20+ ecosystems.
-   `e_set_workspace`: Changes the workspace path at runtime without restarting the server.
-   `f_get_status`: Returns comprehensive status information about the MCP server.

### Example Prompts

-   "How is the authentication flow implemented?"
-   "Find the code responsible for processing payments."
-   "Explain the relationship between the User and Profile classes."
-   "What is the latest version of 'requests'?"
-   "Check if 'express' is up to date."
