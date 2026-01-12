---
name: Smart Coding
description: Understand and navigate the codebase using semantic search and AI analysis.
---
The **Smart Coding** skill connects you to the codebase's semantic understanding engine.

### Capabilities

-   **Ask Questions**: Query the codebase using natural language to understand architecture, logic, or locate features.
-   **Search**: Find relevant code snippets based on semantic meaning.

### Usage

When you need to understand how a component works or where code is located, use the available smart coding tools.

Each repository configured with `smart-coding-mcp` will expose its own set of tools, typically prefixed or namespaced if multiple are present. If you are working with multiple repositories, ensure you are using the tools associated with the correct repository context.

**Common Tools:**
-   `ask_codebase`: Ask a question about the codebase.
-   `search_codebase`: Search for code semantically.

**Example Prompts:**
-   "How is the authentication flow implemented?"
-   "Find the code responsible for processing payments."
-   "Explain the relationship between the User and Profile classes."
