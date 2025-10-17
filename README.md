# Repository Manager

![PyPI - Version](https://img.shields.io/pypi/v/repository-manager)
![PyPI - Downloads](https://img.shields.io/pypi/dd/repository-manager)
![GitHub Repo stars](https://img.shields.io/github/stars/Knuckles-Team/repository-manager)
![GitHub forks](https://img.shields.io/github/forks/Knuckles-Team/repository-manager)
![GitHub contributors](https://img.shields.io/github/contributors/Knuckles-Team/repository-manager)
![PyPI - License](https://img.shields.io/pypi/l/repository-manager)
![GitHub](https://img.shields.io/github/license/Knuckles-Team/repository-manager)

![GitHub last commit (by committer)](https://img.shields.io/github/last-commit/Knuckles-Team/repository-manager)
![GitHub pull requests](https://img.shields.io/github/issues-pr/Knuckles-Team/repository-manager)
![GitHub closed pull requests](https://img.shields.io/github/issues-pr-closed/Knuckles-Team/repository-manager)
![GitHub issues](https://img.shields.io/github/issues/Knuckles-Team/repository-manager)

![GitHub top language](https://img.shields.io/github/languages/top/Knuckles-Team/repository-manager)
![GitHub language count](https://img.shields.io/github/languages/count/Knuckles-Team/repository-manager)
![GitHub repo size](https://img.shields.io/github/repo-size/Knuckles-Team/repository-manager)
![GitHub repo file count (file type)](https://img.shields.io/github/directory-file-count/Knuckles-Team/repository-manager)
![PyPI - Wheel](https://img.shields.io/pypi/wheel/repository-manager)
![PyPI - Implementation](https://img.shields.io/pypi/implementation/repository-manager)

*Version: 1.1.11*

Manage your Git projects

Run all Git supported tasks using Git Actions command

Run as an MCP Server for Agentic AI!

#### Using an an MCP Server:

AI Prompt:
```text
Clone all the git projects located in the file "/home/genius/Development/repositories-list/repositories.txt" to my "/home/genius/Development" folder.
Afterwards, pull all the projects located in the "/home/genius/Development" repository folder.
```

AI Response:
```text
All projects in "/home/genius/Development/repositories-list/repositories.txt" have been cloned to "/home/genius/Development"
and all projects in "/home/genius/Development" and been pulled from the repositories. Let me know if you need any further actions! 🚀.
```

This repository is actively maintained - Contributions are welcome!

<details>
  <summary><b>Usage:</b></summary>

### CLI

| Short Flag | Long Flag        | Description                            |
|------------|------------------|----------------------------------------|
| -h         | --help           | See Usage                              |
| -b         | --default-branch | Checkout default branch                |
| -c         | --clone          | Clone projects specified               |
| -d         | --directory      | Directory to clone/pull projects       |
| -f         | --file           | File with repository links             |
| -p         | --pull           | Pull projects in parent directory      |
| -r         | --repositories   | Comma separated Git URLs               |
| -t         | --threads        | Number of parallel threads - Default 4 |

```bash
repository-manager \
    --clone  \
    --pull  \
    --directory '/home/user/Downloads'  \
    --file '/home/user/Downloads/repositories.txt'  \
    --repositories 'https://github.com/Knucklessg1/media-downloader,https://github.com/Knucklessg1/genius-bot' \
    --threads 8
```

### MCP CLI

| Short Flag | Long Flag                          | Description                                                                 |
|------------|------------------------------------|-----------------------------------------------------------------------------|
| -h         | --help                             | Display help information                                                    |
| -t         | --transport                        | Transport method: 'stdio', 'http', or 'sse' [legacy] (default: stdio)       |
| -s         | --host                             | Host address for HTTP transport (default: 0.0.0.0)                          |
| -p         | --port                             | Port number for HTTP transport (default: 8000)                              |
|            | --auth-type                        | Authentication type: 'none', 'static', 'jwt', 'oauth-proxy', 'oidc-proxy', 'remote-oauth' (default: none) |
|            | --token-jwks-uri                   | JWKS URI for JWT verification                                              |
|            | --token-issuer                     | Issuer for JWT verification                                                |
|            | --token-audience                   | Audience for JWT verification                                              |
|            | --oauth-upstream-auth-endpoint     | Upstream authorization endpoint for OAuth Proxy                             |
|            | --oauth-upstream-token-endpoint    | Upstream token endpoint for OAuth Proxy                                    |
|            | --oauth-upstream-client-id         | Upstream client ID for OAuth Proxy                                         |
|            | --oauth-upstream-client-secret     | Upstream client secret for OAuth Proxy                                     |
|            | --oauth-base-url                   | Base URL for OAuth Proxy                                                   |
|            | --oidc-config-url                  | OIDC configuration URL                                                     |
|            | --oidc-client-id                   | OIDC client ID                                                             |
|            | --oidc-client-secret               | OIDC client secret                                                         |
|            | --oidc-base-url                    | Base URL for OIDC Proxy                                                    |
|            | --remote-auth-servers              | Comma-separated list of authorization servers for Remote OAuth             |
|            | --remote-base-url                  | Base URL for Remote OAuth                                                  |
|            | --allowed-client-redirect-uris     | Comma-separated list of allowed client redirect URIs                       |
|            | --eunomia-type                     | Eunomia authorization type: 'none', 'embedded', 'remote' (default: none)   |
|            | --eunomia-policy-file              | Policy file for embedded Eunomia (default: mcp_policies.json)              |
|            | --eunomia-remote-url               | URL for remote Eunomia server                                              |

### Using as an MCP Server

The MCP Server can be run in two modes: `stdio` (for local testing) or `http` (for networked access). To start the server, use the following commands:

#### Run in stdio mode (default):
```bash
repository-manager-mcp --transport "stdio"
```

#### Run in HTTP mode:
```bash
repository-manager-mcp --transport "http"  --host "0.0.0.0"  --port "8000"
```

### Use in Python

```python
from repository_manager.repository_manager import Git

gitlab = Git()

gitlab.set_repository_directory("<directory>")

gitlab.set_threads(threads=8)

gitlab.set_git_projects("<projects>")

gitlab.set_default_branch(set_to_default_branch=True)

gitlab.clone_projects_in_parallel()

gitlab.pull_projects_in_parallel()
```


### Deploy MCP Server as a Service

The ServiceNow MCP server can be deployed using Docker, with configurable authentication, middleware, and Eunomia authorization.

#### Using Docker Run

```bash
docker pull knucklessg1/repository-manager:latest

docker run -d \
  --name repository-manager-mcp \
  -p 8004:8004 \
  -e HOST=0.0.0.0 \
  -e PORT=8004 \
  -e TRANSPORT=http \
  -e AUTH_TYPE=none \
  -e EUNOMIA_TYPE=none \
  -v development:/root/Development \
  knucklessg1/repository-manager:latest
```

For advanced authentication (e.g., JWT, OAuth Proxy, OIDC Proxy, Remote OAuth) or Eunomia, add the relevant environment variables:

```bash
docker run -d \
  --name repository-manager-mcp \
  -p 8004:8004 \
  -e HOST=0.0.0.0 \
  -e PORT=8004 \
  -e TRANSPORT=http \
  -e AUTH_TYPE=oidc-proxy \
  -e OIDC_CONFIG_URL=https://provider.com/.well-known/openid-configuration \
  -e OIDC_CLIENT_ID=your-client-id \
  -e OIDC_CLIENT_SECRET=your-client-secret \
  -e OIDC_BASE_URL=https://your-server.com \
  -e ALLOWED_CLIENT_REDIRECT_URIS=http://localhost:*,https://*.example.com/* \
  -e EUNOMIA_TYPE=embedded \
  -e EUNOMIA_POLICY_FILE=/app/mcp_policies.json \
  -v development:/root/Development \
  knucklessg1/repository-manager:latest
```

#### Using Docker Compose

Create a `docker-compose.yml` file:

```yaml
services:
  repository-manager-mcp:
    image: knucklessg1/repository-manager:latest
    environment:
      - HOST=0.0.0.0
      - PORT=8004
      - TRANSPORT=http
      - AUTH_TYPE=none
      - EUNOMIA_TYPE=none
    volumes:
      - development:/root/Development
    ports:
      - 8004:8004
```

For advanced setups with authentication and Eunomia:

```yaml
services:
  repository-manager-mcp:
    image: knucklessg1/repository-manager:latest
    environment:
      - HOST=0.0.0.0
      - PORT=8004
      - TRANSPORT=http
      - AUTH_TYPE=oidc-proxy
      - OIDC_CONFIG_URL=https://provider.com/.well-known/openid-configuration
      - OIDC_CLIENT_ID=your-client-id
      - OIDC_CLIENT_SECRET=your-client-secret
      - OIDC_BASE_URL=https://your-server.com
      - ALLOWED_CLIENT_REDIRECT_URIS=http://localhost:*,https://*.example.com/*
      - EUNOMIA_TYPE=embedded
      - EUNOMIA_POLICY_FILE=/app/mcp_policies.json
    ports:
      - 8004:8004
    volumes:
      - development:/root/Development
      - ./mcp_policies.json:/app/mcp_policies.json
```

Run the service:

```bash
docker-compose up -d
```

#### Configure `mcp.json` for AI Integration

```json
{
  "mcpServers": {
    "repository_manager": {
      "command": "uv",
      "args": [
        "run",
        "--with",
        "repository-manager",
        "repository-manager-mcp"
      ],
      "env": {
        "REPOSITORY_MANAGER_DIRECTORY": "/home/user/Development/",                       // Optional - Can be specified at prompt
        "REPOSITORY_MANAGER_THREADS": "12",                                              // Optional - Can be specified at prompt
        "REPOSITORY_MANAGER_DEFAULT_BRANCH": "True",                                     // Optional - Can be specified at prompt
        "REPOSITORY_MANAGER_PROJECTS_FILE": "/home/user/Development/repositories.txt"    // Optional - Can be specified at prompt
      },
      "timeout": 300000
    }
  }
}

```

</details>

<details>
  <summary><b>Installation Instructions:</b></summary>

Install Python Package
```bash
python -m pip install --upgrade repository-manager
```

or

```bash
uv pip install --upgrade repository-manager
```

</details>

<details>
  <summary><b>Repository Owners:</b></summary>

<img width="100%" height="180em" src="https://github-readme-stats.vercel.app/api?username=Knucklessg1&show_icons=true&hide_border=true&&count_private=true&include_all_commits=true" />

![GitHub followers](https://img.shields.io/github/followers/Knucklessg1)
![GitHub User's stars](https://img.shields.io/github/stars/Knucklessg1)
</details>
