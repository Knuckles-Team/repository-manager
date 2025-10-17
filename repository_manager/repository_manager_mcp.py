#!/usr/bin/env python
# coding: utf-8
import os
import sys
import argparse
import logging
from typing import Optional, Dict, List, Union
from pydantic import Field
from fastmcp import FastMCP
from fastmcp.server.auth.oidc_proxy import OIDCProxy
from fastmcp.server.auth import OAuthProxy, RemoteAuthProvider
from fastmcp.server.auth.providers.jwt import JWTVerifier, StaticTokenVerifier
from fastmcp.server.middleware.logging import LoggingMiddleware
from fastmcp.server.middleware.timing import TimingMiddleware
from fastmcp.server.middleware.rate_limiting import RateLimitingMiddleware
from fastmcp.server.middleware.error_handling import ErrorHandlingMiddleware
from repository_manager.repository_manager import setup_logging, Git

# Initialize logging for MCP server
logger = setup_logging(is_mcp_server=True, log_file="repository_manager_mcp.log")

mcp = FastMCP(name="GitRepositoryManager")


def to_boolean(string: Union[str, bool] = None) -> bool:
    if isinstance(string, bool):
        return string
    if not string:
        return False
    normalized = str(string).strip().lower()
    true_values = {"t", "true", "y", "yes", "1"}
    false_values = {"f", "false", "n", "no", "0"}
    if normalized in true_values:
        return True
    elif normalized in false_values:
        return False
    else:
        raise ValueError(f"Cannot convert '{string}' to boolean")


def to_integer(string: Union[str, int] = None) -> int:
    if isinstance(string, int):
        return string
    if not string:
        return 0
    try:
        return int(string.strip())
    except ValueError:
        raise ValueError(f"Cannot convert '{string}' to integer")


@mcp.tool(
    annotations={
        "title": "Execute Git Command",
        "readOnlyHint": False,
        "destructiveHint": False,
        "idempotentHint": False,
        "openWorldHint": False,
    },
    tags={"git_operations"},
)
async def git_action(
    command: str = Field(
        description="The Git command to execute (e.g., 'git pull', 'git clone <repository_url>')"
    ),
    repository_directory: Optional[str] = Field(
        description="The directory to execute the command in. Defaults to REPOSITORY_MANAGER_DIRECTORY env variable.",
        default=os.environ.get("REPOSITORY_MANAGER_DIRECTORY", None),
    ),
    projects: Optional[List[str]] = Field(
        description="List of repository URLs for Git operations.", default=None
    ),
    projects_file: Optional[str] = Field(
        description="Path to a file containing a list of repository URLs. Defaults to PROJECTS_FILE env variable.",
        default=os.environ.get("PROJECTS_FILE", None),
    ),
    threads: Optional[int] = Field(
        description="Number of threads for parallel processing. Defaults to REPOSITORY_MANAGER_THREADS env variable.",
        default=to_integer(os.environ.get("REPOSITORY_MANAGER_THREADS", "6")),
    ),
    set_to_default_branch: Optional[bool] = Field(
        description="Whether to checkout the default branch. Defaults to REPOSITORY_MANAGER_DEFAULT_BRANCH env variable.",
        default=to_boolean(os.environ.get("REPOSITORY_MANAGER_DEFAULT_BRANCH", None)),
    ),
) -> Dict:
    """
    Executes a Git command in the specified directory.
    Returns details from that git action run
    """
    logger.debug(
        f"Executing git_action with command: {command}, directory: {repository_directory}"
    )
    try:
        git = Git(
            repository_directory=repository_directory,
            projects=projects,
            threads=threads,
            set_to_default_branch=set_to_default_branch,
            capture_output=True,
            is_mcp_server=True,
        )
        if projects_file:
            git.read_project_list_file(file=projects_file)
        response = git.git_action(command=command)
        return response
    except Exception as e:
        logger.error(f"Error in git_action: {e}")
        raise


@mcp.tool(
    annotations={
        "title": "Clone Single Git Project",
        "readOnlyHint": False,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False,
    },
    tags={"git_operations"},
)
async def clone_project(
    git_project: Optional[str] = Field(
        description="The repository URL to clone.", default=None
    ),
    repository_directory: Optional[str] = Field(
        description="The directory to clone the project into. Defaults to REPOSITORY_MANAGER_DIRECTORY env variable.",
        default=os.environ.get("REPOSITORY_MANAGER_DIRECTORY", None),
    ),
    threads: Optional[int] = Field(
        description="Number of threads for parallel processing. Defaults to REPOSITORY_MANAGER_THREADS env variable.",
        default=to_integer(os.environ.get("REPOSITORY_MANAGER_THREADS", "6")),
    ),
    set_to_default_branch: Optional[bool] = Field(
        description="Whether to checkout the default branch. Defaults to REPOSITORY_MANAGER_DEFAULT_BRANCH env variable.",
        default=to_boolean(os.environ.get("REPOSITORY_MANAGER_DEFAULT_BRANCH", None)),
    ),
) -> str:
    """
    Clones a single Git project to the specified directory.
    Returns details about the cloned project
    """
    logger.debug(f"Cloning project: {git_project}, directory: {repository_directory}")
    try:
        if not git_project:
            raise ValueError("git_project must not be empty")
        git = Git(
            repository_directory=repository_directory,
            threads=threads,
            set_to_default_branch=set_to_default_branch,
            capture_output=True,
            is_mcp_server=True,
        )
        response = git.clone_project(git_project=git_project)
        return f"Project {git_project} cloned to {repository_directory} successfully!\nFull Response: \n{response}"
    except Exception as e:
        logger.error(f"Error in clone_project: {e}")
        raise


@mcp.tool(
    annotations={
        "title": "Clone Multiple Git Projects",
        "readOnlyHint": False,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False,
    },
    tags={"git_operations"},
)
async def clone_projects(
    projects: Optional[List[str]] = Field(
        description="List of repository URLs to clone.", default=None
    ),
    projects_file: Optional[str] = Field(
        description="Path to a file containing a list of repository URLs. Defaults to PROJECTS_FILE env variable.",
        default=os.environ.get("PROJECTS_FILE", None),
    ),
    repository_directory: Optional[str] = Field(
        description="The directory to clone projects into. Defaults to REPOSITORY_MANAGER_DIRECTORY env variable.",
        default=os.environ.get("REPOSITORY_MANAGER_DIRECTORY", None),
    ),
    threads: Optional[int] = Field(
        description="Number of threads for parallel processing. Defaults to REPOSITORY_MANAGER_THREADS env variable.",
        default=to_integer(os.environ.get("REPOSITORY_MANAGER_THREADS", "6")),
    ),
    set_to_default_branch: Optional[bool] = Field(
        description="Whether to checkout the default branch. Defaults to REPOSITORY_MANAGER_DEFAULT_BRANCH env variable.",
        default=to_boolean(os.environ.get("REPOSITORY_MANAGER_DEFAULT_BRANCH", None)),
    ),
) -> str:
    """
    Clones multiple Git projects in parallel to the specified directory.
    Returns a list of projects that were cloned
    """
    logger.debug(f"Cloning projects to directory: {repository_directory}")
    try:
        if not projects and not projects_file:
            raise ValueError("Either projects or projects_file must be provided")
        if projects_file and not os.path.exists(projects_file):
            raise FileNotFoundError(f"Projects file not found: {projects_file}")
        if repository_directory and not os.path.exists(repository_directory):
            raise FileNotFoundError(
                f"Repository directory not found: {repository_directory}"
            )
        git = Git(
            repository_directory=repository_directory,
            projects=projects,
            threads=threads,
            set_to_default_branch=set_to_default_branch,
            capture_output=True,
            is_mcp_server=True,
        )
        if projects_file:
            git.read_project_list_file(file=projects_file)
        response = git.clone_projects_in_parallel()
        return f"Project {git.projects} cloned to {repository_directory} successfully!\nFull Response: \n{response}"
    except Exception as e:
        logger.error(f"Error in clone_projects: {e}")
        raise


@mcp.tool(
    annotations={
        "title": "Pull Single Git Project",
        "readOnlyHint": False,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False,
    },
    tags={"git_operations"},
)
async def pull_project(
    git_project: str = Field(description="The name of the project directory to pull."),
    repository_directory: Optional[str] = Field(
        description="The parent directory containing the project. Defaults to REPOSITORY_MANAGER_DIRECTORY env variable.",
        default=os.environ.get("REPOSITORY_MANAGER_DIRECTORY", None),
    ),
    threads: Optional[int] = Field(
        description="Number of threads for parallel processing. Defaults to REPOSITORY_MANAGER_THREADS env variable.",
        default=to_integer(os.environ.get("REPOSITORY_MANAGER_THREADS", "6")),
    ),
    set_to_default_branch: Optional[bool] = Field(
        description="Whether to checkout the default branch. Defaults to REPOSITORY_MANAGER_DEFAULT_BRANCH env variable.",
        default=to_boolean(os.environ.get("REPOSITORY_MANAGER_DEFAULT_BRANCH", None)),
    ),
) -> str:
    """
    Pulls updates for a single Git project.
    Returns details about project pulled using git
    """
    logger.debug(f"Pulling project: {git_project}, directory: {repository_directory}")
    try:
        if not git_project:
            raise ValueError("git_project must not be empty")
        git = Git(
            repository_directory=repository_directory,
            threads=threads,
            set_to_default_branch=set_to_default_branch,
            capture_output=True,
            is_mcp_server=True,
        )
        response = git.pull_project(git_project=git_project)
        return f"Project {git_project} pulled to {repository_directory} successfully!\nFull Response: \n{response}"
    except Exception as e:
        logger.error(f"Error in pull_project: {e}")
        raise


@mcp.tool(
    annotations={
        "title": "Pull Multiple Git Projects",
        "readOnlyHint": False,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False,
    },
    tags={"git_operations"},
)
async def pull_projects(
    repository_directory: Optional[str] = Field(
        description="The directory containing the projects to pull. Defaults to REPOSITORY_MANAGER_DIRECTORY env variable.",
        default=os.environ.get("REPOSITORY_MANAGER_DIRECTORY", None),
    ),
    threads: Optional[int] = Field(
        description="Number of threads for parallel processing. Defaults to REPOSITORY_MANAGER_THREADS env variable.",
        default=to_integer(os.environ.get("REPOSITORY_MANAGER_THREADS", "6")),
    ),
    set_to_default_branch: Optional[bool] = Field(
        description="Whether to checkout the default branch. Defaults to REPOSITORY_MANAGER_DEFAULT_BRANCH env variable.",
        default=to_boolean(os.environ.get("REPOSITORY_MANAGER_DEFAULT_BRANCH", None)),
    ),
) -> str:
    """
    Pulls updates for multiple Git projects in parallel.
    Returns a list of projects that were pulled
    """
    logger.debug(f"Pulling projects from directory: {repository_directory}")
    try:
        if repository_directory and not os.path.exists(repository_directory):
            raise FileNotFoundError(
                f"Repository directory not found: {repository_directory}"
            )
        git = Git(
            repository_directory=repository_directory,
            threads=threads,
            set_to_default_branch=set_to_default_branch,
            capture_output=True,
            is_mcp_server=True,
        )
        response = git.pull_projects_in_parallel()
        return f"All projects in {repository_directory} pulled successfully!\nFull Response: \n{response}"
    except Exception as e:
        logger.error(f"Error in pull_projects: {e}")
        raise


def repository_manager_mcp():
    parser = argparse.ArgumentParser(description="Repository Manager MCP Utility")
    parser.add_argument(
        "-t",
        "--transport",
        default="stdio",
        choices=["stdio", "http", "sse"],
        help="Transport method: 'stdio', 'http', or 'sse' [legacy] (default: stdio)",
    )
    parser.add_argument(
        "-s",
        "--host",
        default="0.0.0.0",
        help="Host address for HTTP transport (default: 0.0.0.0)",
    )
    parser.add_argument(
        "-p",
        "--port",
        type=int,
        default=8000,
        help="Port number for HTTP transport (default: 8000)",
    )
    parser.add_argument(
        "--auth-type",
        default="none",
        choices=["none", "static", "jwt", "oauth-proxy", "oidc-proxy", "remote-oauth"],
        help="Authentication type for MCP server: 'none' (disabled), 'static' (internal), 'jwt' (external token verification), 'oauth-proxy', 'oidc-proxy', 'remote-oauth' (external) (default: none)",
    )
    # JWT/Token params
    parser.add_argument(
        "--token-jwks-uri", default=None, help="JWKS URI for JWT verification"
    )
    parser.add_argument(
        "--token-issuer", default=None, help="Issuer for JWT verification"
    )
    parser.add_argument(
        "--token-audience", default=None, help="Audience for JWT verification"
    )
    # OAuth Proxy params
    parser.add_argument(
        "--oauth-upstream-auth-endpoint",
        default=None,
        help="Upstream authorization endpoint for OAuth Proxy",
    )
    parser.add_argument(
        "--oauth-upstream-token-endpoint",
        default=None,
        help="Upstream token endpoint for OAuth Proxy",
    )
    parser.add_argument(
        "--oauth-upstream-client-id",
        default=None,
        help="Upstream client ID for OAuth Proxy",
    )
    parser.add_argument(
        "--oauth-upstream-client-secret",
        default=None,
        help="Upstream client secret for OAuth Proxy",
    )
    parser.add_argument(
        "--oauth-base-url", default=None, help="Base URL for OAuth Proxy"
    )
    # OIDC Proxy params
    parser.add_argument(
        "--oidc-config-url", default=None, help="OIDC configuration URL"
    )
    parser.add_argument("--oidc-client-id", default=None, help="OIDC client ID")
    parser.add_argument("--oidc-client-secret", default=None, help="OIDC client secret")
    parser.add_argument("--oidc-base-url", default=None, help="Base URL for OIDC Proxy")
    # Remote OAuth params
    parser.add_argument(
        "--remote-auth-servers",
        default=None,
        help="Comma-separated list of authorization servers for Remote OAuth",
    )
    parser.add_argument(
        "--remote-base-url", default=None, help="Base URL for Remote OAuth"
    )
    # Common
    parser.add_argument(
        "--allowed-client-redirect-uris",
        default=None,
        help="Comma-separated list of allowed client redirect URIs",
    )
    # Eunomia params
    parser.add_argument(
        "--eunomia-type",
        default="none",
        choices=["none", "embedded", "remote"],
        help="Eunomia authorization type: 'none' (disabled), 'embedded' (built-in), 'remote' (external) (default: none)",
    )
    parser.add_argument(
        "--eunomia-policy-file",
        default="mcp_policies.json",
        help="Policy file for embedded Eunomia (default: mcp_policies.json)",
    )
    parser.add_argument(
        "--eunomia-remote-url", default=None, help="URL for remote Eunomia server"
    )

    args = parser.parse_args()

    if args.port < 0 or args.port > 65535:
        print(f"Error: Port {args.port} is out of valid range (0-65535).")
        sys.exit(1)

    # Set auth based on type
    auth = None
    allowed_uris = (
        args.allowed_client_redirect_uris.split(",")
        if args.allowed_client_redirect_uris
        else None
    )

    if args.auth_type == "none":
        auth = None
    elif args.auth_type == "static":
        # Internal static tokens (hardcoded example)
        auth = StaticTokenVerifier(
            tokens={
                "test-token": {"client_id": "test-user", "scopes": ["read", "write"]},
                "admin-token": {"client_id": "admin", "scopes": ["admin"]},
            }
        )
    elif args.auth_type == "jwt":
        if not (args.token_jwks_uri and args.token_issuer and args.token_audience):
            print(
                "Error: jwt requires --token-jwks-uri, --token-issuer, --token-audience"
            )
            sys.exit(1)
        auth = JWTVerifier(
            jwks_uri=args.token_jwks_uri,
            issuer=args.token_issuer,
            audience=args.token_audience,
        )
    elif args.auth_type == "oauth-proxy":
        if not (
            args.oauth_upstream_auth_endpoint
            and args.oauth_upstream_token_endpoint
            and args.oauth_upstream_client_id
            and args.oauth_upstream_client_secret
            and args.oauth_base_url
            and args.token_jwks_uri
            and args.token_issuer
            and args.token_audience
        ):
            print(
                "Error: oauth-proxy requires --oauth-upstream-auth-endpoint, --oauth-upstream-token-endpoint, --oauth-upstream-client-id, --oauth-upstream-client-secret, --oauth-base-url, --token-jwks-uri, --token-issuer, --token-audience"
            )
            sys.exit(1)
        token_verifier = JWTVerifier(
            jwks_uri=args.token_jwks_uri,
            issuer=args.token_issuer,
            audience=args.token_audience,
        )
        auth = OAuthProxy(
            upstream_authorization_endpoint=args.oauth_upstream_auth_endpoint,
            upstream_token_endpoint=args.oauth_upstream_token_endpoint,
            upstream_client_id=args.oauth_upstream_client_id,
            upstream_client_secret=args.oauth_upstream_client_secret,
            token_verifier=token_verifier,
            base_url=args.oauth_base_url,
            allowed_client_redirect_uris=allowed_uris,
        )
    elif args.auth_type == "oidc-proxy":
        if not (
            args.oidc_config_url
            and args.oidc_client_id
            and args.oidc_client_secret
            and args.oidc_base_url
        ):
            print(
                "Error: oidc-proxy requires --oidc-config-url, --oidc-client-id, --oidc-client-secret, --oidc-base-url"
            )
            sys.exit(1)
        auth = OIDCProxy(
            config_url=args.oidc_config_url,
            client_id=args.oidc_client_id,
            client_secret=args.oidc_client_secret,
            base_url=args.oidc_base_url,
            allowed_client_redirect_uris=allowed_uris,
        )
    elif args.auth_type == "remote-oauth":
        if not (
            args.remote_auth_servers
            and args.remote_base_url
            and args.token_jwks_uri
            and args.token_issuer
            and args.token_audience
        ):
            print(
                "Error: remote-oauth requires --remote-auth-servers, --remote-base-url, --token-jwks-uri, --token-issuer, --token-audience"
            )
            sys.exit(1)
        auth_servers = [url.strip() for url in args.remote_auth_servers.split(",")]
        token_verifier = JWTVerifier(
            jwks_uri=args.token_jwks_uri,
            issuer=args.token_issuer,
            audience=args.token_audience,
        )
        auth = RemoteAuthProvider(
            token_verifier=token_verifier,
            authorization_servers=auth_servers,
            base_url=args.remote_base_url,
        )
    mcp.auth = auth
    if args.eunomia_type != "none":
        from eunomia_mcp import create_eunomia_middleware

        if args.eunomia_type == "embedded":
            if not args.eunomia_policy_file:
                print("Error: embedded Eunomia requires --eunomia-policy-file")
                sys.exit(1)
            middleware = create_eunomia_middleware(policy_file=args.eunomia_policy_file)
            mcp.add_middleware(middleware)
        elif args.eunomia_type == "remote":
            if not args.eunomia_remote_url:
                print("Error: remote Eunomia requires --eunomia-remote-url")
                sys.exit(1)
            middleware = create_eunomia_middleware(
                use_remote_eunomia=args.eunomia_remote_url
            )
            mcp.add_middleware(middleware)

    mcp.add_middleware(
        ErrorHandlingMiddleware(include_traceback=True, transform_errors=True)
    )
    mcp.add_middleware(
        RateLimitingMiddleware(max_requests_per_second=10.0, burst_capacity=20)
    )
    mcp.add_middleware(TimingMiddleware())
    mcp.add_middleware(LoggingMiddleware())

    if args.transport == "stdio":
        mcp.run(transport="stdio")
    elif args.transport == "http":
        mcp.run(transport="http", host=args.host, port=args.port)
    elif args.transport == "sse":
        mcp.run(transport="sse", host=args.host, port=args.port)
    else:
        logger = logging.getLogger("RepositoryManager")
        logger.error("Transport not supported")
        sys.exit(1)


if __name__ == "__main__":
    repository_manager_mcp()
