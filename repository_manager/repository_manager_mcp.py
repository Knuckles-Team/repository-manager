#!/usr/bin/env python
# coding: utf-8
import os
import sys
import argparse
from typing import Optional, Dict, List, Union, Any
from pydantic import Field
from starlette.requests import Request
from starlette.responses import JSONResponse
from fastmcp import FastMCP, Context
from fastmcp.server.auth.oidc_proxy import OIDCProxy
from fastmcp.server.auth import OAuthProxy, RemoteAuthProvider
from fastmcp.server.auth.providers.jwt import JWTVerifier, StaticTokenVerifier
from fastmcp.server.middleware.logging import LoggingMiddleware
from fastmcp.server.middleware.timing import TimingMiddleware
from fastmcp.server.middleware.rate_limiting import RateLimitingMiddleware
from fastmcp.server.middleware.error_handling import ErrorHandlingMiddleware
import requests
import subprocess
import logging

from eunomia_mcp.middleware import EunomiaMcpMiddleware
from fastmcp.utilities.logging import get_logger
from repository_manager.utils import get_projects_file_path, to_integer, to_boolean
from repository_manager.repository_manager import Git
from repository_manager.middlewares import (
    UserTokenMiddleware,
    JWTClaimsLoggingMiddleware,
)

logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = get_logger("SearXNGMCPServer")

config = {
    "enable_delegation": to_boolean(os.environ.get("ENABLE_DELEGATION", "False")),
    "audience": os.environ.get("AUDIENCE", None),
    "delegated_scopes": os.environ.get("DELEGATED_SCOPES", "api"),
    "token_endpoint": None,  # Will be fetched dynamically from OIDC config
    "oidc_client_id": os.environ.get("OIDC_CLIENT_ID", None),
    "oidc_client_secret": os.environ.get("OIDC_CLIENT_SECRET", None),
    "oidc_config_url": os.environ.get("OIDC_CONFIG_URL", None),
    "jwt_jwks_uri": os.getenv("FASTMCP_SERVER_AUTH_JWT_JWKS_URI", None),
    "jwt_issuer": os.getenv("FASTMCP_SERVER_AUTH_JWT_ISSUER", None),
    "jwt_audience": os.getenv("FASTMCP_SERVER_AUTH_JWT_AUDIENCE", None),
    "jwt_algorithm": os.getenv("FASTMCP_SERVER_AUTH_JWT_ALGORITHM", None),
    "jwt_secret": os.getenv("FASTMCP_SERVER_AUTH_JWT_PUBLIC_KEY", None),
    "jwt_required_scopes": os.getenv("FASTMCP_SERVER_AUTH_JWT_REQUIRED_SCOPES", None),
}

DEFAULT_TRANSPORT = os.environ.get("TRANSPORT", "stdio")
DEFAULT_HOST = os.environ.get("HOST", "0.0.0.0")
DEFAULT_PORT = to_integer(os.environ.get("PORT", "8000"))


def register_tools(mcp: FastMCP):
    @mcp.custom_route("/health", methods=["GET"])
    async def health_check(request: Request) -> JSONResponse:
        return JSONResponse({"status": "OK"})

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
        project: Optional[str] = Field(
            description="The project to execute the command in.", default=None
        ),
        projects: Optional[List[str]] = Field(
            description="List of repository URLs for Git operations.", default=None
        ),
        projects_file: Optional[str] = Field(
            description="Path to a file containing a list of repository URLs. Defaults to PROJECTS_FILE env variable.",
            default=os.environ.get("PROJECTS_FILE", get_projects_file_path()),
        ),
        threads: Optional[int] = Field(
            description="Number of threads for parallel processing. Defaults to REPOSITORY_MANAGER_THREADS env variable.",
            default=to_integer(os.environ.get("REPOSITORY_MANAGER_THREADS", "6")),
        ),
        set_to_default_branch: Optional[bool] = Field(
            description="Whether to checkout the default branch. Defaults to REPOSITORY_MANAGER_DEFAULT_BRANCH env variable.",
            default=to_boolean(
                os.environ.get("REPOSITORY_MANAGER_DEFAULT_BRANCH", False)
            ),
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
            response = git.git_action(
                command=command, directory=repository_directory, project=project
            )
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
            default=to_boolean(
                os.environ.get("REPOSITORY_MANAGER_DEFAULT_BRANCH", None)
            ),
        ),
    ) -> str:
        """
        Clones a single Git project to the specified directory.
        Returns details about the cloned project
        """
        logger.debug(
            f"Cloning project: {git_project}, directory: {repository_directory}"
        )
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
            default=os.environ.get("PROJECTS_FILE", get_projects_file_path()),
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
            default=to_boolean(
                os.environ.get("REPOSITORY_MANAGER_DEFAULT_BRANCH", None)
            ),
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
        git_project: str = Field(
            description="The name of the project directory to pull."
        ),
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
            default=to_boolean(
                os.environ.get("REPOSITORY_MANAGER_DEFAULT_BRANCH", None)
            ),
        ),
    ) -> str:
        """
        Pulls updates for a single Git project.
        Returns details about project pulled using git
        """
        logger.debug(
            f"Pulling project: {git_project}, directory: {repository_directory}"
        )
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
            default=to_boolean(
                os.environ.get("REPOSITORY_MANAGER_DEFAULT_BRANCH", None)
            ),
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

    @mcp.tool(
        annotations={
            "title": "Run Command",
            "readOnlyHint": False,
            "destructiveHint": True,
            "idempotentHint": False,
            "openWorldHint": True,
        },
        tags={"system_operations"},
    )
    async def run_command(
        command: str = Field(description="The command to run"),
        ctx: Context = Field(
            description="MCP context for progress reporting.", default=None
        ),
    ) -> Dict[str, Any]:
        """
        Run a bash command on the local system.
        """
        logger.debug(f"Running command: {command}")
        if ctx:
            await ctx.report_progress(progress=0, total=100)
        try:
            result = subprocess.run(
                command, shell=True, capture_output=True, text=True, check=False
            )
            output = result.stdout
            if result.stderr:
                output += f"\nSTDERR:\n{result.stderr}"
            if ctx:
                await ctx.report_progress(progress=100, total=100)
            return {
                "status": 200 if result.returncode == 0 else 500,
                "output": output,
                "return_code": result.returncode,
            }
        except Exception as e:
            return {"status": 500, "error": str(e)}

    @mcp.tool(
        annotations={
            "title": "Text Editor",
            "readOnlyHint": False,
            "destructiveHint": True,
            "idempotentHint": False,
            "openWorldHint": True,
        },
        tags={"text_editor", "files"},
    )
    async def text_editor(
        command: str = Field(
            description="The command to perform: view, create, str_replace, insert, undo_edit"
        ),
        path: str = Field(description="Path to the file"),
        file_text: Optional[str] = Field(
            description="Content to write or insert", default=None
        ),
        view_range: Optional[List[int]] = Field(
            description="Line range to view [start, end]", default=None
        ),
        old_str: Optional[str] = Field(description="String to replace", default=None),
        new_str: Optional[str] = Field(description="Replacement string", default=None),
        insert_line: Optional[int] = Field(
            description="Line number to insert at", default=None
        ),
        ctx: Context = Field(
            description="MCP context for progress reporting.", default=None
        ),
    ) -> Dict[str, Any]:
        """
        View and edit files on the local filesystem.
        """
        logger.debug(f"Text editor command: {command} on {path}")
        expanded_path = os.path.abspath(os.path.expanduser(path))

        try:
            if command == "view":
                if not os.path.exists(expanded_path):
                    return {"status": 404, "error": "File not found"}
                with open(expanded_path, "r") as f:
                    lines = f.readlines()
                content = "".join(lines)
                if view_range and len(view_range) == 2:
                    start, end = view_range
                    # 1-based indexing for view_range typically? Let's assume 1-based to match editors
                    start = max(1, start)
                    end = min(len(lines), end)
                    content = "".join(lines[start - 1 : end])
                return {"status": 200, "content": content, "path": expanded_path}

            elif command == "create":
                if os.path.exists(expanded_path):
                    return {"status": 400, "error": "File already exists"}
                os.makedirs(os.path.dirname(expanded_path), exist_ok=True)
                with open(expanded_path, "w") as f:
                    f.write(file_text or "")
                return {"status": 200, "message": "File created", "path": expanded_path}

            elif command == "str_replace":
                if not os.path.exists(expanded_path):
                    return {"status": 404, "error": "File not found"}
                with open(expanded_path, "r") as f:
                    content = f.read()
                if old_str not in content:
                    return {"status": 400, "error": "Target string not found"}
                new_content = content.replace(
                    old_str, new_str or "", 1
                )  # Replace first occurrence only? Anthropic usually implies uniqueness or single block
                with open(expanded_path, "w") as f:
                    f.write(new_content)
                return {"status": 200, "message": "File updated", "path": expanded_path}

            elif command == "insert":
                if not os.path.exists(expanded_path):
                    return {"status": 404, "error": "File not found"}
                with open(expanded_path, "r") as f:
                    lines = f.readlines()
                if insert_line is None:
                    return {"status": 400, "error": "insert_line required"}
                # Insert AFTER the line? Or AT? Anthropic usually 0-indexed or 1-indexed? Assume 1-based
                idx = max(0, insert_line)
                # If idx is 0, insert at start?
                # Let's append
                new_lines = file_text.splitlines(keepends=True)
                # handle missing newlines
                if new_lines and not new_lines[-1].endswith("\n"):
                    new_lines[-1] += "\n"

                lines[idx:idx] = new_lines
                with open(expanded_path, "w") as f:
                    f.writelines(lines)
                return {
                    "status": 200,
                    "message": "Content inserted",
                    "path": expanded_path,
                }

            return {"status": 400, "error": f"Unknown command {command}"}

        except Exception as e:
            return {"status": 500, "error": str(e)}


def repository_manager_mcp():
    parser = argparse.ArgumentParser(description="Repository Manager MCP Utility")
    parser.add_argument(
        "-t",
        "--transport",
        default=DEFAULT_TRANSPORT,
        choices=["stdio", "streamable-http", "sse"],
        help="Transport method: 'stdio', 'streamable-http', or 'sse' [legacy] (default: stdio)",
    )
    parser.add_argument(
        "-s",
        "--host",
        default=DEFAULT_HOST,
        help="Host address for HTTP transport (default: 0.0.0.0)",
    )
    parser.add_argument(
        "-p",
        "--port",
        type=int,
        default=DEFAULT_PORT,
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
    parser.add_argument(
        "--token-algorithm",
        default=os.getenv("FASTMCP_SERVER_AUTH_JWT_ALGORITHM"),
        choices=[
            "HS256",
            "HS384",
            "HS512",
            "RS256",
            "RS384",
            "RS512",
            "ES256",
            "ES384",
            "ES512",
        ],
        help="JWT signing algorithm (required for HMAC or static key). Auto-detected for JWKS.",
    )
    parser.add_argument(
        "--token-secret",
        default=os.getenv("FASTMCP_SERVER_AUTH_JWT_PUBLIC_KEY"),
        help="Shared secret for HMAC (HS*) or PEM public key for static asymmetric verification.",
    )
    parser.add_argument(
        "--token-public-key",
        default=os.getenv("FASTMCP_SERVER_AUTH_JWT_PUBLIC_KEY"),
        help="Path to PEM public key file or inline PEM string (for static asymmetric keys).",
    )
    parser.add_argument(
        "--required-scopes",
        default=os.getenv("FASTMCP_SERVER_AUTH_JWT_REQUIRED_SCOPES"),
        help="Comma-separated list of required scopes (e.g., gitlab.read,gitlab.write).",
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
    # Delegation params
    parser.add_argument(
        "--enable-delegation",
        action="store_true",
        default=to_boolean(os.environ.get("ENABLE_DELEGATION", "False")),
        help="Enable OIDC token delegation",
    )
    parser.add_argument(
        "--audience",
        default=os.environ.get("AUDIENCE", None),
        help="Audience for the delegated token",
    )
    parser.add_argument(
        "--delegated-scopes",
        default=os.environ.get("DELEGATED_SCOPES", "api"),
        help="Scopes for the delegated token (space-separated)",
    )
    parser.add_argument(
        "--openapi-file",
        default=None,
        help="Path to the OpenAPI JSON file to import additional tools from",
    )
    parser.add_argument(
        "--openapi-base-url",
        default=None,
        help="Base URL for the OpenAPI client (overrides instance URL)",
    )
    parser.add_argument(
        "--openapi-use-token",
        action="store_true",
        help="Use the incoming Bearer token (from MCP request) to authenticate OpenAPI import",
    )

    parser.add_argument(
        "--openapi-username",
        default=os.getenv("OPENAPI_USERNAME"),
        help="Username for basic auth during OpenAPI import",
    )

    parser.add_argument(
        "--openapi-password",
        default=os.getenv("OPENAPI_PASSWORD"),
        help="Password for basic auth during OpenAPI import",
    )

    parser.add_argument(
        "--openapi-client-id",
        default=os.getenv("OPENAPI_CLIENT_ID"),
        help="OAuth client ID for OpenAPI import",
    )

    parser.add_argument(
        "--openapi-client-secret",
        default=os.getenv("OPENAPI_CLIENT_SECRET"),
        help="OAuth client secret for OpenAPI import",
    )

    args = parser.parse_args()

    if args.port < 0 or args.port > 65535:
        print(f"Error: Port {args.port} is out of valid range (0-65535).")
        sys.exit(1)

    # Update config with CLI arguments
    config["enable_delegation"] = args.enable_delegation
    config["audience"] = args.audience or config["audience"]
    config["delegated_scopes"] = args.delegated_scopes or config["delegated_scopes"]
    config["oidc_config_url"] = args.oidc_config_url or config["oidc_config_url"]
    config["oidc_client_id"] = args.oidc_client_id or config["oidc_client_id"]
    config["oidc_client_secret"] = (
        args.oidc_client_secret or config["oidc_client_secret"]
    )

    # Configure delegation if enabled
    if config["enable_delegation"]:
        if args.auth_type != "oidc-proxy":
            logger.error("Token delegation requires auth-type=oidc-proxy")
            sys.exit(1)
        if not config["audience"]:
            logger.error("audience is required for delegation")
            sys.exit(1)
        if not all(
            [
                config["oidc_config_url"],
                config["oidc_client_id"],
                config["oidc_client_secret"],
            ]
        ):
            logger.error(
                "Delegation requires complete OIDC configuration (oidc-config-url, oidc-client-id, oidc-client-secret)"
            )
            sys.exit(1)

        # Fetch OIDC configuration to get token_endpoint
        try:
            logger.info(
                "Fetching OIDC configuration",
                extra={"oidc_config_url": config["oidc_config_url"]},
            )
            oidc_config_resp = requests.get(config["oidc_config_url"])
            oidc_config_resp.raise_for_status()
            oidc_config = oidc_config_resp.json()
            config["token_endpoint"] = oidc_config.get("token_endpoint")
            if not config["token_endpoint"]:
                logger.error("No token_endpoint found in OIDC configuration")
                raise ValueError("No token_endpoint found in OIDC configuration")
            logger.info(
                "OIDC configuration fetched successfully",
                extra={"token_endpoint": config["token_endpoint"]},
            )
        except Exception as e:
            print(f"Failed to fetch OIDC configuration: {e}")
            logger.error(
                "Failed to fetch OIDC configuration",
                extra={"error_type": type(e).__name__, "error_message": str(e)},
            )
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
        auth = StaticTokenVerifier(
            tokens={
                "test-token": {"client_id": "test-user", "scopes": ["read", "write"]},
                "admin-token": {"client_id": "admin", "scopes": ["admin"]},
            }
        )
    elif args.auth_type == "jwt":
        # Fallback to env vars if not provided via CLI
        jwks_uri = args.token_jwks_uri or os.getenv("FASTMCP_SERVER_AUTH_JWT_JWKS_URI")
        issuer = args.token_issuer or os.getenv("FASTMCP_SERVER_AUTH_JWT_ISSUER")
        audience = args.token_audience or os.getenv("FASTMCP_SERVER_AUTH_JWT_AUDIENCE")
        algorithm = args.token_algorithm
        secret_or_key = args.token_secret or args.token_public_key
        public_key_pem = None

        if not (jwks_uri or secret_or_key):
            logger.error(
                "JWT auth requires either --token-jwks-uri or --token-secret/--token-public-key"
            )
            sys.exit(1)
        if not (issuer and audience):
            logger.error("JWT requires --token-issuer and --token-audience")
            sys.exit(1)

        # Load static public key from file if path is given
        if args.token_public_key and os.path.isfile(args.token_public_key):
            try:
                with open(args.token_public_key, "r") as f:
                    public_key_pem = f.read()
                logger.info(f"Loaded static public key from {args.token_public_key}")
            except Exception as e:
                print(f"Failed to read public key file: {e}")
                logger.error(f"Failed to read public key file: {e}")
                sys.exit(1)
        elif args.token_public_key:
            public_key_pem = args.token_public_key  # Inline PEM

        # Validation: Conflicting options
        if jwks_uri and (algorithm or secret_or_key):
            logger.warning(
                "JWKS mode ignores --token-algorithm and --token-secret/--token-public-key"
            )

        # HMAC mode
        if algorithm and algorithm.startswith("HS"):
            if not secret_or_key:
                logger.error(f"HMAC algorithm {algorithm} requires --token-secret")
                sys.exit(1)
            if jwks_uri:
                logger.error("Cannot use --token-jwks-uri with HMAC")
                sys.exit(1)
            public_key = secret_or_key
        else:
            public_key = public_key_pem

        # Required scopes
        required_scopes = None
        if args.required_scopes:
            required_scopes = [
                s.strip() for s in args.required_scopes.split(",") if s.strip()
            ]

        try:
            auth = JWTVerifier(
                jwks_uri=jwks_uri,
                public_key=public_key,
                issuer=issuer,
                audience=audience,
                algorithm=(
                    algorithm if algorithm and algorithm.startswith("HS") else None
                ),
                required_scopes=required_scopes,
            )
            logger.info(
                "JWTVerifier configured",
                extra={
                    "mode": (
                        "JWKS"
                        if jwks_uri
                        else (
                            "HMAC"
                            if algorithm and algorithm.startswith("HS")
                            else "Static Key"
                        )
                    ),
                    "algorithm": algorithm,
                    "required_scopes": required_scopes,
                },
            )
        except Exception as e:
            print(f"Failed to initialize JWTVerifier: {e}")
            logger.error(f"Failed to initialize JWTVerifier: {e}")
            sys.exit(1)
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
                "oauth-proxy requires oauth-upstream-auth-endpoint, oauth-upstream-token-endpoint, "
                "oauth-upstream-client-id, oauth-upstream-client-secret, oauth-base-url, token-jwks-uri, "
                "token-issuer, token-audience"
            )
            logger.error(
                "oauth-proxy requires oauth-upstream-auth-endpoint, oauth-upstream-token-endpoint, "
                "oauth-upstream-client-id, oauth-upstream-client-secret, oauth-base-url, token-jwks-uri, "
                "token-issuer, token-audience",
                extra={
                    "auth_endpoint": args.oauth_upstream_auth_endpoint,
                    "token_endpoint": args.oauth_upstream_token_endpoint,
                    "client_id": args.oauth_upstream_client_id,
                    "base_url": args.oauth_base_url,
                    "jwks_uri": args.token_jwks_uri,
                    "issuer": args.token_issuer,
                    "audience": args.token_audience,
                },
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
            logger.error(
                "oidc-proxy requires oidc-config-url, oidc-client-id, oidc-client-secret, oidc-base-url",
                extra={
                    "config_url": args.oidc_config_url,
                    "client_id": args.oidc_client_id,
                    "base_url": args.oidc_base_url,
                },
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
            logger.error(
                "remote-oauth requires remote-auth-servers, remote-base-url, token-jwks-uri, token-issuer, token-audience",
                extra={
                    "auth_servers": args.remote_auth_servers,
                    "base_url": args.remote_base_url,
                    "jwks_uri": args.token_jwks_uri,
                    "issuer": args.token_issuer,
                    "audience": args.token_audience,
                },
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

    # === 2. Build Middleware List ===
    middlewares: List[
        Union[
            UserTokenMiddleware,
            ErrorHandlingMiddleware,
            RateLimitingMiddleware,
            TimingMiddleware,
            LoggingMiddleware,
            JWTClaimsLoggingMiddleware,
            EunomiaMcpMiddleware,
        ]
    ] = [
        ErrorHandlingMiddleware(include_traceback=True, transform_errors=True),
        RateLimitingMiddleware(max_requests_per_second=10.0, burst_capacity=20),
        TimingMiddleware(),
        LoggingMiddleware(),
        JWTClaimsLoggingMiddleware(),
    ]
    if config["enable_delegation"] or args.auth_type == "jwt":
        middlewares.insert(0, UserTokenMiddleware(config=config))  # Must be first

    if args.eunomia_type in ["embedded", "remote"]:
        try:
            from eunomia_mcp import create_eunomia_middleware

            policy_file = args.eunomia_policy_file or "mcp_policies.json"
            eunomia_endpoint = (
                args.eunomia_remote_url if args.eunomia_type == "remote" else None
            )
            eunomia_mw = create_eunomia_middleware(
                policy_file=policy_file, eunomia_endpoint=eunomia_endpoint
            )
            middlewares.append(eunomia_mw)
            logger.info(f"Eunomia middleware enabled ({args.eunomia_type})")
        except Exception as e:
            print(f"Failed to load Eunomia middleware: {e}")
            logger.error("Failed to load Eunomia middleware", extra={"error": str(e)})
            sys.exit(1)

    mcp = FastMCP(name="GitRepositoryManager", auth=auth)
    register_tools(mcp)

    for mw in middlewares:
        mcp.add_middleware(mw)

    print("\nStarting Git MCP Server")
    print(f"  Transport: {args.transport.upper()}")
    print(f"  Auth: {args.auth_type}")
    print(f"  Delegation: {'ON' if config['enable_delegation'] else 'OFF'}")
    print(f"  Eunomia: {args.eunomia_type}")

    if args.transport == "stdio":
        mcp.run(transport="stdio")
    elif args.transport == "streamable-http":
        mcp.run(transport="streamable-http", host=args.host, port=args.port)
    elif args.transport == "sse":
        mcp.run(transport="sse", host=args.host, port=args.port)
    else:
        logger.error("Invalid transport", extra={"transport": args.transport})
        sys.exit(1)


if __name__ == "__main__":
    repository_manager_mcp()
