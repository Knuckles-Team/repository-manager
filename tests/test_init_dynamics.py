import importlib
import inspect
import runpy
import sys
from unittest.mock import MagicMock, patch

import pytest


def test_dynamic_attributes_and_imports():
    """Verify dynamic loading and attribute availability flags in repository_manager."""
    original_import = importlib.import_module

    class MockClass:
        pass

    def mock_import_func(name):
        if name in ["repository_manager.agent_server", "repository_manager.mcp_server"]:
            mock_mod = MagicMock()
            mock_mod.SomeClass = MockClass
            return mock_mod
        return original_import(name)

    with patch("importlib.import_module", side_effect=mock_import_func):
        if "repository_manager" in sys.modules:
            importlib.reload(sys.modules["repository_manager"])
        else:
            import repository_manager

        import repository_manager

        # Check dynamic variables
        assert repository_manager._MCP_AVAILABLE is True
        assert repository_manager._AGENT_AVAILABLE is True

        # Test getting attribute from optional module
        val = repository_manager.SomeClass
        assert val is MockClass

        # Test __dir__
        d = dir(repository_manager)
        assert "SomeClass" in d


def test_dynamic_attributes_import_failure():
    """Verify fallback behavior when optional modules throw ImportError."""
    original_import = importlib.import_module

    def mock_import_failure(name):
        if name in ["repository_manager.agent_server", "repository_manager.mcp_server"]:
            raise ImportError("Mocked import error")
        return original_import(name)

    with patch("importlib.import_module", side_effect=mock_import_failure):
        if "repository_manager" in sys.modules:
            importlib.reload(sys.modules["repository_manager"])
        else:
            import repository_manager

        import repository_manager

        assert repository_manager._MCP_AVAILABLE is False
        assert repository_manager._AGENT_AVAILABLE is False

        with pytest.raises(AttributeError):
            _ = repository_manager.NonexistentClass


def test_main_entrypoint():
    """Import and test the __main__.py file in isolation with runpy."""
    with patch("repository_manager.repository_manager.main") as mock_main:
        runpy.run_module("repository_manager.__main__", run_name="__main__")
        mock_main.assert_called_once()

    # Clean up to avoid polluting other modules
    if "repository_manager" in sys.modules:
        importlib.reload(sys.modules["repository_manager"])


def test_expose_members_never_shadows_a_submodule_name():
    """D-SKC-4: ``mcp_server.py`` exports a public function literally named
    ``mcp_server`` (its CLI entrypoint). ``_expose_members`` must never bind
    that function into ``repository_manager`` package globals -- doing so
    shadows the ``repository_manager.mcp_server`` MODULE for any later
    ``from repository_manager import mcp_server``, which is exactly what
    made ``tests/test_action_discovery.py`` (which does
    ``import repository_manager.mcp_server as mcp_server`` then reads
    ``mcp_server.RM_GIT_ACTIONS``, a module-level constant) fail collection
    ONLY when some other test file exposing the optional modules ran first
    in the same process -- a verdict that depended on invocation order, not
    on the code. Reverting the ``_RESERVED_SUBMODULE_NAMES`` skip in
    ``_expose_members`` makes this go red.
    """
    import repository_manager as rm
    from repository_manager import mcp_server as mcp_server_module

    # Force the lazy optional-module exposure the real bug depends on --
    # touching ANY missing attribute triggers it for every optional module,
    # exactly like an earlier, unrelated test in the same pytest process
    # would (the invocation-order dependence this item reports).
    _ = rm._MCP_AVAILABLE

    assert inspect.ismodule(mcp_server_module)
    assert hasattr(mcp_server_module, "RM_GIT_ACTIONS")
    # The package attribute itself must resolve to the submodule too, not a
    # bare function that shadows it.
    assert inspect.ismodule(rm.mcp_server)
