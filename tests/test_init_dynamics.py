import importlib
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
