import importlib
import sys


def test_importing_web_utils_does_not_import_agent_backend() -> None:
    module_names = [
        "agents",
        "agents.data_collection",
        "agents.data_collection.data_collection_agent",
        "agents.data_collection.smolagents_backend",
        "agents.data_collection.web_utils",
        "smolagents",
    ]
    saved_modules = {module_name: sys.modules.get(module_name) for module_name in module_names}
    try:
        for module_name in module_names:
            sys.modules.pop(module_name, None)

        importlib.import_module("agents.data_collection.web_utils")

        assert "agents.data_collection.data_collection_agent" not in sys.modules
        assert "agents.data_collection.smolagents_backend" not in sys.modules
        assert "smolagents" not in sys.modules
    finally:
        for module_name in module_names:
            sys.modules.pop(module_name, None)
        for module_name, module in saved_modules.items():
            if module is not None:
                sys.modules[module_name] = module


def test_importing_pdf_utils_does_not_import_agent_backend() -> None:
    module_names = [
        "agents",
        "agents.data_collection",
        "agents.data_collection.data_collection_agent",
        "agents.data_collection.smolagents_backend",
        "agents.data_collection.pdf_utils",
        "smolagents",
    ]
    saved_modules = {module_name: sys.modules.get(module_name) for module_name in module_names}
    try:
        for module_name in module_names:
            sys.modules.pop(module_name, None)

        importlib.import_module("agents.data_collection.pdf_utils")

        assert "agents.data_collection.data_collection_agent" not in sys.modules
        assert "agents.data_collection.smolagents_backend" not in sys.modules
        assert "smolagents" not in sys.modules
    finally:
        for module_name in module_names:
            sys.modules.pop(module_name, None)
        for module_name, module in saved_modules.items():
            if module is not None:
                sys.modules[module_name] = module
