from typing import List
import importlib
import sys

def safe_get_module_paths(module) -> List[str]:
    """Safely get module paths without accessing internal attributes"""
    try:
        # Use importlib to get the module spec
        if isinstance(module, str):
            spec = importlib.util.find_spec(module)
            if spec and spec.submodule_search_locations:
                return list(spec.submodule_search_locations)
        elif hasattr(module, '__spec__'):
            if module.__spec__ and module.__spec__.submodule_search_locations:
                return list(module.__spec__.submodule_search_locations)
        return []
    except Exception:
        return []

def patch_streamlit_watcher():
    """Patch streamlit's local sources watcher to avoid warnings"""
    try:
        import streamlit.watcher.local_sources_watcher as local_sources_watcher
        
        # Replace the module path getter
        def get_module_paths(module):
            if isinstance(module, str):
                try:
                    module = importlib.import_module(module)
                except ImportError:
                    return []
            return safe_get_module_paths(module)
        
        local_sources_watcher.get_module_paths = get_module_paths
    except ImportError:
        pass 