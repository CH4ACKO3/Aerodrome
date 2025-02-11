"""Functions for registering environments"""

# Global registry of environments.
registry: dict[str, str] = {}

def register(id: str, entry_point: str):
    """Register a new environment by ID"""
    registry[id] = entry_point
