"""
User Interface (UI) subsystem for dens-city.
Provides high-performance 3D Raylib molecular visualization and unified CLI tools.
"""

from dens_city.ui.cli import main
from dens_city.ui.viewer import (
    ELEMENT_COLORS,
    ELEMENT_RADII,
    MoleculeViewer,
    get_atom_color,
    get_atom_element,
    get_atom_radius,
    run_interactive_viewer,
)

__all__ = [
    "MoleculeViewer",
    "run_interactive_viewer",
    "main",
    "ELEMENT_COLORS",
    "ELEMENT_RADII",
    "get_atom_element",
    "get_atom_color",
    "get_atom_radius",
]
