"""
User Interface (UI) subsystem for dens-city.
Provides high-performance 3D Raylib molecular visualization, non-blocking cDFT and BG solvers,
and unified CLI tools.
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
from dens_city.ui.worker import (
    CDFTBGWorker,
    TelemetryData,
    compute_end_to_end_distance,
    compute_radius_of_gyration,
    count_steric_clashes,
)

__all__ = [
    "MoleculeViewer",
    "CDFTBGWorker",
    "TelemetryData",
    "compute_radius_of_gyration",
    "compute_end_to_end_distance",
    "count_steric_clashes",
    "run_interactive_viewer",
    "main",
    "ELEMENT_COLORS",
    "ELEMENT_RADII",
    "get_atom_element",
    "get_atom_color",
    "get_atom_radius",
]
