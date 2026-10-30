"""Project-local paths; never search sibling checkouts."""

from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DATA_DIR = str(REPO / "data")
URDF_PATH = str(REPO / "data/g1_description/g1_29dof_rev_1_0.urdf")
MJCF_PATH = str(REPO / "data/g1_description/g1_29dof.xml")
USD_DIR = str(REPO / "data/g1_usd")
USD_PATH = str(REPO / "data/g1_usd/g1_29dof/g1_29dof.usda")
