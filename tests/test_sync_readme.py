import subprocess
import sys
from pathlib import Path


def test_sync_readme_deterministic_check():
    """Verify that sync_readme_metrics.py runs cleanly and README is in sync with runs/ tracking records."""
    repo_root = Path(__file__).parent.parent
    script_path = repo_root / "scripts" / "sync_readme_metrics.py"
    assert script_path.exists(), "sync_readme_metrics.py must exist in scripts/"

    # Run --check
    res = subprocess.run([sys.executable, str(script_path), "--check"], capture_output=True, text=True)
    assert res.returncode == 0, f"README is out of sync with runs/ records:\nSTDOUT: {res.stdout}\nSTDERR: {res.stderr}"
