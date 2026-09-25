"""Launch the City View: ``python -m scalerl.dashboard`` (needs the dashboard extra)."""

import sys
from pathlib import Path

try:
    from streamlit.web import cli
except ImportError as error:
    raise SystemExit(
        'The City View needs the dashboard extra: pip install -e ".[dashboard]"'
    ) from error

sys.argv = ["streamlit", "run", str(Path(__file__).with_name("app.py")), *sys.argv[1:]]
sys.exit(cli.main())
