"""Serverless entrypoint.

Vercel's Python runtime imports ``app`` from this module and serves it as
a WSGI application. Nothing is computed here: the app reads the
committed serving database (data/web/plpredict-web.db), which is exported
without a write-ahead log so it can be opened read-only on a host that
gives the function a read-only filesystem.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from plpredict.web.app import create_app  # noqa: E402

app = create_app()
