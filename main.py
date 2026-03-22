"""Dev shim — delegates to hermit.app.

For production use: hermit start (after installing with uv tool install)
"""

import uvicorn

from hermit.app import app  # noqa: F401
from hermit.config import HOST, PORT

if __name__ == "__main__":
    uvicorn.run("hermit.app:app", host=HOST, port=PORT, reload=False)

