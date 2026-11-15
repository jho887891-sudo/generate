from __future__ import annotations

import os

import uvicorn


if __name__ == "__main__":
    uvicorn.run(
        "svp_text_app.main:app",
        host=os.environ.get("SVP_TEXT_HOST", "0.0.0.0"),
        port=int(os.environ.get("SVP_TEXT_PORT", "8088")),
        reload=False,
        workers=1,
    )

