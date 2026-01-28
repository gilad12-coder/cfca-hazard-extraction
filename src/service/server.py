import os

import uvicorn


def main() -> None:
    """Run the API server, binding to 0.0.0.0 on the configured port."""
    port = int(os.getenv("PORT", "8080"))
    reload = os.getenv("UVICORN_RELOAD", "false").lower() == "true"
    uvicorn.run(
        "src.service.extraction_service:app",
        host="0.0.0.0",
        port=port,
        reload=reload,
    )


if __name__ == "__main__":
    main()
