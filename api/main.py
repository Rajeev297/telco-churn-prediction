from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routes import router
from api.audit import setup_audit_logging
from src.config import load_config, get


def create_app() -> FastAPI:
    load_config()
    setup_audit_logging()

    app = FastAPI(
        title=get("app.name", "Churn Analytics Platform"),
        version=get("app.version", "2.0.0"),
        docs_url="/docs",
        redoc_url="/redoc",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(router, prefix="/api/v1")

    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api.main:app",
        host=get("api.host", "0.0.0.0"),
        port=get("api.port", 8000),
        reload=get("app.debug", False),
    )
