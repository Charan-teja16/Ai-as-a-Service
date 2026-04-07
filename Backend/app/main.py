from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .routers.api import router as api_router
from .routers.auth import router as auth_router

app = FastAPI(
    title="AI-as-a-Service",
    description="Code-free ML training and prediction platform.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth_router)
app.include_router(api_router)


@app.get("/health")
async def health_check():
    return {"status": "ok"}

