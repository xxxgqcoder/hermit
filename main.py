import logging
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI

from hermit.api.routes import router
from hermit.config import HOST, PORT
from hermit.retrieval import embedder, reranker

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting Hermit — loading models...")
    embedder.warmup()
    reranker.warmup()
    logger.info("Hermit ready.")
    yield
    logger.info("Shutting down Hermit.")


app = FastAPI(title="Hermit", version="0.1.0", lifespan=lifespan)
app.include_router(router)


if __name__ == "__main__":
    uvicorn.run("main:app", host=HOST, port=PORT, reload=False)

