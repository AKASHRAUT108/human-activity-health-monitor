"""FastAPI application for the Human Activity Health Monitor."""

from __future__ import annotations

import tempfile
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from starlette.concurrency import run_in_threadpool

from api.inference_service import VideoInferenceService


ALLOWED_EXTENSIONS = {
    ".mp4",
    ".avi",
    ".mov",
    ".mkv",
}

MAX_UPLOAD_SIZE = 200 * 1024 * 1024
UPLOAD_CHUNK_SIZE = 1024 * 1024


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load inference components once when the application starts."""

    service = VideoInferenceService()
    app.state.inference_service = service

    try:
        yield
    finally:
        service.close()


app = FastAPI(
    title="Human Activity Health Monitor API",
    description=(
        "Video-based human activity recognition using MediaPipe Pose "
        "and a trained LSTM model."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/")
async def root():
    return {
        "message": "Human Activity Health Monitor API",
        "status": "running",
        "docs": "/docs",
    }


@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "model": "human_activity_lstm",
        "model_directory": "models/kth_tuned_conf03",
    }


@app.post("/predict")
async def predict_video(file: UploadFile = File(...)):
    """Upload a video and return its activity predictions."""

    filename = file.filename or "uploaded_video"
    extension = Path(filename).suffix.lower()

    if extension not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=415,
            detail=(
                "Unsupported video format. Allowed formats: "
                "MP4, AVI, MOV, MKV."
            ),
        )

    temporary_path = None
    total_size = 0

    try:
        with tempfile.NamedTemporaryFile(
            suffix=extension,
            prefix="activity_upload_",
            delete=False,
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)

            while True:
                chunk = await file.read(UPLOAD_CHUNK_SIZE)

                if not chunk:
                    break

                total_size += len(chunk)

                if total_size > MAX_UPLOAD_SIZE:
                    raise HTTPException(
                        status_code=413,
                        detail="Video exceeds the 200 MB upload limit.",
                    )

                temporary_file.write(chunk)

        if total_size == 0:
            raise HTTPException(
                status_code=400,
                detail="The uploaded video is empty.",
            )

        service = app.state.inference_service

        result = await run_in_threadpool(
            service.predict_video,
            temporary_path,
        )

        return result

    except HTTPException:
        raise

    except ValueError as exc:
        raise HTTPException(
            status_code=422,
            detail=str(exc),
        ) from exc

    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=500,
            detail=str(exc),
        ) from exc

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Video inference failed: {exc}",
        ) from exc

    finally:
        await file.close()

        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
