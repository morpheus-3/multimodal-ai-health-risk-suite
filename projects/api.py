import json
from functools import lru_cache
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import StreamingResponse, HTMLResponse
from pydantic import BaseModel, Field, ConfigDict
from . import health
from .common import ROOT, metadata, groq_explain
from pathlib import Path

app = FastAPI(title='Health Risk Lab', version="1.0.0")

@app.get("/", response_class=HTMLResponse)
def home():
    return (Path(__file__).parent / "index.html").read_text(encoding="utf-8")

@app.get("/status")
def status():
    report = ROOT / "health" / "report.json"
    return metadata(report) if report.exists() else {"trained": False}

@app.exception_handler(FileNotFoundError)
async def missing_model(request, exc):
    from fastapi.responses import JSONResponse
    return JSONResponse(status_code=503, content={"detail": "Model not trained. Run the documented training command first."})

@app.exception_handler(ValueError)
async def invalid_input(request, exc):
    from fastapi.responses import JSONResponse
    return JSONResponse(status_code=422, content={"detail": str(exc)})

class HealthInput(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    age: float = Field(ge=18, le=100)
    bmi: float = Field(ge=10, le=70)
    systolic_bp: float = Field(ge=70, le=250)
    glucose: float = Field(ge=40, le=500)
    cholesterol: float = Field(ge=80, le=500)
    smoker: int = Field(ge=0, le=1)
    activity_minutes: float = Field(ge=0, le=300)

class ExplanationInput(BaseModel):
    context: dict

@app.post("/health/predict")
def health_predict(body: HealthInput):
    return health.predict(body.model_dump())

@app.post("/explain")
def explain(body: ExplanationInput):
    if len(json.dumps(body.context)) > 12000:
        raise HTTPException(413, "Context too large.")
    try:
        return {"provider": "Groq", "explanation": groq_explain(body.context)}
    except ValueError as exc:
        raise HTTPException(503, str(exc))
    except Exception:
        raise HTTPException(502, "Groq request failed; check server configuration and quota.")
