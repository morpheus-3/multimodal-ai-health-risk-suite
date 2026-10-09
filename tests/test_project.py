import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL","2")
os.environ.setdefault("TF_NUM_INTRAOP_THREADS","2")
os.environ.setdefault("TF_NUM_INTEROP_THREADS","2")
import numpy as np
import pytest
from fastapi.testclient import TestClient
from projects import common, health
from projects.fixtures import generate

@pytest.fixture(scope="session")
def data(tmp_path_factory):
    return generate(tmp_path_factory.mktemp("data"))

@pytest.fixture(scope="session", autouse=True)
def artifact_root(tmp_path_factory):
    previous = common.ROOT
    common.ROOT = tmp_path_factory.mktemp("models")
    from projects import api
    api.ROOT = common.ROOT
    yield common.ROOT
    common.ROOT = previous

@pytest.fixture(scope="session")
def trained_health(data):
    return health.train(data / "health.csv",epochs=2,synthetic=True)

def test_health_training_and_explanation(trained_health, data):
    import pandas as pd
    assert trained_health["synthetic"] is True
    assert set(trained_health["targets"]) == set(health.TARGETS)
    row = pd.read_csv(data / "health.csv").iloc[0].to_dict()
    output = health.predict(row)
    for target in health.TARGETS:
        result = output["results"][target]
        assert 0 <= result["probability"] <= 1
        assert set(result["shap"]) == set(health.FEATURES)
        assert abs(result["base_probability"] + sum(result["shap"].values()) - result["probability"]) < .015
        assert 0 <= trained_health["targets"][target]["roc_auc"] <= 1

def test_health_rejects_invalid_values():
    with pytest.raises(ValueError):
        health.validate(dict(zip(health.FEATURES,[40,25,120,float("nan"),190,0,20])))

def test_groq_missing_key_and_provider_contract(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY",raising=False)
    from projects.api import app
    assert TestClient(app).post("/explain",json={"context":{}}).status_code == 503
    captured={}
    class FakeResponse:
        def raise_for_status(self): pass
        def json(self): return {"choices":[{"message":{"content":"Model explanation"}}]}
    def post(self,url,**kwargs):
        captured.update(url=url,**kwargs)
        return FakeResponse()
    monkeypatch.setenv("GROQ_API_KEY","test-key-not-real")
    monkeypatch.setattr(common.httpx.Client,"post",post)
    assert common.groq_explain({"score":.5}) == "Model explanation"
    assert captured["url"].startswith("https://api.groq.com/")
    assert captured["headers"]["Authorization"] == "Bearer test-key-not-real"

def test_home_status_and_invalid_input():
    from projects.api import app
    client=TestClient(app)
    assert client.get("/").status_code==200
    assert "Health Risk Lab" in client.get("/").text
    assert client.get("/status").status_code==200
    assert client.post("/health/predict",json={"age":999}).status_code==422
