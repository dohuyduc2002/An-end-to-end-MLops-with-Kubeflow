import sys
from pathlib import Path
import pytest
import random
from faker import Faker
import pandas as pd
import json
from unittest.mock import MagicMock
from contextlib import asynccontextmanager

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from client.api.schema import RawItem

fake = Faker()

FAKE_PRED_BATCH = [
    {"result": "Accept", "prob_accept": 0.7, "prob_decline": 0.3},
    {"result": "Decline", "prob_accept": 0.2, "prob_decline": 0.8},
]
FAKE_PRED_BY_ID = [{"result": "Accept", "prob_accept": 0.95, "prob_decline": 0.05}]


@pytest.fixture
def sample_payload():
    payload_file = ROOT / "src" / "payload" / "sample_payload.json"
    return json.loads(payload_file.read_text())


def generate_fake_value(origin_payload_type):
    if origin_payload_type == int:
        return random.randint(0, 100)
    if origin_payload_type == float:
        return round(random.uniform(0.0, 10000.0), 2)
    if origin_payload_type == str:
        return fake.word()
    return None


def build_fake_rawitem_dict():
    from typing import Union, get_origin, get_args

    data = {}
    for field, ann in RawItem.__annotations__.items():
        origin, args = get_origin(ann), get_args(ann)
        origin_payload_type = args[0] if origin is Union and type(None) in args else ann
        val = generate_fake_value(origin_payload_type)
        if val is not None:
            data[field] = val
    return data


@pytest.fixture
def fake_csv(tmp_path: Path):
    rows = [build_fake_rawitem_dict() for _ in range(10)]
    df = pd.DataFrame(rows)
    df["TARGET"] = [0, 1] * (len(df) // 2) + [0] * (len(df) % 2)
    dst = tmp_path / "fake.csv"
    df.to_csv(dst, index=False)
    return dst


@pytest.fixture
def mock_predict_batch(mocker):
    return mocker.patch(
        "src.client.api.main.predict_batch", return_value=FAKE_PRED_BATCH
    )


@pytest.fixture
def mock_predict_by_id(mocker):
    return mocker.patch(
        "src.client.api.main.predict_by_id", return_value=FAKE_PRED_BY_ID
    )


@pytest.fixture
def fake_lifespan():
    @asynccontextmanager
    async def _lifespan(app):
        app.state.binning = MagicMock()
        app.state.selector = MagicMock()
        app.state.model = MagicMock()
        yield

    return _lifespan


@pytest.fixture
def mock_minio_client(mocker):
    mocker.patch(
        "src.client.api.main.ApiConfig.get_minio_client", return_value=MagicMock()
    )
