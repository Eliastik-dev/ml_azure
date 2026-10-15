"""Tests Gradio + API (pytest)."""
from __future__ import annotations

from datetime import date, timedelta

from fastapi.testclient import TestClient

from api import api
from app import predict


def test_app_probability_between_0_and_1():
    today = date.today()
    txt, decision, proba = predict(
        site="Urbaine",
        date_reservation=today.isoformat(),
        date_arrivee=(today + timedelta(days=30)).isoformat(),
        nuits_totales=3,
        nb_adultes=2,
        garantie="Aucune",
        canal_vente="Intermédiaire",
        client_fidele="Non",
        tarif_nuit=120,
        formule="Petit-déjeuner",
        seuil=0.18,
    )
    assert isinstance(txt, str)
    assert decision in ("Appeler le client", "Ne pas appeler")
    assert 0.0 <= float(proba) <= 1.0


def test_app_rejects_impossible_dates():
    today = date.today()
    msg, _, _ = predict(
        site="Urbaine",
        date_reservation=today.isoformat(),
        date_arrivee=(today - timedelta(days=1)).isoformat(),
        nuits_totales=2,
        nb_adultes=2,
        garantie="Aucune",
        canal_vente="Direct",
        client_fidele="Non",
        tarif_nuit=100,
        formule="Petit-déjeuner",
        seuil=0.18,
    )
    assert "arrivée" in msg.lower() or "≥" in msg


def test_api_predict_ok():
    client = TestClient(api)
    today = date.today()
    payload = {
        "site": "Littoral",
        "date_reservation": today.isoformat(),
        "date_arrivee": (today + timedelta(days=45)).isoformat(),
        "nuits_semaine": 4,
        "nuits_week_end": 2,
        "nb_adultes": 2,
        "garantie": "Aucune",
        "canal_vente": "Direct",
        "tarif_nuit": 95,
    }
    r = client.post("/predict", json=payload)
    assert r.status_code == 200
    body = r.json()
    assert 0.0 <= body["probabilite_annulation"] <= 1.0
    assert body["recommandation"] in ("appeler", "ne_pas_appeler")


def test_api_invalid_returns_422():
    client = TestClient(api)
    today = date.today()
    payload = {
        "site": "Urbaine",
        "date_reservation": today.isoformat(),
        "date_arrivee": (today - timedelta(days=5)).isoformat(),  # invalide
        "nb_adultes": 2,
    }
    r = client.post("/predict", json=payload)
    assert r.status_code == 422
