"""API FastAPI : POST /predict."""
from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Literal

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field, model_validator

import features  # noqa: F401 — requis pour unpickler FunctionTransformer

ROOT = Path(__file__).resolve().parent
MODEL = joblib.load(ROOT / "model.pkl")
META = json.loads((ROOT / "model_meta.json").read_text(encoding="utf-8"))
DEFAULTS = META["defaults"]

api = FastAPI(title="Résidences Azur — prédiction d'annulation", version="1.0.0")


class ReservationIn(BaseModel):
    site: Literal["Urbaine", "Littoral"] = "Urbaine"
    date_reservation: date
    date_arrivee: date
    nuits_week_end: int = Field(0, ge=0, le=30)
    nuits_semaine: int = Field(2, ge=0, le=60)
    nb_adultes: int = Field(2, ge=1, le=20)
    nb_enfants: float = Field(0, ge=0, le=10)
    nb_bebes: int = Field(0, ge=0, le=5)
    formule: str = "Petit-déjeuner"
    pays_client: str = "FRA"
    origine_commerciale: str = "Direct"
    canal_vente: str = "Intermédiaire"
    client_fidele: int = Field(0, ge=0, le=1)
    annulations_passees: int = Field(0, ge=0, le=50)
    sejours_honores_passes: int = Field(0, ge=0, le=100)
    logement_demande: str = "L10"
    garantie: Literal["Aucune", "Non remboursable", "Remboursable"] = "Aucune"
    id_agence: float | None = None
    profil_client: str = "Individuel"
    tarif_nuit: float = Field(100, ge=0, le=1000)
    nb_places_parking: int = Field(0, ge=0, le=10)
    nb_demandes_speciales: int = Field(0, ge=0, le=20)
    code_promo: str = "AZ493"
    indice_affinite: float = Field(0.5, ge=0, le=1)

    @model_validator(mode="after")
    def check_dates(self):
        if self.date_arrivee < self.date_reservation:
            raise ValueError("date_arrivee doit être ≥ date_reservation")
        return self


class PredictionOut(BaseModel):
    probabilite_annulation: float
    seuil: float
    recommandation: str


def _row_from_payload(payload: ReservationIn) -> pd.DataFrame:
    row = {k: v for k, v in DEFAULTS.items() if k != "delai_jours_defaut"}
    data = payload.model_dump()
    data["date_reservation"] = payload.date_reservation.isoformat()
    data["date_arrivee"] = payload.date_arrivee.isoformat()
    row.update(data)
    return pd.DataFrame([row])[META["raw_features"]]


@api.get("/")
def root():
    return {"service": "residences-azur", "docs": "/docs"}


@api.post("/predict", response_model=PredictionOut)
def predict(payload: ReservationIn):
    X = _row_from_payload(payload)
    proba = float(MODEL.predict_proba(X)[0, 1])
    seuil = float(META.get("threshold", 0.18))
    reco = "appeler" if proba >= seuil else "ne_pas_appeler"
    return PredictionOut(
        probabilite_annulation=round(proba, 6),
        seuil=seuil,
        recommandation=reco,
    )
