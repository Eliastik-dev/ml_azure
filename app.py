"""Application Gradio : probabilité d'annulation + recommandation d'appel."""
from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path

import gradio as gr
import joblib
import pandas as pd

import features  # noqa: F401 — requis pour unpickler FunctionTransformer

ROOT = Path(__file__).resolve().parent
MODEL = joblib.load(ROOT / "model.pkl")
META = json.loads((ROOT / "model_meta.json").read_text(encoding="utf-8"))
DEFAULTS = META["defaults"]
DEFAULT_THRESHOLD = float(META.get("threshold", 0.18))


def _base_row() -> dict:
    row = {k: v for k, v in DEFAULTS.items() if k != "delai_jours_defaut"}
    lead = int(DEFAULTS.get("delai_jours_defaut", 67))
    today = date.today()
    row["date_reservation"] = today.isoformat()
    row["date_arrivee"] = (today + timedelta(days=lead)).isoformat()
    return row


def predict(
    site: str,
    date_reservation: str,
    date_arrivee: str,
    nuits_totales: float,
    nb_adultes: float,
    garantie: str,
    canal_vente: str,
    client_fidele: str,
    tarif_nuit: float,
    formule: str,
    seuil: float,
):
    try:
        d_res = date.fromisoformat(str(date_reservation)[:10])
        d_arr = date.fromisoformat(str(date_arrivee)[:10])
    except ValueError:
        return "Dates invalides (format AAAA-MM-JJ).", "—", 0.0

    if d_arr < d_res:
        return "La date d'arrivée doit être ≥ date de réservation.", "—", 0.0
    if nuits_totales < 0 or nuits_totales > 60:
        return "Nombre de nuits hors bornes (0–60).", "—", 0.0
    if nb_adultes < 1 or nb_adultes > 20:
        return "Nombre d'adultes hors bornes (1–20).", "—", 0.0
    if tarif_nuit < 0 or tarif_nuit > 1000:
        return "Tarif / nuit hors bornes (0–1000).", "—", 0.0
    if not (0.0 <= seuil <= 1.0):
        return "Seuil hors bornes (0–1).", "—", 0.0

    row = _base_row()
    row.update(
        {
            "site": site,
            "date_reservation": d_res.isoformat(),
            "date_arrivee": d_arr.isoformat(),
            "nuits_semaine": int(max(0, round(nuits_totales))),
            "nuits_week_end": 0,
            "nb_adultes": int(nb_adultes),
            "garantie": garantie,
            "canal_vente": canal_vente,
            "client_fidele": 1 if client_fidele == "Oui" else 0,
            "tarif_nuit": float(tarif_nuit),
            "formule": formule,
        }
    )
    X = pd.DataFrame([row])[META["raw_features"]]
    proba = float(MODEL.predict_proba(X)[0, 1])
    decision = "Appeler le client" if proba >= seuil else "Ne pas appeler"
    return f"{proba:.1%}", decision, round(proba, 4)


def build_ui():
    with gr.Blocks(title="Résidences Azur — risque d'annulation") as demo:
        gr.Markdown(
            "## Résidences Azur — risque d'annulation\n"
            "Estime la probabilité d'annulation à l'enregistrement et recommande "
            "d'appeler ou non (seuil économique indicatif : **0,18**)."
        )
        with gr.Row():
            site = gr.Dropdown(["Urbaine", "Littoral"], value="Urbaine", label="Site")
            garantie = gr.Dropdown(
                ["Aucune", "Non remboursable", "Remboursable"],
                value=str(DEFAULTS.get("garantie", "Aucune")),
                label="Garantie",
            )
            canal = gr.Dropdown(
                ["Intermédiaire", "Direct", "Entreprise", "Réseau de distribution", "Non renseigné"],
                value=str(DEFAULTS.get("canal_vente", "Intermédiaire")),
                label="Canal de vente",
            )
            formule = gr.Dropdown(
                ["Petit-déjeuner", "Demi-pension", "Pension complète", "Sans repas"],
                value=str(DEFAULTS.get("formule", "Petit-déjeuner")),
                label="Formule",
            )
        with gr.Row():
            date_res = gr.Textbox(value=date.today().isoformat(), label="Date réservation (AAAA-MM-JJ)")
            date_arr = gr.Textbox(
                value=(date.today() + timedelta(days=67)).isoformat(),
                label="Date arrivée (AAAA-MM-JJ)",
            )
            nuits = gr.Number(value=2, label="Nuits totales", precision=0)
            adultes = gr.Number(value=2, label="Adultes", precision=0)
        with gr.Row():
            fidele = gr.Radio(["Non", "Oui"], value="Non", label="Client fidèle")
            tarif = gr.Number(value=float(DEFAULTS.get("tarif_nuit", 100)), label="Tarif / nuit (€)")
            seuil = gr.Slider(0.05, 0.95, value=DEFAULT_THRESHOLD, step=0.01, label="Seuil d'appel")
        btn = gr.Button("Estimer", variant="primary")
        with gr.Row():
            proba_txt = gr.Textbox(label="Probabilité d'annulation")
            decision = gr.Textbox(label="Recommandation")
            proba_num = gr.Number(label="Proba (0–1)", precision=4)
        btn.click(
            predict,
            inputs=[site, date_res, date_arr, nuits, adultes, garantie, canal, fidele, tarif, formule, seuil],
            outputs=[proba_txt, decision, proba_num],
        )
    return demo


if __name__ == "__main__":
    build_ui().launch()
