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

# Défauts « démo » : la médiane brute (0 séjour honoré) fait exploser le risque
# car le modèle associe fortement l'absence d'historique à l'annulation.
APP_DEFAULTS = {
    "sejours_honores_passes": 2.0,
    "annulations_passees": 0.0,
    "origine_commerciale": "Direct",
    "nb_demandes_speciales": 1.0,
}


def _base_row() -> dict:
    row = {k: v for k, v in DEFAULTS.items() if k != "delai_jours_defaut"}
    row.update(APP_DEFAULTS)
    lead = 30  # délai court = profil plus « sûr » pour la démo
    today = date.today()
    row["date_reservation"] = today.isoformat()
    row["date_arrivee"] = (today + timedelta(days=lead)).isoformat()
    return row


def predict(
    site: str,
    date_reservation: str,
    date_arrivee: str,
    nuits_totales: float,
    nuits_week_end: float,
    nb_adultes: float,
    nb_enfants: float,
    garantie: str,
    canal_vente: str,
    client_fidele: str,
    tarif_nuit: float,
    formule: str,
    annulations_passees: float,
    sejours_honores_passes: float,
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
    if nuits_week_end < 0 or nuits_week_end > nuits_totales:
        return "Nuits week-end invalides (0 … nuits totales).", "—", 0.0
    if nb_adultes < 1 or nb_adultes > 20:
        return "Nombre d'adultes hors bornes (1–20).", "—", 0.0
    if nb_enfants < 0 or nb_enfants > 10:
        return "Nombre d'enfants hors bornes (0–10).", "—", 0.0
    if tarif_nuit < 0 or tarif_nuit > 1000:
        return "Tarif / nuit hors bornes (0–1000).", "—", 0.0
    if annulations_passees < 0 or annulations_passees > 50:
        return "Annulations passées hors bornes (0–50).", "—", 0.0
    if sejours_honores_passes < 0 or sejours_honores_passes > 100:
        return "Séjours honorés hors bornes (0–100).", "—", 0.0
    if not (0.0 <= seuil <= 1.0):
        return "Seuil hors bornes (0–1).", "—", 0.0

    n_tot = int(max(0, round(nuits_totales)))
    n_we = int(max(0, round(nuits_week_end)))
    n_we = min(n_we, n_tot)

    row = _base_row()
    row.update(
        {
            "site": site,
            "date_reservation": d_res.isoformat(),
            "date_arrivee": d_arr.isoformat(),
            "nuits_week_end": n_we,
            "nuits_semaine": n_tot - n_we,
            "nb_adultes": int(nb_adultes),
            "nb_enfants": int(nb_enfants),
            "garantie": garantie,
            "canal_vente": canal_vente,
            "client_fidele": 1 if client_fidele == "Oui" else 0,
            "tarif_nuit": float(tarif_nuit),
            "formule": formule,
            "annulations_passees": int(annulations_passees),
            "sejours_honores_passes": int(sejours_honores_passes),
        }
    )
    X = pd.DataFrame([row])[META["raw_features"]]
    proba = float(MODEL.predict_proba(X)[0, 1])
    decision = "Appeler le client" if proba >= seuil else "Ne pas appeler"
    return f"{proba:.1%}", decision, round(proba, 4)


def build_ui():
    today = date.today()
    with gr.Blocks(title="Résidences Azur — risque d'annulation") as demo:
        gr.Markdown(
            "## Résidences Azur — risque d'annulation\n"
            "Estime la probabilité d'annulation à l'enregistrement et recommande "
            "d'appeler ou non (seuil économique indicatif : **0,18**).\n\n"
            "Astuce : un client **sans séjour honoré** (0) est souvent au-dessus du seuil — "
            "teste avec 2–5 séjours passés pour voir « Ne pas appeler »."
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
                value="Direct",
                label="Canal de vente",
            )
            formule = gr.Dropdown(
                ["Petit-déjeuner", "Demi-pension", "Pension complète", "Sans repas"],
                value=str(DEFAULTS.get("formule", "Petit-déjeuner")),
                label="Formule",
            )
        with gr.Row():
            date_res = gr.Textbox(value=today.isoformat(), label="Date réservation (AAAA-MM-JJ)")
            date_arr = gr.Textbox(
                value=(today + timedelta(days=30)).isoformat(),
                label="Date arrivée (AAAA-MM-JJ)",
            )
            nuits = gr.Number(value=3, label="Nuits totales", precision=0)
            nuits_we = gr.Number(value=1, label="Dont nuits week-end", precision=0)
            adultes = gr.Number(value=2, label="Adultes", precision=0)
            enfants = gr.Number(
                value=int(DEFAULTS.get("nb_enfants", 0)),
                label="Enfants",
                precision=0,
            )
        with gr.Row():
            fidele = gr.Radio(["Non", "Oui"], value="Non", label="Client fidèle")
            tarif = gr.Number(value=float(DEFAULTS.get("tarif_nuit", 100)), label="Tarif / nuit (€)")
            annulations = gr.Number(value=0, label="Annulations passées", precision=0)
            sejours = gr.Number(value=2, label="Séjours honorés passés", precision=0)
            seuil = gr.Slider(0.05, 0.95, value=DEFAULT_THRESHOLD, step=0.01, label="Seuil d'appel")
        btn = gr.Button("Estimer", variant="primary")
        with gr.Row():
            proba_txt = gr.Textbox(label="Probabilité d'annulation")
            decision = gr.Textbox(label="Recommandation")
            proba_num = gr.Number(label="Proba (0–1)", precision=4)
        btn.click(
            predict,
            inputs=[
                site,
                date_res,
                date_arr,
                nuits,
                nuits_we,
                adultes,
                enfants,
                garantie,
                canal,
                fidele,
                tarif,
                formule,
                annulations,
                sejours,
                seuil,
            ],
            outputs=[proba_txt, decision, proba_num],
        )
        gr.Examples(
            examples=[
                [
                    "Urbaine",
                    today.isoformat(),
                    (today + timedelta(days=30)).isoformat(),
                    3,
                    1,
                    2,
                    0,
                    "Aucune",
                    "Direct",
                    "Non",
                    109,
                    "Petit-déjeuner",
                    0,
                    2,
                    0.18,
                ],
                [
                    "Littoral",
                    today.isoformat(),
                    (today + timedelta(days=120)).isoformat(),
                    2,
                    2,
                    2,
                    0,
                    "Remboursable",
                    "Intermédiaire",
                    "Non",
                    140,
                    "Petit-déjeuner",
                    2,
                    0,
                    0.18,
                ],
            ],
            inputs=[
                site,
                date_res,
                date_arr,
                nuits,
                nuits_we,
                adultes,
                enfants,
                garantie,
                canal,
                fidele,
                tarif,
                formule,
                annulations,
                sejours,
                seuil,
            ],
            label="Exemples : profil sûr vs profil risqué",
        )
    return demo


if __name__ == "__main__":
    build_ui().launch()
