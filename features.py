"""Variables dérivées utilisées par le pipeline (train, app, api)."""
from __future__ import annotations

import pandas as pd

# Colonnes brutes attendues en entrée du pipeline (hors cible)
RAW_FEATURES = [
    "site",
    "date_reservation",
    "date_arrivee",
    "nuits_week_end",
    "nuits_semaine",
    "nb_adultes",
    "nb_enfants",
    "nb_bebes",
    "formule",
    "pays_client",
    "origine_commerciale",
    "canal_vente",
    "client_fidele",
    "annulations_passees",
    "sejours_honores_passes",
    "logement_demande",
    "garantie",
    "id_agence",
    "profil_client",
    "tarif_nuit",
    "nb_places_parking",
    "nb_demandes_speciales",
    "code_promo",
    "indice_affinite",
]

# Exclues (fuite / identifiant / post-réservation) :
# id_reservation, annule, statut_dossier, date_statut,
# jours_attente_confirmation, logement_attribue, nb_modifications, id_entreprise

NUMERIC_FEATURES = [
    "nuits_week_end",
    "nuits_semaine",
    "nb_adultes",
    "nb_enfants",
    "nb_bebes",
    "client_fidele",
    "annulations_passees",
    "sejours_honores_passes",
    "tarif_nuit",
    "nb_places_parking",
    "nb_demandes_speciales",
    "indice_affinite",
    "delai_jours",
    "nuits_totales",
    "taille_groupe",
    "mois_arrivee",
    "ratio_weekend",
    "via_agence",
    "taux_annulation_passe",
]

CATEGORICAL_FEATURES = [
    "site",
    "formule",
    "pays_client",
    "origine_commerciale",
    "canal_vente",
    "logement_demande",
    "garantie",
    "profil_client",
    "code_promo",
]


def add_features(X: pd.DataFrame) -> pd.DataFrame:
    """Ajoute au moins 3 variables dérivées à partir des dates et de la composition."""
    out = X.copy()
    for col in ("date_reservation", "date_arrivee"):
        out[col] = pd.to_datetime(out[col], errors="coerce")

    out["delai_jours"] = (out["date_arrivee"] - out["date_reservation"]).dt.days
    out["delai_jours"] = out["delai_jours"].clip(lower=0)

    out["nuits_totales"] = out["nuits_week_end"].fillna(0) + out["nuits_semaine"].fillna(0)
    out["taille_groupe"] = (
        out["nb_adultes"].fillna(0)
        + out["nb_enfants"].fillna(0)
        + out["nb_bebes"].fillna(0)
    )
    out["mois_arrivee"] = out["date_arrivee"].dt.month.fillna(7).astype(int)
    out["ratio_weekend"] = out["nuits_week_end"].fillna(0) / out["nuits_totales"].clip(lower=1)
    out["via_agence"] = out["id_agence"].notna().astype(int)

    denom = (
        out["annulations_passees"].fillna(0)
        + out["sejours_honores_passes"].fillna(0)
        + 1
    )
    out["taux_annulation_passe"] = out["annulations_passees"].fillna(0) / denom

    # Regrouper les pays rares (conserve le top, le reste → AUTRE)
    if "pays_client" in out.columns:
        top = {
            "FRA",
            "GBR",
            "PRT",
            "ESP",
            "DEU",
            "ITA",
            "IRL",
            "BEL",
            "BRA",
            "NLD",
            "USA",
            "CHE",
            "CN",
            "AUT",
            "SWE",
            "POL",
            "CHN",
            "ISR",
            "RUS",
            "NOR",
        }
        pays = out["pays_client"].fillna("INCONNU").astype(str)
        out["pays_client"] = pays.where(pays.isin(top) | (pays == "INCONNU"), "AUTRE")

    if "code_promo" in out.columns:
        out["code_promo"] = out["code_promo"].fillna("AUCUN").astype(str)

    for col in CATEGORICAL_FEATURES:
        if col in out.columns:
            out[col] = out[col].fillna("INCONNU").astype(str)

    return out
