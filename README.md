# Projet Résidences Azur — Groupe 2 (MLV)

Anticiper les annulations de séjour et recommander un appel commercial.

## Contenu

| Fichier | Rôle |
|---|---|
| `notebook.ipynb` | Démarche complète (EDA → échelle → évaluation) |
| `audit_variables.md` | Audit disponibilité / fuite / utilité |
| `features.py` | Variables dérivées du pipeline |
| `train.py` | Entraîne et sauve `model.pkl` + `model_meta.json` |
| `app.py` | Interface Gradio |
| `api.py` | API FastAPI (`uvicorn api:api`) |
| `test_app.py` | Tests pytest |
| `note_methodologique.md` | Note 2–3 pages |
| `soutenance.pptx` | Support de soutenance (7 slides) |
| `model.pkl` / `model_meta.json` | Pipeline + métadonnées (seuil, défauts) |
| `requirements.txt` | Dépendances figées |

Données : `../data/reservations_azur.csv`

## Lancer

```text
pip install -r requirements.txt
python train.py
uvicorn api:api --reload
python app.py
pytest
```

Seuil économique par défaut : **0,18**.
