# Note méthodologique

Groupe : 2 · Classe : MLV · Membres : Hyndi FANNIR, Marco SALGADO SOARES DE FARIA, Frederic TOUFIK

## 1. Problème et décision métier

Environ un séjour sur trois est annulé chez Résidences Azur. À l’enregistrement, on estime la **probabilité d’annulation** `p` et on décide d’appeler le client si `p` dépasse un seuil.

Ordre de grandeur économique : un appel coûte 6 € et évite l’annulation dans un cas sur trois ; une annulation évitée vaut ~100 €. On appelle si `p × (1/3) × 100 > 6`, soit **`p > 0,18`**. Le seuil est réglable dans l’application Gradio. Métrique principale de ranking : **ROC-AUC** et **AUPRC** (classes déséquilibrées ~33 %).

## 2. Variables

**Exclusions (fuite / inutiles)** : `statut_dossier` et `date_statut` (déterminent la cible) ; `jours_attente_confirmation` et `nb_modifications` (post-réservation) ; `logement_attribue` (attribution souvent postérieure, écart demande/attribut très lié au non-annulé) ; `id_reservation`, `id_entreprise` (~94 % NA).

**Conservées** : site, dates (via dérivées), nuits, composition, formule, pays (rares → AUTRE), origine, canal, fidélité, historique client, logement demandé, garantie, agence (binaire), profil, tarif, parking, demandes spéciales, code promo, indice d’affinité.

**Dérivées** : `delai_jours`, `nuits_totales`, `taille_groupe`, `mois_arrivee`, `ratio_weekend`, `via_agence`, `taux_annulation_passe`. Détail dans `audit_variables.md`.

## 3. Méthodes

Protocole commun : suppression des lignes identiques (hors id), **découpage chronologique** 80/20 sur `date_reservation`, même prétraitement.

| Palier | Idée | Observation attendue |
|---|---|---|
| Baseline | Toujours prédire le taux moyen / classe majoritaire | Référence AUPRC ≈ taux d’annulation |
| Régression logistique | Linéaire, avec / sans dérivées | Les dérivées (surtout le délai) améliorent le score |
| Arbre | Prof. 3 / choisie / illimitée | Prof. faible sous-apprend ; illimitée sur-apprend |
| Forêt aléatoire | Bagging d’arbres | Plus stable que l’arbre seul |
| HistGradientBoosting | Modèle final retenu | Meilleur compromis score / temps / robustesse |

Le modèle déployé est un **`HistGradientBoostingClassifier`** dans un pipeline unique (`features` → imputation / encodage → modèle), sauvegardé dans `model.pkl`.

## 4. Résultats et évaluation

Découpage **chronologique** pour coller à un usage en production (prédire le futur). Sur le test : **ROC-AUC ≈ 0,83**, **AUPRC ≈ 0,60**. Seuil métier **0,18** : matrice de confusion et taux d’appels dans le notebook et `model_meta.json`.

Diagnostic : comparer le score train vs test ; un écart trop large indiquerait du sur-apprentissage (limité ici par `min_samples_leaf`, `l2_regularization`, profondeur max).

Interprétation commerciale : **garantie non remboursable**, **canal intermédiaire**, **délai de réservation long**, **site urbain** et **historique d’annulations** poussent à appeler ; **client fidèle**, **demandes spéciales** et **parking** vont plutôt dans le sens opposé.

## 5. Application et tests

```text
python train.py
uvicorn api:api --reload   # http://127.0.0.1:8000/docs
python app.py              # Gradio
pytest
```

L’app expose 10 champs influents ; le reste prend les modes / médianes d’entraînement (`model_meta.json`). L’API `POST /predict` valide le schéma Pydantic (422 si invalide). Les tests vérifient une proba ∈ [0,1], un 200 sur entrée valide et un 422 sur dates incohérentes.

## 6. Limites et veille

- Dérive possible si la politique tarifaire / canaux changent.
- `garantie = Non remboursable` est quasi déterministe dans l’historique : à surveiller.
- Pays et codes promo à forte cardinalité : regroupement approximatif.
- Pas de calibration fine des probabilités ni d’optimisation bayésienne des hyperparamètres (bonus possible).
- Avec plus de temps : calibration (Platt / isotonic), suivi de drift, déploiement Spaces, modèle DL comparatif.

## 7. Usage d’assistants de code

Assistant Cursor / Composer utilisé pour accélérer le squelette (`features.py`, `train.py`, Gradio, FastAPI, tests, notebook). Le groupe a relu l’audit anti-fuite, le seuil économique, le protocole de split et a vérifié localement `train.py`, `pytest` et le démarrage de l’API / Gradio. Chaque membre doit pouvoir expliquer les choix (variables exclues, modèle final, seuil 0,18).
