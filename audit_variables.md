# Audit des variables — Résidences Azur

Une ligne par variable. La prédiction se fait **à l'enregistrement** de la réservation.

| Variable | Disponible à la réservation ? | Utile ? | Décision | Justification |
|---|---|---|---|---|
| `id_reservation` | Oui | Non | **Exclure** | Identifiant technique, pas prédictif |
| `site` | Oui | Oui | **Garder** | Urbaine ~37 % d'annulation vs littoral ~24 % |
| `annule` | Sans objet | Cible | Cible | 1 = séjour annulé |
| `date_reservation` | Oui | Oui (dérivée) | **Garder** (via `delai_jours`, etc.) | Date brute → délai avant arrivée |
| `date_arrivee` | Oui | Oui (dérivée) | **Garder** (via `delai_jours`, `mois_arrivee`) | Connue à la réservation |
| `nuits_week_end` | Oui | Oui | **Garder** | Durée / structure du séjour |
| `nuits_semaine` | Oui | Oui | **Garder** | Idem ; `nuits_totales` dérivée |
| `nb_adultes` | Oui | Oui | **Garder** | Composition du séjour |
| `nb_enfants` | Oui | Oui | **Garder** | Quelques NA → imputation médiane |
| `nb_bebes` | Oui | Oui | **Garder** | Composition |
| `formule` | Oui | Oui | **Garder** | Pension complète ~55 % d'annulation |
| `pays_client` | Oui | Oui | **Garder** (pays rares → AUTRE) | 177 modalités → regroupement |
| `origine_commerciale` | Oui | Oui | **Garder** | Canal d'acquisition |
| `canal_vente` | Oui | Oui | **Garder** | Intermédiaire plus risqué que Direct |
| `client_fidele` | Oui | Oui | **Garder** | Fidèles ~12 % vs ~34 % |
| `annulations_passees` | Oui | Oui | **Garder** | Historique client |
| `sejours_honores_passes` | Oui | Oui | **Garder** | Historique ; alimente `taux_annulation_passe` |
| `logement_demande` | Oui | Oui | **Garder** | Choix connu à la réservation |
| `logement_attribue` | **Non / douteux** | Fuite | **Exclure** | Attribué souvent après ; écart demande/attribut ≠ 0 très lié à « non annulé » |
| `nb_modifications` | **Non** | Fuite | **Exclure** | Compteur post-réservation (souvent > 0 alors qu'à t0 = 0) |
| `garantie` | Oui | Oui | **Garder** | Non remboursable ≈ 99 % d'annulation |
| `id_agence` | Oui (si connu) | Partiel | **Garder** comme `via_agence` | Haute cardinalité / NA → indicateur binaire |
| `id_entreprise` | Oui (si connu) | Faible | **Exclure** | ~94 % manquant, cardinalité élevée |
| `jours_attente_confirmation` | **Non** | Fuite | **Exclure** | Connu après le délai de confirmation |
| `profil_client` | Oui | Oui | **Garder** | Individuel / contrat / groupe |
| `tarif_nuit` | Oui | Oui | **Garder** | Signal commercial |
| `nb_places_parking` | Oui | Oui | **Garder** | Corrélé négativement à l'annulation |
| `nb_demandes_speciales` | Oui | Oui | **Garder** | Engagement du client |
| `code_promo` | Oui | Oui | **Garder** | Modalités regroupées si besoin |
| `indice_affinite` | Oui (marketing) | Faible | **Garder** | Corrélation faible mais disponible |
| `statut_dossier` | **Non** | Fuite totale | **Exclure** | Détermine parfaitement `annule` |
| `date_statut` | **Non** | Fuite | **Exclure** | Date du statut final |

## Variables dérivées créées (≥ 3)

| Variable | Formule | Intérêt |
|---|---|---|
| `delai_jours` | arrivée − réservation | Fort signal (corr ≈ 0,29) |
| `nuits_totales` | week-end + semaine | Durée du séjour |
| `taille_groupe` | adultes + enfants + bébés | Occupation |
| `mois_arrivee` | mois de `date_arrivee` | Saisonnalité |
| `ratio_weekend` | nuits WE / nuits totales | Structure du séjour |
| `via_agence` | `id_agence` non nul | Canal B2B |
| `taux_annulation_passe` | annulations / (annulations + honorés + 1) | Risque client |
