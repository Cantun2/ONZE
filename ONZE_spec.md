# ONZE — Moteur de distribution de scores (Coupe du Monde 2026)

**Cahier des charges technique — conception mathématique + architecture + plan de build**

Nom de code : **ONZE**. Objectif : produire, pour chaque match restant, une **distribution de probabilité complète sur les scores**, et en dériver tous les marchés (1N2, over/under, BTTS, qualification, vainqueur du tournoi). Extensible ensuite aux ligues de clubs.

---

## 0. Principe directeur

On ne prédit **jamais un score unique**. On prédit une matrice de probabilité $P(X=x, Y=y)$ où $X$ = buts domicile, $Y$ = buts extérieur. Tout le reste (résultat 1N2, score le plus probable, proba de qualification, bracket) se dérive de cette matrice. C'est le seul objet mathématiquement honnête, et c'est aussi le plus riche à afficher.

Chaîne de valeur :

```
Données historiques ─┐
                     ├─► Ratings Elo (par date) ─► Modèle de buts (Dixon-Coles)
Données live (cotes) ┘                                      │
                                                            ▼
                                    Matrice de scores P(x,y) par match
                                                            │
                          ┌─────────────────┬───────────────┼────────────────┐
                          ▼                 ▼               ▼                 ▼
                     Marchés 1N2       Score exact     Module KO         Simulation
                     O/U, BTTS         le + probable   (prol. + tab)     Monte-Carlo
                                                       P(qualif)         du bracket
                                                            │
                                                            ▼
                                              API (FastAPI) ─► Interface (React)
                                                            │
                                                            ▼
                                        Évaluation : RPS / log-loss / calibration
                                                vs baseline bookmaker
```

---

## 1. Périmètre & sorties

**Entrée** : la liste des matchs restants (8es à 48 → finale du 19 juillet), et l'historique complet.

**Sorties par match** :
- Matrice $P(x,y)$ pour $x,y \in \{0,\dots,K\}$ (on tronque à $K=10$, la masse au-delà est négligeable).
- Score le plus probable + top-5 scores.
- Probabilités $P(\text{dom})$, $P(\text{nul 90'})$, $P(\text{ext})$.
- Marchés dérivés : over/under 0.5…4.5, BTTS, écart de buts.
- En phase à élimination : $P(\text{équipe } i \text{ se qualifie})$ après prolongation + tirs au but.

**Sortie tournoi** : par simulation Monte-Carlo du bracket restant, $P(\text{atteint quart / demie / finale})$ et $P(\text{championne})$ pour chaque équipe.

---

## 2. Données

### 2.1 Sources

| Source | Contenu | Usage |
|---|---|---|
| `martj42/international_results` | 49 000+ matchs internationaux 1872→2024, scores | **Base d'entraînement** des ratings + du modèle de buts |
| `jfjelstul/worldcup` (Fjelstul) | CdM 1930→2022, événements détaillés | Enrichissement, features tournoi |
| `salimt/football-datasets` (Transfermarkt) | Valeurs marchandes, blessures, effectifs | Feature « qualité de squad » (proxy) |
| `soccerdata` (package) | xG, cotes historiques (Football-Data.co.uk) | **Baseline bookmaker** pour l'évaluation |
| The Odds API | Cotes live multi-books | Comparaison temps réel |

### 2.2 Schéma de données (cible, SQLite/Parquet)

```
teams(team_id, name_canonical, confederation, elo_current)
matches(match_id, date, home_id, away_id, home_goals, away_goals,
        neutral, tournament, importance_k)
ratings_history(team_id, date, elo)          # Elo tel qu'AVANT chaque match
fixtures(fixture_id, date, home_id, away_id, stage, neutral, host_flag)
predictions(fixture_id, model_version, matrix_json, p_home, p_draw,
            p_away, most_likely_score, created_at)
odds(fixture_id, bookmaker, o_home, o_draw, o_away, captured_at)
```

### 2.3 Harmonisation des noms (piège n°1)

Les datasets nomment les équipes différemment (« Korea Republic » / « South Korea », « USA » / « United States »). Prévoir une **table de correspondance canonique** dès le départ. Le registre `withqwerty/reep` mappe les identifiants entre providers — utile si tu croises Transfermarkt/FBref. Ne code rien d'autre tant que ça n'est pas propre : une équipe mal résolue pollue tout son historique Elo.

---

## 3. Fondements mathématiques

### 3.1 Ratings Elo adaptés au football international

On maintient un rating $R_i$ par équipe, mis à jour match par match dans l'ordre chronologique (formule *World Football Elo*).

Espérance de résultat pour l'équipe à domicile :

$$
W_e = \frac{1}{1 + 10^{-(R_{\text{dom}} + H - R_{\text{ext}})/400}}
$$

- $H$ = bonus terrain en points Elo (≈ 100 ; **$H=0$ sur terrain neutre**, mais garder $H>0$ pour les hôtes USA/Canada/Mexique en 2026).

Mise à jour après le match :

$$
R'_{\text{dom}} = R_{\text{dom}} + K \cdot G \cdot (W - W_e)
$$

- $W \in \{1, 0.5, 0\}$ = résultat réel (victoire / nul / défaite).
- $G$ = multiplicateur d'écart de buts :
$$
G = \begin{cases}
1 & |\Delta| \le 1\\
1.5 & |\Delta| = 2\\
\dfrac{11 + |\Delta|}{8} & |\Delta| \ge 3
\end{cases}
$$
- $K$ = poids selon l'importance (amical $\approx 10$, qualif $\approx 25$, phase finale CdM $\approx 60$). Stocké dans `matches.importance_k`.

L'équipe extérieure reçoit l'update opposé (somme des ratings conservée). On stocke le rating **avant** chaque match dans `ratings_history` : c'est indispensable pour éviter toute fuite d'information au backtest (§3.6).

> Pourquoi Elo et pas une estimation directe des forces d'attaque par équipe ? Les sélections jouent peu (≈10 matchs/an) : estimer un paramètre par équipe est instable. Elo agrège toute l'histoire dans un scalaire robuste, et se met à jour en ligne.

### 3.2 Du rating aux buts attendus $\lambda$

On relie la différence de rating aux buts attendus des deux camps via une **régression de Poisson à coefficients globaux** (seulement 3 paramètres, estimés sur des dizaines de milliers de matchs → très stable) :

$$
\log \lambda_{\text{dom}} = c_0 + c_1 \,(R_{\text{dom}} + H - R_{\text{ext}}) 
$$
$$
\log \lambda_{\text{ext}} = c_0 + c_1 \,(R_{\text{ext}} - R_{\text{dom}} - H)
$$

- $\lambda_{\text{dom}}, \lambda_{\text{ext}}$ = buts attendus de chaque équipe.
- $c_0$ fixe le niveau de scoring moyen ; $c_1 > 0$ la sensibilité au différentiel de force ; $H$ déjà dans Elo.

On estime $(c_0, c_1)$ par maximum de vraisemblance Poisson sur l'historique, en utilisant le rating **de chaque équipe à la date du match**.

*Variante avancée (optionnelle)* : modèle bilinéaire de Maher/Dixon-Coles avec un paramètre d'attaque $\alpha_i$ et de défense $\beta_i$ **par équipe** :
$$
\log \lambda_{\text{dom}} = \gamma + \alpha_i - \beta_j, \qquad \log \lambda_{\text{ext}} = \alpha_j - \beta_i
$$
avec contrainte d'identifiabilité $\sum_i \alpha_i = 0$. Plus expressif mais gourmand en données — à réserver à l'extension clubs, où chaque équipe a beaucoup de matchs.

### 3.3 Modèle de buts : Poisson → Dixon-Coles

**Étape naïve.** Si $X \sim \text{Poisson}(\lambda_{\text{dom}})$ et $Y \sim \text{Poisson}(\lambda_{\text{ext}})$ indépendants :
$$
P(X=x, Y=y) = e^{-\lambda_{\text{dom}}}\frac{\lambda_{\text{dom}}^x}{x!}\cdot e^{-\lambda_{\text{ext}}}\frac{\lambda_{\text{ext}}^y}{y!}
$$

**Problème** : le Poisson indépendant sous-estime les scores serrés (0-0, 1-1) et la corrélation entre les deux scores. **Correction de Dixon-Coles** : on multiplie par un facteur $\tau$ pour les 4 scores bas :

$$
\tau_{\lambda,\mu}(x,y) = \begin{cases}
1 - \lambda\mu\rho & (x,y)=(0,0)\\
1 + \lambda\rho & (x,y)=(0,1)\\
1 + \mu\rho & (x,y)=(1,0)\\
1 - \rho & (x,y)=(1,1)\\
1 & \text{sinon}
\end{cases}
$$

D'où la loi jointe :
$$
\boxed{P(X=x, Y=y) = \tau_{\lambda_{\text{dom}},\lambda_{\text{ext}}}(x,y)\cdot \text{Pois}(x;\lambda_{\text{dom}})\cdot \text{Pois}(y;\lambda_{\text{ext}})}
$$

$\rho$ est le paramètre de dépendance (contrainte de validité : $\tau \ge 0$, en pratique $\rho \in [-0.2, 0]$ environ).

**Pondération temporelle.** Les matchs anciens comptent moins. On pondère chaque observation par une décroissance exponentielle :
$$
\phi(\Delta t) = e^{-\xi \,\Delta t}
$$
où $\Delta t$ est l'ancienneté (en jours ou demi-vies). $\xi$ se règle par backtest (une demi-vie de ~2 ans est un bon point de départ pour l'international).

**Vraisemblance à maximiser** (log, en ignorant les constantes de factorielle) :
$$
\ell(c_0, c_1, \rho) = \sum_{m \in \text{matchs}} \phi(\Delta t_m)\Big[ \ln \tau(x_m, y_m) - \lambda_m + x_m \ln \lambda_m - \mu_m + y_m \ln \mu_m \Big]
$$

Optimisation : `scipy.optimize.minimize` sur $-\ell$ (L-BFGS-B), en injectant $\lambda_m, \mu_m$ issus de l'Elo à la date $m$ (§3.2). On estime donc conjointement $(c_0, c_1, \rho)$.

### 3.4 De la matrice aux marchés dérivés

Une fois $P(x,y)$ calculée pour un match :

- **Résultat 90'** :
$$
P(\text{dom}) = \sum_{x>y} P(x,y), \quad P(\text{nul}) = \sum_{x=y} P(x,y), \quad P(\text{ext}) = \sum_{x<y} P(x,y)
$$
- **Score le plus probable** : $\arg\max_{(x,y)} P(x,y)$.
- **Over/Under $n$** : $P(X+Y > n) = \sum_{x+y > n} P(x,y)$.
- **BTTS (les deux marquent)** : $1 - P(X=0) - P(Y=0) + P(0,0)$.

### 3.5 Phase à élimination : prolongation + tirs au but

En KO, pas de nul. On enchaîne trois briques conditionnelles.

**(a) Temps réglementaire.** Matrice $P_{90}(x,y)$ comme ci-dessus. Si $x \ne y$ → match résolu.

**(b) Prolongation (si nul à 90').** 30 min supplémentaires, jeu plus prudent et fatigue. Buts additionnels de chaque équipe :
$$
X_{ET} \sim \text{Poisson}\!\left(\lambda_{\text{dom}} \cdot \tfrac{30}{90}\cdot \kappa\right), \quad \kappa \approx 0.8 \text{ (facteur prudence)}
$$
On recalcule la résolution sur ces buts de prolongation (idéalement re-Dixon-Coles sur le mini-score de prolongation).

**(c) Tirs au but (si toujours nul).** Base historique ≈ 50/50, très faiblement corrélée à la force :
$$
P(i \text{ gagne aux tab}) = \sigma\!\big(\theta \,(R_i - R_j)\big), \quad \theta \text{ petit}
$$
$\sigma$ = sigmoïde. Par défaut $\theta \to 0$ (pièce équilibrée) ; on peut caler $\theta$ sur l'historique des séances de tab, mais ne pas surinterpréter.

**Probabilité de qualification** (loi des probabilités totales) :
$$
P(i \text{ qualifié}) = P_{90}(i) + P_{90}(\text{nul})\Big[P_{ET}(i) + P_{ET}(\text{nul})\,P_{\text{tab}}(i)\Big]
$$

### 3.6 Backtest sans fuite d'information

Règle absolue : pour prédire un match à la date $t$, n'utiliser **que** l'information disponible avant $t$ (ratings de `ratings_history` calculés jusqu'à $t^-$, coefficients estimés sur $<t$). Procédure **walk-forward** : on avance dans le temps, on prédit, on observe, on met à jour. Toute estimation calée sur des matchs postérieurs invalide le backtest.

### 3.7 Évaluation & calibration

- **RPS (Ranked Probability Score)** pour le 1N2 ordonné ($r=3$ issues) :
$$
\text{RPS} = \frac{1}{r-1}\sum_{i=1}^{r-1}\left(\sum_{j=1}^{i} (p_j - e_j)\right)^2
$$
où $p$ = probas prédites cumulées, $e$ = indicatrice cumulée du résultat réel. Plus bas = mieux.
- **Log-loss** et **Brier** pour les marchés binaires (BTTS, O/U).
- **Diagramme de fiabilité** : regrouper les prédictions par bac de probabilité, comparer proba prédite moyenne vs fréquence observée. Une bonne calibration suit la diagonale.
- **Baselines de comparaison** :
  1. Elo nu (proba 1N2 directe depuis $W_e$, sans buts).
  2. **Cotes de clôture du bookmaker**, « dé-viggées ». Passage cote → proba :
$$
p_i = \frac{1}{o_i}, \qquad q_i = \frac{p_i}{\sum_k p_k} \ \text{(normalisation simple)}
$$
     Raffinement : méthode de Shin ou *power method* pour retirer la marge de façon moins biaisée. Le RPS du book est la **barre à battre** ; s'en approcher est déjà un excellent résultat.

> Note d'honnêteté méthodologique : les ~30 matchs restants forment un échantillon minuscule. Le tournoi en direct sert de démo et de plaisir, **pas** de validation statistique. La validation se fait sur le backtest historique.

---

## 4. Architecture logicielle

### 4.1 Stack

- **Langage** : Python 3.11+.
- **Données** : `pandas`, `numpy`, `pyarrow` (Parquet), `requests` (odds live), `duckdb` ou `SQLite` (stockage).
- **Modèle** : `scipy.optimize` (MLE Dixon-Coles), `numpy` (matrices de score). `statsmodels` GLM Poisson possible pour la baseline §3.2.
- **API** : `FastAPI` + `uvicorn`, sérialisation JSON des matrices.
- **Interface** : `React` + `Vite`, charting `Recharts`/`visx` (heatmap de scores), `React-Query` pour le fetch. *MVP rapide alternatif* : `Streamlit` (une journée de dev, mais moins « produit »).
- **Tests** : `pytest`.

### 4.2 Arborescence du dépôt

```
onze/
├── config.yaml                  # K par tournoi, H, ξ, K_max scores, chemins
├── data/
│   ├── raw/                     # datasets téléchargés bruts
│   └── processed/               # parquet nettoyés + table canonique noms
├── src/
│   ├── ingest/
│   │   ├── loaders.py           # lecture martj42, Fjelstul, Transfermarkt
│   │   ├── canonical.py         # harmonisation des noms d'équipes
│   │   └── odds.py              # client The Odds API
│   ├── ratings/
│   │   └── elo.py               # calcul Elo walk-forward, ratings_history
│   ├── model/
│   │   ├── lambdas.py           # Elo diff -> (λ_dom, λ_ext), fit c0,c1
│   │   ├── dixon_coles.py       # τ, vraisemblance, MLE de ρ, matrice P(x,y)
│   │   └── knockout.py          # prolongation + tab + P(qualif)
│   ├── predict/
│   │   ├── fixture.py           # pipeline complet pour un match
│   │   ├── markets.py           # dérivés 1N2/OU/BTTS depuis la matrice
│   │   └── bracket.py           # simulation Monte-Carlo du tournoi
│   ├── eval/
│   │   ├── metrics.py           # RPS, log-loss, Brier
│   │   ├── calibration.py       # diagrammes de fiabilité
│   │   └── backtest.py          # walk-forward, comparaison baselines
│   └── api/
│       └── main.py              # endpoints FastAPI
├── frontend/                    # app React
├── notebooks/                   # exploration, tuning ξ et K
└── tests/
```

### 4.3 Signatures clés (pseudocode)

```python
# ratings/elo.py
def compute_elo_history(matches: pd.DataFrame,
                        H: float, k_by_tournament: dict,
                        base_rating: float = 1500) -> pd.DataFrame:
    """Retourne ratings_history: rating de chaque équipe AVANT chaque match."""

# model/lambdas.py
def fit_lambda_coeffs(matches, ratings_history, H) -> tuple[float, float]:
    """MLE Poisson: renvoie (c0, c1)."""
def predict_lambdas(elo_home, elo_away, H, c0, c1) -> tuple[float, float]:
    """Renvoie (λ_dom, λ_ext)."""

# model/dixon_coles.py
def tau(x, y, lam, mu, rho) -> float: ...
def fit_rho(matches, ratings_history, coeffs, xi) -> float:
    """MLE pondéré temporellement du paramètre de dépendance ρ."""
def score_matrix(lam, mu, rho, K=10) -> np.ndarray:
    """Matrice (K+1)x(K+1) de P(x,y). Normalisée."""

# predict/markets.py
def derive_markets(P: np.ndarray) -> dict:
    """{p_home, p_draw, p_away, most_likely, over_under{...}, btts}."""

# model/knockout.py
def advance_prob(lam, mu, rho, elo_home, elo_away, theta) -> dict:
    """{p_home_advance, p_away_advance} via 90'+prolongation+tab."""

# predict/bracket.py
def simulate_tournament(remaining_fixtures, model, n_sims=100_000) -> pd.DataFrame:
    """Monte-Carlo: P(atteint chaque tour) et P(championne) par équipe."""
```

### 4.4 Endpoints API

```
GET  /fixtures                      -> matchs restants + méta
GET  /predict/{fixture_id}          -> matrice P(x,y) + marchés dérivés
GET  /bracket                       -> probas d'avancement par équipe
GET  /eval/backtest?from=YYYY       -> RPS modèle vs Elo vs bookmaker
POST /refresh                       -> recalcul (nouveaux résultats/cotes)
```

---

## 5. Interface (le « produit »)

Quatre vues :

1. **Tableau des matchs restants** — une ligne par match : score le plus probable, barres 1N2, % de qualification, indicateur de confiance. Tri par date/stade.
2. **Détail d'un match** — la pièce maîtresse : **heatmap de la matrice $P(x,y)$** (buts dom en abscisse, ext en ordonnée, intensité = probabilité), + panneau des marchés dérivés (O/U, BTTS), + top-5 scores. Si cotes dispo : superposer la proba implicite du book pour visualiser l'écart.
3. **Bracket** — l'arbre KO restant avec, à chaque équipe, sa $P(\text{championne})$ issue de la simulation Monte-Carlo. Se met à jour à chaque résultat.
4. **Dashboard calibration** — courbe de fiabilité, RPS courant du modèle vs bookmaker, historique des prédictions vs réel. C'est la vue qui te permet, à toi, de tirer tes conclusions.

---

## 6. Plan de build par phases

| Phase | Livrable | Contenu |
|---|---|---|
| **0. Socle données** | Parquet propres + table noms canoniques | Ingest martj42 + Fjelstul, harmonisation, schéma SQLite |
| **1. Elo** | `ratings_history` + baseline 1N2 | Elo walk-forward, première éval RPS Elo nu |
| **2. Modèle de buts** | Matrices $P(x,y)$ | Fit $(c_0,c_1)$, MLE Dixon-Coles $\rho$, décroissance $\xi$ |
| **3. Marchés + KO** | Dérivés + $P(\text{qualif})$ | 1N2/OU/BTTS, module prolongation+tab |
| **4. Backtest** | Rapport RPS vs book | Walk-forward multi-décennies, calibration, tuning $\xi$ |
| **5. API + interface** | Produit utilisable | FastAPI + React, 4 vues, simulation bracket |
| **6. Extension clubs** | Généralisation | Modèle bilinéaire attaque/défense par équipe, ligues via `soccerdata` |

Ordre pensé pour avoir un résultat mesurable dès la phase 1 (Elo donne déjà des probas 1N2), puis raffiner. Chaque phase est backtestable indépendamment.

---

## 7. Pièges à anticiper

1. **Noms d'équipes** (déjà dit, mais c'est le tueur silencieux) : résoudre avant tout le reste.
2. **Fuite d'information** au backtest : ratings et coefficients strictement « as-of ».
3. **Terrain neutre vs hôtes** : CdM 2026 = terrains neutres, sauf avantage réel pour USA/Canada/Mexique. Gérer `neutral` et `host_flag`.
4. **Réglage de $\xi$** (décroissance temporelle) : trop fort → oublie tout ; trop faible → traîne des données obsolètes. À optimiser par RPS de backtest, pas au pif.
5. **Troncature $K$** : $K=10$ suffit ; renormaliser la matrice après troncature et correction $\tau$.
6. **Petits échantillons internationaux** : ne pas surinterpréter un modèle par équipe ; c'est pourquoi Elo + coefficients globaux est le bon compromis ici.
7. **Contrainte de validité de $\rho$** : borner l'optimisation pour garder $\tau \ge 0$ sur tous les scores bas.

---

*Fin du cahier des charges. Le cœur mathématique est aux §3.3 (Dixon-Coles) et §3.5 (KO) ; le cœur produit aux §5–6.*
