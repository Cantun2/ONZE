# ONZE — Flotte d'agents : mode d'emploi

Douze agents spécialisés pour coder ONZE de bout en bout et le rendre utilisable. Chaque agent a un périmètre borné, un contrat d'entrée/sortie, une *definition of done* et des garde-fous. Le tout est pensé pour Claude Code (sous-agents), mais les rôles sont réutilisables dans n'importe quel orchestrateur.

## Installation
```
votre-repo/
├── ONZE_spec.md            # le cahier des charges (à placer à la racine)
├── CLAUDE.md               # mémoire projet, lue à chaque session
└── .claude/
    └── agents/             # les 12 sous-agents
```
Copiez ces fichiers dans le dépôt, lancez `claude` dedans, puis pilotez depuis la session principale (le "coach"). Les agents se déclenchent automatiquement selon leur `description`, ou explicitement : *« Use the elo-engineer subagent to implement src/ratings/elo.py ».*

## Le roster

| Agent | Rôle | Possède | Modèle | Écrit ? |
|---|---|---|---|---|
| **data-engineer** | Ingestion + harmonisation des noms (GATE) | `src/ingest/`, `data/` | sonnet | oui |
| **elo-engineer** | Ratings Elo walk-forward | `src/ratings/` | sonnet | oui |
| **model-engineer** | Poisson→λ + Dixon-Coles (cœur maths) | `src/model/lambdas,dixon_coles` | **opus** | oui |
| **knockout-engineer** | Prolongation + tab + P(qualif) | `src/model/knockout` | sonnet | oui |
| **prediction-engineer** | Pipeline match + marchés + bracket MC | `src/predict/` | sonnet | oui |
| **eval-engineer** | RPS/log-loss/calibration + backtest sans fuite | `src/eval/` | **opus** | oui |
| **api-engineer** | FastAPI | `src/api/` | sonnet | oui |
| **frontend-engineer** | React : heatmap, bracket, dashboard | `frontend/` | sonnet | oui |
| **test-writer** | Tests pytest + property-based | `tests/` | sonnet | tests seulement |
| **model-reviewer** | Audit intégrité + FUITE (lecture seule) | — | **opus** | **non** |
| **devops-packager** | Scaffold, config, Docker, Make, CI | plomberie | sonnet | oui (infra) |
| **docs-writer** | README, méthodo, docstrings | `docs/`, READMEs | sonnet | oui (docs) |

Opus est réservé aux trois rôles de jugement (cœur maths, évaluation, revue) ; Sonnet fait l'implémentation. C'est le bon compromis coût/qualité.

## Vagues de build (dépendances)

```
Wave 0 :  data-engineer  (GATE)   ‖   devops-packager (scaffold)
Wave 1 :  elo-engineer
Wave 2 :  model-engineer
Wave 3 :  knockout-engineer   ‖   prediction-engineer
Wave 4 :  eval-engineer       ‖   api-engineer
Wave 5 :  frontend-engineer
Continu:  test-writer (après chaque module)
          model-reviewer (avant tout "done" maths/data)
          docs-writer (quand ça se stabilise)
```
`‖` = parallélisable. Les agents d'une même vague sont indépendants et peuvent tourner en concurrence.

## Trois gates non négociables
1. **Harmonisation des noms** : rien ne démarre en Wave 1 tant que data-engineer ne reporte pas zéro nom non résolu.
2. **Revue d'intégrité** : aucun module maths/data n'est "done" tant que model-reviewer ne renvoie pas PASS. Une seule fuite = CHANGES REQUIRED.
3. **Invariants testés** : matrice qui somme à 1, probas dans [0,1], pas de fuite au backtest — verts avant de passer à la suite.

## Pourquoi cette découpe
- **Périmètres disjoints** : chaque agent possède un dossier, personne ne marche sur les fichiers d'un autre → pas de conflit d'écriture.
- **Least-privilege** : le reviewer n'a pas le droit d'écrire (il ne peut pas "réparer" en douce une fuite qu'il devrait signaler) ; le test-writer ne touche pas la source.
- **Le cœur mathématique isolé** sur Opus + double filet (reviewer + property tests), parce que c'est là que se logent les erreurs coûteuses et silencieuses (fuite, mauvaise vraisemblance, matrice non renormalisée).

## Astuce d'orchestration
Gardez un humain dans la boucle sur deux moments : (1) validation du rapport d'ingestion (les noms), (2) validation du verdict du reviewer avant de considérer le modèle figé. Le reste peut s'enchaîner en délégation.
