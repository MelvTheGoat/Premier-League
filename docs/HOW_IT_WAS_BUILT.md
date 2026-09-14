# How this was built

A record of the design decisions behind the Premier League predictor: what
each requirement demanded, how it was met, where it lives in the code, and
what the model is measurably worth.

- **15 commits**, 56 tracked files, ~5,600 lines of Python, 55 tests
- **213 features** per match, built in one chronological pass
- **Backtested** over 1,050 matches: 52.1% outcome accuracy, 0.9948 log loss

---

## 1. The central constraint

> *"Do not build this as a hardcoded checklist that gets manually filled in
> per match."*

This was the requirement that shaped everything else, so it is worth being
precise about what it rules out and what it demands.

**Ruled out:** any code path of the form *if the manager is new, shift the
home win probability by X*. No such path exists anywhere in the repository.
There is no lookup table of adjustments, no per-match override file, and no
hand-tuned constant applied to a prediction.

**Demanded:** every contextual signal becomes a **number in a feature
table**, sitting alongside goals and points, and the model decides what it
is worth. If congestion matters more for a squad missing key players, the
gradient-boosted model finds that interaction from the data. If a signal
turns out to be worthless, it gets a low split gain and is effectively
ignored — without anyone having to decide that in advance.

The consequence worth stating plainly: **the model is allowed to disagree
with the premise.** A feature included because it seemed important can end
up contributing nothing, and the feature-importance ranking (§7) reports
that honestly rather than hiding it.

---

## 2. Each signal, and how it was turned into a feature

Where a signal resisted direct measurement, the answer was a better proxy —
never a rule.

| Signal | How it became features | Count |
| --- | --- | --- |
| **New manager bounce** | Not a lookup, so it became three things that are measurable: days since the appointment, matches played under the current manager, and points per game under the current manager *minus* the previous one. The model decides whether a bounce exists and how long it lasts. | 12 |
| **League position & incentive** | Position alone is weak — what changes behaviour is distance to what a club is chasing or fearing. So: points off the top, off the top four, off the top six, and points *above* the relegation zone, plus position, points and points per game. In May these mean something entirely different than in August, which is why season progress is also a feature. | 24 |
| **Missing key players** | `players_out` and `key_players_out` per club per gameweek. A club with no row is treated as **unknown**, not as fully fit — so an absent file removes the signal rather than biasing it toward full strength. **Currently 0**: the file ships empty, so the six columns carry no values and are not admitted to the feature set at all. They appear automatically once the file is populated. | 0 (of 6) |
| **Squad rotation risk** | Rotation cannot be observed before a team sheet, so its *causes* are features instead: days of rest, matches in the last 14 and 21 days, days until the next fixture, whether the kick-off is midweek, and which European competition the club is in (tiered: Champions > Europa > Conference > none). | 18 |
| **Recent form** | Rolling windows of 4, 6 and 10 matches — **across all competitions**, because a club's last six games are its last six games whatever badge was on the fixture. Each window carries points per game, goals for and against, goal difference, win rate, clean-sheet rate, shots, shots on target, and *the average Elo of the opposition faced* — so a good run against weak opponents is distinguishable from a good run against strong ones. | 108 |
| **Historical squad quality** | FIFA / EA FC ratings aggregated to one row per club per season: overall (a 70/30 blend of best XI and squad depth), attack, midfield and defence unit ratings. Standardised **within season**, because ratings inflate over time and only a club's standing relative to its league is comparable across years. | 12 |
| **Head-to-head & time since last meeting** | Points per game and goal difference over the last six meetings, the number of meetings on record, and days since the last one. | 4 |
| **Home/away splits** | Separate home record for the home side and away record for the away side, plus previous-season points per game and goal difference. | 9 |
| **Season shape** | Gameweek number, fraction of season elapsed, gameweeks remaining, weekend flag. | 4 |
| **Strength & history** | Cross-division Elo, Elo-implied win expectation, previous-season points per game and goal difference, whether that season was in the top flight, seasons tracked, top-flight seasons, promoted flag. | 22 |

Total: **213**. The availability columns are the one family currently
contributing nothing, and that is the design working rather than failing —
see §10.

Every family above is computed for the home side, the away side, **and as the
difference between them** — giving the model subtraction for free rather than
making it learn arithmetic from scratch.

### The promoted-club problem

Three clubs a season arrive with no Premier League form, no league-table
position and no meaningful head-to-head record. Left alone they are blanks
for two months.

The fix is that **Elo is rated across divisions on a single scale** — Premier
League, Championship, League One and both domestic cups — with ratings
carried between seasons (regressed part-way toward the mean, because squads
turn over) and a starting offset by division. A club coming up has a full
season of results against opposition whose strength is already known, and
that information now survives promotion instead of being discarded.

The same problem bit the scoreline model separately and needed its own fix
(§6).

---

## 3. Data sources

| Source | Supplies | Access | Coverage |
| --- | --- | --- | --- |
| **openfootball/england** | Results, fixtures, gameweek numbers, kick-off dates — PL, Championship, League One, FA Cup, EFL Cup | Public git repo, no key | 2000-01 onward, updated within a day or two of each round |
| **datasets/football-datasets** | Shots, shots on target, corners, fouls, cards, referee (a football-data.co.uk mirror) | Public git repo, no key | 1993-94 onward, **lags the live season** |
| **FIFA / EA FC ratings** | Squad-quality proxy | Not vendored; built from whichever dumps you supply | Depends on dumps |
| `data/manual/managers.csv` | Manager tenure and changes | Hand-maintained | 2025-26 onward |
| `data/manual/unavailability.csv` | Injuries, suspensions, international duty | Hand-maintained | Empty by default |
| `data/manual/european_participation.csv` | Midweek European commitments | Hand-maintained | 2026-27 |
| Derived from results | League table, form, congestion, head-to-head, Elo | — | Complete |

Everything needed to predict the current gameweek comes from openfootball,
which is free and current. The rest improve the model where available.

**No API keys are required anywhere.** That was a deliberate choice: an
API-Football or Understat key would have added expected goals and richer
team news, but also a credential to manage, a rate limit to respect, and a
paid dependency that stops the pipeline dead when it lapses.

### Match statistics as an xG substitute

Understat was investigated for expected goals and rejected — no free bulk
access without scraping. Shots on target is the closest freely available
stand-in, and rolling shot-creation and shot-concession rates read
underlying performance better than goals do in small samples. Two of the
top six features the model actually uses are shot-based (§7), so the
substitution earned its place.

### The lagging-source problem

Match statistics and video-game ratings are both published *later* than
results. This creates a specific trap: **a feature present for every
training row but missing for the gameweek being predicted is worse than no
feature at all**, because the model learns to depend on evidence it will not
have at serving time.

Two different fixes, because the two sources fail differently:

- **Match statistics** — any column populated for fewer than **half** the
  fixtures being predicted is dropped from that run's model entirely. Not
  "at least one value": a single surviving value cannot justify a feature
  the model may have leaned on across thousands of training matches. The
  column returns automatically on the next retrain once the source catches
  up, with no code change. Each run records what it dropped, visible on the
  `/model` page.

- **Squad ratings** — EA rate squads once a season and the current season's
  edition is not available in usable form until well into it, so the
  freshest rating for a live gameweek is typically a season old. Rather than
  discard the signal, the most recent rating is **carried forward and its
  age in seasons is supplied as its own feature**, letting the model
  discount a stale rating on its own terms instead of being handed a silent
  lie.

---

## 4. Architecture

Three layers, deliberately separable, so any one can be replaced without
touching the others.

```
plpredict/
  config.py                  paths, seasons, hyper-parameters
  db.py                      SQLite schema and helpers

  data/                      ── DATA PIPELINE: fetch, clean, store
    sources/openfootball.py    results and fixtures (primary)
    sources/footballdata.py    match statistics (optional enrichment)
    sources/fifa_ratings.py    squad-quality aggregation
    teams.py                   club-name normalisation
    ingest.py                  everything above → database

  features/                  ── FEATURE ENGINEERING
    elo.py                     cross-competition Elo
    state.py                   rolling per-club state, forward-only
    build.py                   the chronological pass, versioned output

  models/                    ── MODELLING
    outcome.py                 LightGBM + multinomial blend
    scoreline.py               Dixon-Coles bivariate Poisson
    evaluate.py                walk-forward backtesting

  pipeline/run.py            ── the rolling retrain loop
  web/                       ── Flask app, templates, CSS
```

The ingest layer knows about sources and the database and nothing else. The
feature layer knows about the database and nothing about models. The web
layer loads no model and computes nothing on a request.

### Club-name normalisation

An unglamorous piece that everything depends on. The sources spell the same
club several ways — `Manchester United FC`, `Manchester United`, `Man
United`; `AFC Bournemouth` and `Bournemouth`; `Nott'm Forest`. Left alone, a
club's history silently splits in two and its form features reset
mid-archive.

Names fold to a canonical form through rules plus an override table
(`data/manual/team_aliases.csv`). Verified by reducing 72 raw spellings
across 27 seasons to exactly the 46 clubs that have played Premier League
football since 2000-01, with zero unmatched names across both sources.

### Parsing

The openfootball archive has drifted between two layouts over the years and
needed both handled, plus year rollover at New Year, unplayed fixtures
(which is how the current gameweek arrives), and knockout notation where a
shoot-out score precedes the regulation one — `10-9 pen. (1-1, 0-1)` is a
1-1 draw, not a 10-9 win.

Validated by parsing every season 2000-01 to 2026-27 and asserting exactly
380 matches each, with a matchday and a date on every row: 36,597 matches
across all competitions, zero malformed.

---

## 5. The rolling retrain, and the guarantee underneath it

The whole loop is one rule:

> Predictions for gameweek N are made by a model trained only on matches
> that kicked off **before gameweek N's first fixture**.

That holds in training and in serving, which is what makes the recorded
history meaningful rather than decorative.

1. `ingest` pulls the latest results into `matches`.
2. The feature table is rebuilt in **one chronological pass**. State only
   moves forward, so a feature *cannot* see a result that has not been fed
   in yet — the guarantee is structural, not a matter of remembering to
   filter correctly.
3. Features for every match in a gameweek are computed from the state before
   that gameweek's **first** kick-off — never from Saturday's results when
   the fixture is on Monday. Training and serving therefore see identical
   information.
4. Both models are fitted on everything before the cutoff.
5. The gameweek is predicted and stored with a new `model_runs` row.

**Nothing is overwritten.** Each run records what it trained on and writes
its own batch of predictions, so a forecast made before kick-off stays on
the record exactly as made. Re-running a gameweek adds a run rather than
editing one. The history page shows what was *actually* forecast, not what
today's model would say with hindsight.

Point 3 is the one most easily got wrong, and it is the most heavily tested:
the suite asserts that a gameweek is never trained on itself, that the table
context never reflects more gameweeks than have been played, and that every
match within a gameweek is featurised from an identical table. If a feature
could see a result from its own gameweek, every accuracy figure reported
below would be inflated and nothing else would be worth checking.

---

## 6. The two models

Two related tasks, two different tools, because they answer different
questions.

### Outcome — home win / draw / away win

A blend of a **LightGBM gradient-boosted classifier** over the full feature
table and a **multinomial logistic regression** over a compact core of
always-populated strength and form features.

The two fail in different places. The booster finds the interactions that
make the contextual features worth collecting at all, but it needs data to
do that and is over-confident when a season is young. The linear model
cannot invent an interaction it has no evidence for, which makes it markedly
better calibrated early. The blend is what gets served.

**Boosting rounds are chosen by early stopping, not fixed.** This was the
single largest model improvement in the build. The first version used a
fixed 400 rounds and produced log loss of 1.023 — barely better than a tuned
Elo baseline, with visible over-confidence in the 0.5–0.8 band. A config
sweep showed the optimum was nearer 50–170, and that 400 rounds was costing
about **0.03 nats**. Each retrain now runs a probe on a chronological tail
of the training set, stops when validation log loss stops improving, and
refits on everything with the count it settled on. It re-tunes itself as the
season's data accumulates.

### Scoreline — Dixon-Coles bivariate Poisson

Goals are low-count events, so the object to model is each side's goal
*rate*, not a label. Every club gets an attack and a defence strength; the
home side's expected goals is its attack times the opponent's defence times
a shared home-advantage term. A score matrix follows, and from it the
expected score and the most likely scoreline.

Three refinements:

- **Low-score correction** (`rho`, fitted at −0.051): independent Poisson
  under-predicts 0-0 and 1-1 and over-predicts 1-0 and 0-1.
- **Time decay**: older matches carry exponentially less weight, so ratings
  follow a squad as it changes rather than averaging over an era.
- **Championship results fitted alongside**, with their own scoring-level
  offset. Without this a promoted club is rated on three or four Premier
  League matches — one with a single goal in three games was being modelled
  at 0.51 expected goals, effectively incapable of scoring. The clubs going
  up and down each season tie the two scales together; the same fixture
  after the fix rated 1.21.

### Making the two agree

The outcome model and the scoreline model can disagree: a home win's
probability is spread across 1-0, 2-0 and 2-1, so **1-1 is often the single
most likely score even when a home win is comfortably the most likely
result**. Displaying "Home win — 1-1" looks broken.

The displayed scoreline is therefore the most likely score *consistent with
the predicted outcome* — the conditional mode. Unconditional expected goals
are shown alongside and are unaffected.

---

## 7. Performance

Measured by `scripts/backtest.py`, which replays the platform's own
procedure gameweek by gameweek — predict one knowing only what came before
it, then move on. A random train/test split would flatter the model badly by
letting it learn from matches that had not happened yet.

**2023-24, 2024-25 and 2025-26, from gameweek 4 onwards — 1,050 matches:**

| Metric | Model | Baseline |
| --- | --- | --- |
| Outcome accuracy | **52.1%** | 43.2% (always predict a home win) |
| Log loss | **0.9948** | 1.0061 (Elo-only) |
| Brier score | 0.5932 | — |
| Exact scoreline | 8.2% | — |
| Goals mean absolute error | 0.93 per side | — |

Per season: **57.1% / 52.0% / 47.1%**. The spread is mostly the seasons
rather than the model — 2025-26 had eleven managerial changes and an
unusually compressed table.

### Calibration

More important than accuracy here, because the platform publishes
probabilities rather than just picks.

| Predicted | Observed | n |
| --- | --- | --- |
| 0.16 | 0.18 | 552 |
| 0.25 | 0.25 | 1,169 |
| 0.35 | 0.34 | 438 |
| 0.45 | 0.44 | 383 |
| 0.55 | 0.52 | 273 |
| 0.65 | 0.65 | 170 |
| 0.74 | 0.70 | 100 |

When the model says 65%, it happens about 65% of the time. This is what the
linear half of the ensemble and the early-stopped rounds are for; the fixed
400-round version showed predicted 0.65 landing at 0.57.

### Draws are essentially never the pick

| Actual ↓ / Predicted → | H | D | A |
| --- | --- | --- | --- |
| **H** | 367 | 1 | 86 |
| **D** | 176 | 0 | 85 |
| **A** | 155 | 0 | 180 |

A draw is almost never the single most likely outcome of a football match,
so the argmax avoids them. The model instead expresses draws as probability
mass — typically 25–30%. This is a property of the task, not a bug: a model
tuned to pick more draws scores better on draws and worse on everything
else. It is why the site shows the full three-way split rather than only the
pick, and why log loss is the number to watch.

### What the model actually leans on

The 15 features ranked highest by split gain in the most recent run:

```
 1. diff_squad_overall_z           9. away_previous_season_gd_per_game
 2. diff_elo                      10. diff_form6_opponent_elo
 3. home_venue_goals_for          11. home_squad_overall_z
 4. elo_expected_home             12. away_previous_season_ppg
 5. diff_form10_shots_against     13. h2h_days_since
 6. diff_form6_shots_against      14. diff_form6_shots_for
 7. diff_squad_defence            15. diff_venue_goals_against
```

This is the evidence that §1 and §2 were worth the effort. The single
highest-ranked feature is the **FIFA/FC squad-quality proxy**, not a
results-derived one. Shot-based form occupies four of the top fifteen.
Opposition-strength-adjusted form (`diff_form6_opponent_elo`) ranks above
raw form. The contextual signals are doing real work, not decorating a
glorified Elo model.

Read honestly: roughly half of matches called correctly, nine points clear
of always backing the home side, and a log loss meaningfully below a tuned
Elo baseline is a reasonable place for a model of this kind to sit. It is
not a betting edge.

---

## 8. Build order

Deliberately sequenced so each layer could be validated before the next
depended on it.

1. **Data access first.** Before writing any modelling code, establish what
   is actually obtainable. Several candidate sources were unreachable in the
   build environment; openfootball and the football-data mirror were
   confirmed working and everything else was designed around that.
2. **Parser, validated hard.** Every season asserted at exactly 380 matches
   before anything was built on top.
3. **Club-name normalisation**, verified to 46 canonical clubs with zero
   unmatched names.
4. **Schema and ingest** — append-only predictions from the start, because
   retrofitting that later would have meant losing the forecast history.
5. **Feature layer**, with the point-in-time guarantee made structural
   rather than procedural.
6. **Outcome model**, then measured against baselines — which is what
   surfaced the 400-round overfitting.
7. **Scoreline model**, then measured — which surfaced the promoted-club
   cold start.
8. **Pipeline, backtest, frontend, tests, deployment, automation.**

Three bugs worth recording, all found by testing rather than inspection:

- **`--backfill` skipped the upcoming gameweek** once it had a prediction,
  so a weekly cron would have silently stopped producing forecasts after its
  first run. Now only completed gameweeks are skipped.
- **The serverless bundle shipped no templates or CSS.** Flask loads both
  from disk at request time rather than importing them, so dependency
  tracing never bundled them; every page would have returned 500. Caught by
  assembling the bundle the build actually produces and serving it in
  isolation.
- **Class-widening crash** when a training window contained no draws — the
  blend broke on mismatched array shapes. Surfaced by a synthetic test
  fixture, fixed by mapping both models' outputs onto the full three-class
  space.

---

## 9. Deployment and automation

The web layer imports no part of the data or modelling stack, which makes it
deployable as a small serverless function.

- **A slim serving database is committed.** `scripts/export_web_db.py`
  copies out only what the pages query — 1.4 MB against 61 MB for the working
  database, 53 MB of which is the feature table no page touches. Exported
  without a write-ahead log so it can be opened read-only, which a read-only
  filesystem requires.
- **Root `requirements.txt` is Flask alone.** The pipeline's dependencies
  live in `requirements-pipeline.txt`. A deployment installing pandas,
  LightGBM and SciPy to read a SQLite file would not fit in a serverless
  function.
- **A scheduled workflow runs daily**, not weekly, because fixtures do not
  keep a weekly rhythm — a gameweek can span four days or sit a fortnight
  away behind an international break. Each run compares how many matches
  have been played and which gameweek is next against what the site is
  serving, and stops there if nothing changed. Without that check the job
  would retrain and commit a 1.4 MB binary daily regardless, adding hundreds
  of megabytes of history over a season to say the same thing.
- **The workflow smoke-tests before committing** — it serves the freshly
  exported database and checks the pages render. Publishing one the site
  cannot read would take the site down until someone noticed.

---

## 10. Known limitations

Stated plainly, because a model's weak points are more useful than its
headline number.

- **Injuries are not automated, and currently contribute nothing.**
  `unavailability.csv` ships empty, so the six availability columns carry no
  values and are excluded from the feature set entirely — the model is not
  guessing about fitness, it simply has no fitness signal. This is the design
  behaving correctly (an empty file removes the feature rather than biasing
  it toward full strength), but it does mean one of the requested contextual
  signals is wired up and inert. It is the largest remaining gap and the
  single most valuable thing to add; team news is genuinely hard to scrape
  reliably, but even rough per-club counts per gameweek would activate it.
- **Manager history starts at 2025-26.** The features are wired in and
  correct, but the effect is estimated from roughly two seasons of matches.
  Extending the file backwards is cheap and helps directly. Some appointment
  dates are month-accurate and marked `manual:approx`.
- **No true expected-goals data.** Shots on target is the stand-in. A real
  xG feed would slot straight into the `match_stats` table.
- **Match statistics lag the live season**, so the live model is thinner
  than the backtested one — handled by dropping affected columns per run,
  but worth knowing when comparing live accuracy to §7.
- **Squad ratings are a season stale by construction.** The rating age is
  supplied as a feature so the model can account for it.
- **Fixtures come from one source.** A postponement is picked up on the next
  sync, usually same-day but not instant.
- **The log-loss margin over Elo is modest** (0.9948 vs 1.0061). The
  contextual features are earning their place, but the coverage gaps above —
  empty injuries, two seasons of manager history, no live shot data — are
  the most likely reason it is not larger.
