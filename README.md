# Premier League Predictor

Predicts Premier League match outcomes and scorelines for the current
gameweek, keeps a public record of what it predicted for every gameweek
already played, and **retrains itself after each gameweek** so the model
that predicts gameweek N+1 has learned from gameweek N.

```
scripts/run_pipeline.py   →  fetch results → rebuild features → retrain → predict next gameweek → store
scripts/serve.py          →  a small web app that reads what was stored
```

---

For the design decisions behind all of this — how each contextual signal
was turned into a feature, what the model measures out at, and where it is
weak — see [`docs/HOW_IT_WAS_BUILT.md`](docs/HOW_IT_WAS_BUILT.md).

## The idea

A scoreline and a league table describe what happened. They do not
describe the situation the match was played in, and the situation often
decides it. A club four days after a European tie fields a different
side. A club that changed manager last month is a different team from
the one that finished last season. A club with nothing left to play for
in May is not the club that was chasing a European place in February.

The temptation is to encode that as a checklist — *if new manager, add
5%* — filled in by hand for each fixture. This project deliberately does
not do that. **Every contextual signal is turned into a number, put in
the feature table alongside goals and points, and left to the model to
weigh.** Nothing anywhere applies a hand-tuned adjustment to a
prediction.

Where a signal resists measurement, the answer is a better proxy, not a
hard-coded rule. "New manager bounce" is not a lookup, so it becomes
three things that are: days since the appointment, matches played under
the new manager, and the difference in points per game either side of
the change. If those matter, the model will find them; if they do not,
it will ignore them. That is the point.

---

## What is in the feature table

204 columns, built for every match in one strictly chronological pass.
Each family below is computed for the home side, the away side, and as
the difference between them.

| Family | Columns | Why it is there |
| --- | --- | --- |
| **Strength** | Cross-competition Elo, Elo-implied win expectation | The only strength measure that survives promotion — see below |
| **Rolling form** | Points, goals for/against, goal difference, win and clean-sheet rate, shots and shots on target, and the average Elo of the opposition faced, over the last 4, 6 and 10 matches | A club's last six games predict the next one better than its season average |
| **Table context** | Position, points, points per game, matches played, and the points gap to first, to the top four, to the top six, and above the relegation zone | Position alone is weak; distance to what a club is chasing or fearing is the thing that changes how it plays |
| **Congestion** | Days of rest, matches in the last 14 and 21 days, days until the next fixture, midweek kick-off flag, European competition tier | Rotation risk, without needing a team sheet |
| **Manager** | Days and matches under the current manager, a new-manager flag, and points per game under the current manager minus the previous one | The best available proxies for a managerial change |
| **Availability** | First-team players out, key players out | Missing regulars change a side's strength in a way results-based features cannot see |
| **Squad quality** | FIFA / EA FC squad rating standardised within season, attack and defence unit ratings, and the age of the rating in seasons | Quality on paper, which the table has not caught up with for a newly promoted or newly rebuilt side |
| **Head to head** | Points per game and goal difference over the last six meetings, and days since the last one | Some fixtures do not follow form |
| **Venue** | Home record for the home side, away record for the away side, plus previous-season points per game and goal difference | Home and away are close to different competitions for some clubs |
| **Season shape** | Gameweek number, fraction of the season elapsed, gameweeks remaining, weekend flag, promoted flag, seasons in the division | Late-season incentives are not early-season incentives |

### Why Elo, and why it spans divisions

Elo ratings are computed over the Premier League, the Championship,
League One and the domestic cups on a single scale, with a rating
carried between seasons (regressed part-way toward the mean, because
squads turn over) and a division offset for a club's first appearance.

This exists to solve the promoted-club problem. A club coming up has no
Premier League form, no Premier League table position and no useful
head-to-head record. It does have a season of results against
opposition whose strength is already known, and rating everything on one
scale is what lets that carry across. Without it, three clubs a season
start as blanks.

---

## The two models

They answer different questions and use different tools.

### Outcome — home win / draw / away win

A blend of a **LightGBM gradient-boosted classifier** over the full
feature table and a **multinomial logistic regression** over a compact
core of always-populated strength and form features.

The two fail in different places. The booster finds the interactions
that make the contextual features worth collecting — how congestion
interacts with missing players, how a new manager's effect depends on
league position — but it needs data to do it and is over-confident when
a season is young. The linear model cannot invent an interaction it has
no evidence for, which makes it markedly better calibrated early. The
blend is what gets served.

Draws are the hard class. A draw is almost never the single most likely
outcome, so a model tuned for accuracy alone learns to stop predicting
them entirely. Training therefore optimises log loss, and the site shows
the full three-way probability rather than just the pick.

### Scoreline — a Dixon-Coles bivariate Poisson goal model

Goals are low-count events, so the object to model is each side's goal
*rate*, not a label. Every club gets an attack and a defence strength;
the home side's expected goals is its attack times the opponent's
defence times a shared home-advantage term. A score matrix follows, and
from it the expected score and the most likely scoreline.

Three refinements:

- **Low-score correction** (the Dixon-Coles `rho`): independent Poisson
  under-predicts 0-0 and 1-1 and over-predicts 1-0 and 0-1.
- **Time decay**: older matches carry exponentially less weight, so the
  ratings follow a squad as it changes.
- **Championship results are fitted too**, with their own scoring-level
  offset. Promoted clubs otherwise get absurd ratings from three or four
  Premier League matches — a club with one goal in three games looks
  incapable of scoring. The clubs that go up and down each season tie the
  two scales together.

**Exact scores are much harder than outcomes.** A well-specified goal
model gets roughly one in eight or nine exactly right. The site shows
predicted and actual scorelines side by side and never marks them right
or wrong; only outcome predictions get a ✓ or ✗.

The displayed scoreline is the most likely score *consistent with the
predicted outcome*, not the most likely score overall. Those differ
often: a home win's probability is spread across 1-0, 2-0, 2-1 and the
rest, so 1-1 can be the single likeliest score even when a home win is
comfortably the likeliest result. Showing "home win, 1-1" would just
look broken. The unconditional expected goals are shown alongside and
are unaffected.

---

## The rolling retrain

The whole loop is one rule:

> Predictions for gameweek N are made by a model trained only on matches
> that kicked off before gameweek N's first fixture.

That holds in training and in serving, which is what makes the recorded
history meaningful.

1. `ingest` pulls the latest results into `matches`.
2. The feature table is rebuilt in one chronological pass. State only
   moves forward, so a feature *cannot* see a result that has not been
   fed in yet — the guarantee is structural, not a matter of remembering
   to filter.
3. Features for every match in a gameweek are computed from the state
   before that gameweek's **first** kick-off, never from Saturday's
   results when the fixture is on Monday. Training and serving therefore
   see the same information.
4. Both models are fitted on everything before the cutoff.
5. The gameweek is predicted and stored with a new `model_runs` row.

**Nothing is overwritten.** Each run records what it was trained on and
writes its own batch of predictions, so a prediction made before
kick-off stays on the record exactly as made. Re-running a gameweek adds
a run rather than editing one. The history page shows what was actually
forecast, not what today's model would say with hindsight.

The working database under `data/db/` is a build artefact and is not
committed, so a scheduled run rebuilds it from source and predicts only
the next gameweek. The committed serving database is therefore the
durable record: the pipeline restores the published forecast history
into the rebuilt database before predicting, which is what stops each
run from publishing the current gameweek alone and erasing everything
before it.

A prediction stored after its gameweek has already kicked off is marked
as such on the site. The model cannot cheat either way — features always
come from the state before the first fixture — but a forecast published
late is a weaker claim than one published in advance, and a reader
cannot tell the difference from the numbers alone.

---

## Data sources

| Source | Used for | Access | Coverage |
| --- | --- | --- | --- |
| [openfootball/england](https://github.com/openfootball/england) | Results, fixtures, gameweek numbers, kick-off dates for the Premier League, Championship, League One, FA Cup and EFL Cup | Public git repo, no key. Cloned once, then `git pull` | 2000-01 to date, updated within a day or two of each round |
| [datasets/football-datasets](https://github.com/datasets/football-datasets) | Shots, shots on target, corners, fouls, cards, referee | Public git repo, no key | 1993-94 onward, **lags the live season** — see below |
| FIFA / EA FC player ratings | Squad-quality proxy | Not vendored; build from any dump you have, see below | Depends on which dumps you supply |
| `data/manual/managers.csv` | Manager tenure and change features | Hand-maintained | 2025-26 onward |
| `data/manual/unavailability.csv` | Injuries, suspensions, international duty | Hand-maintained | Empty by default |
| `data/manual/european_participation.csv` | Midweek European commitments | Hand-maintained | 2026-27 |
| `data/manual/team_aliases.csv` | Club-name normalisation across sources | Hand-maintained | All sources |
| Derived from results | League table, form, congestion, head-to-head, Elo | — | Complete |

Everything the pipeline needs to predict the current gameweek comes from
openfootball, which is free and current. The rest improve the model
where they are available.

### Handling sources that lag the live season

Match statistics and video-game ratings are both published later than
results. A feature that exists for every training row but is missing for
the gameweek being predicted is worse than no feature at all, so the
pipeline handles the two cases explicitly:

- **Match statistics**: a feature column populated for fewer than half
  the fixtures being predicted is *dropped from that run's model*. The
  model is never asked to lean on evidence it will not have, and a
  single surviving value is not enough to justify keeping a feature it
  may have learned to depend on across thousands of training matches.
  The column returns automatically on the next retrain once the source
  catches up. Each run records which columns it dropped, visible on the
  `/model` page.
- **Squad ratings**: EA rate squads once a season, and the current
  season's edition is not available in usable form until well into it.
  The most recent rating is carried forward and its **age in seasons**
  is supplied as its own feature, so the model can discount a stale
  rating on its own terms rather than being handed a silent lie.

### Building the squad-quality table

Rating dumps are large (hundreds of megabytes) and are not vendored.
`scripts/build_squad_ratings.py` aggregates whichever you have into one
small club-per-season CSV. Two layouts are understood:

- `lbenz730/fifa_model` — one `player_stats_YYYY.csv` per game year
- sofifa-style exports — `players_22.csv`, EA FC data hubs, and the
  widely mirrored Kaggle "complete player dataset" files

```bash
python scripts/build_squad_ratings.py \
    --fifa-model-dir /path/to/fifa_model/stats \
    --sofifa /path/to/players_22.csv \
    --sofifa /path/to/fc26_players.csv
```

A game version maps to the season it shipped into: FIFA 20 was released
in September 2019 and describes 2019-20 squads. Second-tier squads are
included so promoted clubs have a rating in the season they come up.

The output lands in `data/external/squad_ratings.csv` and is picked up
on the next ingest. Gaps are fine — the carry-forward handles them.

---

## Manually maintained data

Three files under `data/manual/` are meant to be edited by hand. They
carry their own documentation in comment lines at the top, and `#` lines
are ignored on load.

**`managers.csv`** — one row per managerial spell. Covers 2025-26
onward; the exact day is unknown for some appointments and those rows are
marked `manual:approx` in the `source` column, which does not
meaningfully affect a "matches under this manager" feature. **Adding
older spells directly improves what the model can learn about managerial
changes** — with the current coverage the effect is estimated from about
two seasons of matches, so the features exist and are wired in but the
evidence behind them is thin.

**`unavailability.csv`** — key-player availability, one row per club per
gameweek. This is the hardest input to automate reliably and ships
empty. A club with no row for a gameweek is treated as *unknown*, not as
fully fit, so an empty file removes the feature rather than biasing it.
Filling it in even roughly (`players_out`, `key_players_out`) is the
single highest-value manual addition.

**`european_participation.csv`** — which clubs carry a midweek European
commitment, one row per club per season. Changes once a year.

---

## Running it

```bash
git clone <this repo> && cd Premier-League
python -m venv .venv && .venv/bin/pip install -r requirements-pipeline.txt

# First run: clone the source archives, build everything, and replay the
# gameweeks already played so the history page is populated
.venv/bin/python scripts/run_pipeline.py --backfill
.venv/bin/python scripts/export_web_db.py

# Then browse
.venv/bin/python scripts/serve.py          # http://127.0.0.1:5000
```

`requirements.txt` holds Flask and nothing else — it is what a
deployment installs, because the web layer imports no part of the data
or modelling stack. `requirements-pipeline.txt` adds pandas, LightGBM
and the rest, and is what you want locally.

The first run clones two source repositories into `data/raw/` and takes
a few minutes. Later runs `git pull` them and take well under a minute.

### After every gameweek

This is automated — see below — but the manual equivalent is:

```bash
.venv/bin/python scripts/run_pipeline.py
.venv/bin/python scripts/export_web_db.py
```

Ingest the new results, rebuild the features, retrain, predict the next
gameweek, then refresh the small database the site serves. Running
locally, the web app needs no restart; it reads the database on each
request.

### Automated updates

`.github/workflows/update-predictions.yml` does the whole loop and
commits the refreshed serving database, which is what makes the site
self-updating: the push triggers a redeploy, and nothing retrains on the
host.

It runs **every morning at 06:00 UTC** rather than on a weekly schedule,
because fixtures do not keep to one. A gameweek can be spread over four
days or sit a fortnight away behind an international break. Instead of
guessing, each run ingests the latest results and compares two numbers —
how many matches have been played this season, and which gameweek is
next — against the database the site is currently serving. If they
match, the run stops there: no retrain, no commit, no redeploy. A quiet
day costs about a minute of CI and changes nothing.

Before anything is committed the workflow serves the freshly exported
database and checks that the pages render. A published database the site
cannot read would take the whole site down until someone noticed, and
that check costs seconds.

To publish immediately without waiting for the schedule, run the
workflow by hand from the Actions tab; the **force** input retrains and
publishes even when no new results have arrived.

The workflow needs no secrets — the built-in `GITHUB_TOKEN` is enough,
with `contents: write` to push.

#### Why the job does not trust its own success

A scheduled workflow only ever runs on the repository's default branch,
but the site is deployed from whichever branch the host was pointed at.
Those were the same branch here until the default was renamed, and then
they were not: the job carried on pushing and reporting success while
production went on serving a branch that no longer received anything.
Eleven consecutive green runs, and a site stuck two gameweeks back.

Nothing inside the repository could have caught that, because from the
inside everything was working. So two things changed.

The job no longer depends on the two branch names staying in step. After
pushing to the branch it ran on, it carries the same commit onto every
other branch that can take it as a **fast-forward**. There is no
`--force` anywhere: a branch holding commits of its own is rejected by
the server and left untouched, so only branches that are strictly behind
— which is what a stale deployment branch is — move. Set the repository
variable `SYNC_DEPLOY_BRANCHES` to `false` to switch this off.

And every run, including one with nothing to publish, ends by asking the
site itself. `scripts/check_publication.py` confirms that every gameweek
already played has a prediction of record and that the gameweek about to
be played has been forecast; `scripts/check_live_site.py` then fetches
the deployed page and fails the job unless it is showing that gameweek,
polling for a few minutes because deployments are not instant. A failing
scheduled workflow emails the repository owner, which is the point:
silence should mean healthy.

Point the check at a different URL with the repository variable
`SITE_URL`, or `PLPRED_SITE_URL` when running it locally.

---

## Deploying the site

The web layer is deliberately separable from everything else: it loads
no model, computes nothing on a request, and reads a single pre-built
SQLite file. That makes it deployable as a small serverless function
with no database server and no writable disk.

`data/web/plpredict-web.db` is committed for exactly this reason. It is
built by `scripts/export_web_db.py`, which copies out only what the
pages query — Premier League fixtures, the predictions of record and the
model-run metadata. That is about 1.4 MB, against 61 MB for the working
database, most of which is the feature table no page touches. It is
exported without a write-ahead log so it can be opened read-only, which
is what a read-only filesystem requires.

### Vercel

`vercel.json` and `api/index.py` are set up already:

```bash
npm i -g vercel
vercel            # preview
vercel --prod     # production
```

Vercel installs the root `requirements.txt` (Flask alone), so the
function stays well inside the size limit; the full ML stack would not.
`includeFiles` in `vercel.json` is what ships the serving database
alongside the function.

New gameweeks are published by the scheduled workflow, which commits the
refreshed serving database; the push triggers the redeploy. Nothing
retrains on the host.

### Anywhere else

Any WSGI host works, with no Vercel-specific pieces involved:

```bash
pip install -r requirements.txt gunicorn
gunicorn 'plpredict.web.app:app'
```

Set `PLPRED_WEB_DB` to serve a database from another location. If no
serving database is present, the app falls back to the pipeline's
working database, which is what happens during local development.

### Other commands

```bash
# Data only, no training
.venv/bin/python scripts/ingest.py

# Re-predict one gameweek (adds a run, keeps the old one)
.venv/bin/python scripts/run_pipeline.py --matchday 7

# Walk-forward backtest: predict each gameweek knowing only what came before
.venv/bin/python scripts/backtest.py --seasons 2024-25 2025-26 --from-matchday 4

# Work offline against the archives already in data/raw/
.venv/bin/python scripts/run_pipeline.py --offline

# Is the serving database still forecasting every gameweek?
.venv/bin/python scripts/check_publication.py

# Is the deployed site actually showing it?
.venv/bin/python scripts/check_live_site.py --gameweek 6

# Tests
.venv/bin/python -m pytest tests/
```

### Configuration

Everything path- and season-related lives in `plpredict/config.py` and
can be overridden by environment variable:

| Variable | Default | Meaning |
| --- | --- | --- |
| `PLPRED_SEASON` | `2026-27` | Season being predicted |
| `PLPRED_FIRST_SEASON` | `2010-11` | Earliest season in the training corpus |
| `PLPRED_DB` | `data/db/plpredict.db` | SQLite database |
| `PLPRED_DATA_DIR` | `data/` | Root for raw, manual and external data |
| `PLPRED_MODEL_DIR` | `models/artifacts/` | Saved model artefacts |

The number of boosting rounds is not among them: it is chosen by early
stopping on a chronological tail of the training set at every retrain,
because the right value drifts as the season's data accumulates.

Model hyper-parameters are dataclasses in the same file
(`FeatureConfig`, `OutcomeModelConfig`, `ScorelineModelConfig`).

---

## Layout

The three layers are separate so any one can be replaced without
touching the others.

```
plpredict/
  config.py                  paths, seasons, hyper-parameters
  db.py                      SQLite schema and helpers

  data/                      ── DATA PIPELINE: fetch, clean, store
    sources/openfootball.py    results and fixtures (the primary source)
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

.github/workflows/          scheduled update job
api/index.py                 serverless entrypoint (Vercel)
vercel.json                  deployment config
scripts/                     thin CLI wrappers
data/manual/                 hand-maintained context (versioned)
data/web/                    the committed serving database
tests/
```

### Database

| Table | Holds |
| --- | --- |
| `matches` | Every fixture, played or scheduled, all competitions |
| `match_stats` | Shots, cards, corners, referee, joined to `matches` |
| `managers`, `unavailability`, `squad_ratings`, `european_participation` | Context that is not derivable from results |
| `features` | The versioned, point-in-time feature table, one JSON payload per match |
| `model_runs` | One row per retrain: what it trained on, which features it used and dropped, and what it leaned on |
| `predictions` | Every prediction ever made, keyed by run |
| `current_predictions` | The prediction of record for each match |

---

## What the model is actually worth

Measured by `scripts/backtest.py`, which replays the platform's own
procedure gameweek by gameweek — predicting each one knowing only what
came before it, then moving on. Over 2023-24, 2024-25 and 2025-26 from
gameweek 4 onwards (1,050 matches):

| Metric | Model | Baseline |
| --- | --- | --- |
| Outcome accuracy | **52.1%** | 43.2% (always predict a home win) |
| Log loss | **0.9948** | 1.0061 (Elo-only) |
| Brier score | 0.5932 | — |
| Exact scoreline | 8.2% | — |
| Goals mean absolute error | 0.93 per side | — |

Per season: 57.1% / 52.0% / 47.1%. The spread is mostly the seasons
rather than the model — 2025-26 had eleven managerial changes and an
unusually flat table.

Two things worth reading carefully.

**Calibration is good, which matters more than accuracy here** because
the platform publishes probabilities rather than just picks:

| Predicted | Observed | n |
| --- | --- | --- |
| 0.16 | 0.18 | 552 |
| 0.25 | 0.25 | 1169 |
| 0.35 | 0.34 | 438 |
| 0.45 | 0.44 | 383 |
| 0.55 | 0.52 | 273 |
| 0.65 | 0.65 | 170 |
| 0.74 | 0.70 | 100 |

When the model says 65%, it happens about 65% of the time. Getting this
right is what the linear half of the ensemble and the early-stopped
boosting rounds are for — a fixed 400 rounds produced visible
over-confidence in the 0.5-0.8 band and cost about 0.03 nats.

**Draws are essentially never the pick.** A draw is almost never the
single most likely outcome of a football match, so the argmax avoids
them entirely; the model instead expresses draws as probability mass,
typically 25-30%. This is why the site shows the full three-way split
rather than only the pick, and why log loss is the number to watch. A
model tuned to predict more draws would score better on draws and worse
on everything else.

Roughly half of matches called correctly, nine points clear of always
backing the home side, and a log loss meaningfully below a tuned Elo
baseline is a reasonable place for a model of this kind to sit. The
gains still on the table are listed below, in the order likely to pay.

## Known limitations

- **Injuries are not automated.** `unavailability.csv` ships empty. This
  is the largest gap: team news is genuinely hard to scrape reliably and
  is the single most valuable thing to add.
- **Manager history starts at 2025-26.** The features are wired in and
  correct, but the model has roughly two seasons of matches to learn the
  effect from. Extending the file backwards is cheap and helps directly.
- **No expected-goals data.** Shots on target is the stand-in. A real xG
  feed (Understat or a paid provider) would slot straight into the
  `match_stats` table.
- **Match statistics lag the live season.** Handled explicitly by
  dropping the affected columns per run, but it does mean the live model
  is thinner than the backtested one.
- **Squad ratings are a season stale by construction.** The rating age is
  supplied as a feature so the model can account for it.
- **Fixtures come from one source.** A postponement is picked up on the
  next `git pull`, which is usually same-day but not instant.
