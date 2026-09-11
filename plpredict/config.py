"""Central configuration.

Everything that varies between environments (paths, the season being
predicted, model hyper-parameters) is collected here so the data,
feature and model layers never hard-code a location.

Every path can be overridden with an environment variable, which keeps
the scheduled job configurable without editing code.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _path(env_var: str, default: Path) -> Path:
    raw = os.environ.get(env_var)
    return Path(raw).expanduser().resolve() if raw else default


DATA_DIR = _path("PLPRED_DATA_DIR", PROJECT_ROOT / "data")
RAW_DIR = DATA_DIR / "raw"
MANUAL_DIR = DATA_DIR / "manual"
EXTERNAL_DIR = DATA_DIR / "external"
DB_PATH = _path("PLPRED_DB", DATA_DIR / "db" / "plpredict.db")
# The slim, read-only database the website serves from. Built by
# scripts/export_web_db.py and committed, so a deployment needs no data
# pipeline and no writable disk.
WEB_DB_PATH = _path("PLPRED_WEB_DB", DATA_DIR / "web" / "plpredict-web.db")
MODEL_DIR = _path("PLPRED_MODEL_DIR", PROJECT_ROOT / "models" / "artifacts")

# The season the platform is currently predicting, in openfootball's
# "YYYY-YY" directory form.
CURRENT_SEASON = os.environ.get("PLPRED_SEASON", "2026-27")

# Earliest season pulled into the training corpus. Ten-plus years of
# matches is enough for the outcome model without dragging in an era of
# football that no longer resembles the current one.
FIRST_TRAINING_SEASON = os.environ.get("PLPRED_FIRST_SEASON", "2010-11")

# Competitions ingested from openfootball. The league feeds the model's
# target; the others exist so that fixture congestion and rotation risk
# can see midweek cup and second-tier commitments.
OPENFOOTBALL_REPO = os.environ.get(
    "PLPRED_OPENFOOTBALL_REPO", "https://github.com/openfootball/england"
)
OPENFOOTBALL_CHECKOUT = _path("PLPRED_OPENFOOTBALL_DIR", RAW_DIR / "openfootball-england")

# Optional enrichment: football-data.co.uk match statistics, consumed
# through a mirror that republishes them with stable column names.
FOOTBALLDATA_REPO = os.environ.get(
    "PLPRED_FOOTBALLDATA_REPO", "https://github.com/datasets/football-datasets"
)
FOOTBALLDATA_CHECKOUT = _path("PLPRED_FOOTBALLDATA_DIR", RAW_DIR / "football-datasets")

COMPETITION_FILES = {
    "premier_league": "1-premierleague.txt",
    "championship": "2-championship.txt",
    "league_one": "3-league1.txt",
    "fa_cup": "facup.txt",
    "efl_cup": "eflcup.txt",
}

# Only the Premier League is modelled; the rest are context.
TARGET_COMPETITION = "premier_league"


@dataclass(frozen=True)
class FeatureConfig:
    """Knobs for the feature-engineering layer."""

    # Rolling-form windows (in matches) evaluated for every team.
    form_windows: tuple[int, ...] = (4, 6, 10)
    # Elo parameters. k_factor is the update size; home_advantage is in
    # Elo points; regression pulls ratings toward the mean between
    # seasons so that a squad overhaul is not held against a club for
    # ever.
    elo_k: float = 20.0
    elo_home_advantage: float = 60.0
    elo_season_regression: float = 0.25
    elo_initial: float = 1500.0
    # Second-tier teams start below the Premier League mean, which gives
    # promoted sides a sane cold-start rating instead of an average one.
    elo_division_offsets: dict[str, float] = field(
        default_factory=lambda: {
            "premier_league": 0.0,
            "championship": -180.0,
            "league_one": -320.0,
        }
    )
    # A match is "congested" relative to this many days of rest.
    normal_rest_days: float = 7.0
    max_rest_days: float = 21.0
    # Games older than this drop out of the rolling-form windows, so the
    # previous season's run-in fades out over the opening weeks.
    form_max_age_days: int = 120


@dataclass(frozen=True)
class OutcomeModelConfig:
    """Gradient-boosted classifier for home win / draw / away win."""

    learning_rate: float = 0.04
    num_leaves: int = 16
    min_child_samples: int = 60
    subsample: float = 0.85
    subsample_freq: int = 1
    colsample_bytree: float = 0.7
    reg_lambda: float = 5.0
    random_state: int = 7
    # The number of boosting rounds is chosen by early stopping rather
    # than fixed, because it is the single setting this model is most
    # sensitive to and the right value drifts as the training set grows
    # through a season. On this data a fixed 400 rounds costs about
    # 0.03 nats of log loss against a stopped fit of roughly 60.
    max_estimators: int = 1200
    early_stopping_rounds: int = 60
    # Fraction of the (chronologically ordered) training set held out to
    # stop on. Held out from the end, so the probe is validated on the
    # most recent matches rather than a random sample of history.
    validation_fraction: float = 0.15
    min_estimators: int = 40
    # Blend weight given to the gradient-boosted model when it is mixed
    # with the calibrated multinomial baseline. The blend is what gets
    # served; it is materially better calibrated early in a season when
    # the booster has little current-season signal.
    blend_weight: float = 0.6


@dataclass(frozen=True)
class ScorelineModelConfig:
    """Dixon-Coles bivariate Poisson goal model."""

    # Exponential time decay, in units of 1/day. Matches roughly two
    # years old carry about half the weight of today's.
    time_decay: float = 0.0011
    max_goals: int = 8
    # L2 shrinkage on attack/defence strengths, which keeps promoted
    # teams with a handful of matches from acquiring extreme ratings.
    ridge: float = 0.35
    # Matches older than this are dropped before fitting: at the decay
    # rate above they carry almost no weight, and keeping them leaves
    # long-departed clubs with ratings built on decade-old form.
    max_history_days: int = 1500
    # Second-tier results are fitted alongside the top flight so that
    # promoted clubs arrive with a rating built on a full season.
    include_competitions: tuple[str, ...] = ("premier_league", "championship")


FEATURES = FeatureConfig()
OUTCOME_MODEL = OutcomeModelConfig()
SCORELINE_MODEL = ScorelineModelConfig()


def ensure_dirs() -> None:
    """Create every directory the pipeline writes to."""
    for directory in (RAW_DIR, MANUAL_DIR, EXTERNAL_DIR, DB_PATH.parent, MODEL_DIR):
        directory.mkdir(parents=True, exist_ok=True)
