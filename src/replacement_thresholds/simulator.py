"""Repeated assessment with a finite reserve of replacement items.

A world has 48 learners and six active items. Every period provides 96 responses. Replacing an
active item spends ``calibration_size`` of those responses on calibrating the new item and uses
one item from the reserve. Each item becomes compromised after a geometric number of exposures;
a compromised item is answered correctly with extra probability, which biases learner estimates.

The replacement rule compares the largest standardized residual sum of the active items with a
threshold ``z * ((T - t) / (R_t + 1) / (T / (R_0 + 1))) ** gamma``. With ``gamma = 0`` the rule is
the static threshold ``z``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

LEARNERS = 48
ACTIVE_ITEMS = 6
RESPONSES_PER_PERIOD = 96
POOL_SIZE = ACTIVE_ITEMS + 24
PANEL_SIZE = 80
ROUNDS_PER_PERIOD = 2

ENDPOINT_FIELDS = (
    "mse",
    "rmse",
    "late_mse",
    "late_rmse",
    "bias",
    "false_mastery",
    "true_q",
    "replacements",
    "calibration",
    "diagnostic",
    "early_replacement_fraction",
    "first_exhaustion",
    "reserve_exhausted",
)
PERIOD_FIELDS = (
    "period",
    "mse",
    "bias",
    "false_mastery",
    "true_q",
    "estimated_q",
    "replacements",
    "calibration",
    "diagnostic",
    "reserve_left",
    "decision_threshold",
    "reserve_before",
    "remaining_horizon",
    "selected_slot",
)


@dataclass(frozen=True)
class Scenario:
    reserve: int
    calibration_size: int
    hazard: float
    horizon: int = 30
    rehearsal: float = 0.25
    diagnostic_gain: float = 0.006
    filter_increment: float = 0.025

    def __post_init__(self):
        if not 0 <= self.reserve <= POOL_SIZE - ACTIVE_ITEMS:
            raise ValueError("reserve exceeds the item pool")
        if not 0 < self.calibration_size < RESPONSES_PER_PERIOD:
            raise ValueError("calibration size must be inside the response budget")
        if not np.isfinite(self.filter_increment) or self.filter_increment < 0:
            raise ValueError("filter increment must be finite and nonnegative")


@dataclass(frozen=True)
class Policy:
    z: float
    gamma: float


@dataclass
class WorldResult:
    endpoints: dict
    periods: list
    replacements: list
    responses: list


def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -35, 35)))


def calibrate(difficulty, uniforms):
    """Posterior mean and variance of item difficulty from a Rasch calibration panel.

    Panel learners have ability 0; the prior is N(0, 0.7^2) and the posterior is evaluated on a grid.
    """
    grid = np.linspace(-3, 3, 241)
    successes = int(np.sum(uniforms < sigmoid(-difficulty)))
    p = sigmoid(-grid)
    log_weight = -0.5 * (grid / 0.7) ** 2 + successes * np.log(p) + (len(uniforms) - successes) * np.log1p(-p)
    weight = np.exp(log_weight - log_weight.max())
    weight /= weight.sum()
    mean = float(weight @ grid)
    variance = float(weight @ (grid - mean) ** 2)
    return mean, variance, successes


def decision_threshold(policy, t, reserve_left, initial_reserve, horizon):
    remaining = horizon - t
    return float(policy.z) * (remaining / (reserve_left + 1) / (horizon / (initial_reserve + 1))) ** policy.gamma


def select_slot(policy, t, residual_sum, information, reserve_left, initial_reserve, horizon):
    """Active slot to replace, or -1. Uses only response history observed before period ``t``."""
    if reserve_left <= 0 or t == 0:
        return -1
    z = residual_sum / np.sqrt(np.maximum(information, 1.0))
    threshold = decision_threshold(policy, t, reserve_left, initial_reserve, horizon)
    return int(np.argmax(z)) if max(z) > threshold else -1


def run_world(seed, scenario, policy, *, label="", trace=False):
    """Simulate one world. The same seed gives the same learners and items under every policy."""
    horizon = scenario.horizon

    def rng(stream):
        return np.random.default_rng(np.random.SeedSequence([seed, stream]))

    difficulty = rng(0).normal(0, 0.7, POOL_SIZE)
    panel_u = rng(1).random((POOL_SIZE, PANEL_SIZE))
    compromise_u = rng(2).random(POOL_SIZE)
    if scenario.hazard:
        compromise_after = np.ceil(np.log1p(-compromise_u) / np.log1p(-scenario.hazard))
    else:
        compromise_after = np.full(POOL_SIZE, np.inf)
    # Stream 3 draws item discrimination in misspecified variants and is unused here.
    initial_ability = rng(4).normal(-0.4, 0.55, LEARNERS)
    growth_rate = rng(5).uniform(0.017, 0.035, LEARNERS)
    practice = rng(6).poisson(2, (horizon, LEARNERS))
    noise = rng(7).normal(0, 0.018, (horizon, LEARNERS))
    answer_u = rng(8).random((horizon, ROUNDS_PER_PERIOD, LEARNERS))
    assignment_u = rng(9).random((horizon, ROUNDS_PER_PERIOD, LEARNERS))

    theta = initial_ability.copy()
    mu = np.full(LEARNERS, -0.4)
    var = np.full(LEARNERS, 0.55**2)
    b_mean = np.zeros(POOL_SIZE)
    b_var = np.zeros(POOL_SIZE)
    b_successes = np.zeros(POOL_SIZE, dtype=int)
    for j in range(ACTIVE_ITEMS):
        b_mean[j], b_var[j], b_successes[j] = calibrate(difficulty[j], panel_u[j])

    active = np.arange(ACTIVE_ITEMS)
    exposure = np.zeros(POOL_SIZE, dtype=int)
    seen = np.zeros((LEARNERS, POOL_SIZE), dtype=int)
    last_seen = np.full((LEARNERS, POOL_SIZE), -100)
    residual_sum = np.zeros(POOL_SIZE)
    response_count = np.zeros(POOL_SIZE)
    information = np.zeros(POOL_SIZE)

    replacements = 0
    total_calibration = 0
    periods, events, responses = [], [], []

    for t in range(horizon):
        theta += growth_rate * practice[t] - 0.01 + noise[t]
        mu += 0.042
        var = np.minimum(var + scenario.filter_increment, 1.5)

        reserve_before = scenario.reserve - replacements
        threshold = decision_threshold(policy, t, reserve_before, scenario.reserve, horizon)
        slot = select_slot(
            policy, t, residual_sum[active], information[active], reserve_before, scenario.reserve, horizon
        )
        calibration = 0
        if slot >= 0:
            retired = int(active[slot])
            new = ACTIVE_ITEMS + replacements
            calibration = scenario.calibration_size
            b_mean[new], b_var[new], b_successes[new] = calibrate(difficulty[new], panel_u[new, :calibration])
            active[slot] = new
            replacements += 1
            total_calibration += calibration
            events.append(
                dict(
                    seed=seed,
                    scenario=label,
                    period=t + 1,
                    retired=retired,
                    new=new,
                    calibration_n=calibration,
                    calibration_correct=int(b_successes[new]),
                    estimated_b=b_mean[new],
                    variance_b=b_var[new],
                    true_b=difficulty[new],
                )
            )

        diagnostic = RESPONSES_PER_PERIOD - calibration
        residual_sum *= 0.9
        response_count *= 0.9
        information *= 0.81

        order = np.roll(np.arange(LEARNERS), t % LEARNERS)
        for r in range(ROUNDS_PER_PERIOD):
            people = order[: min(LEARNERS, max(diagnostic - r * LEARNERS, 0))]
            if not len(people):
                continue
            items = active[np.minimum((assignment_u[t, r, people] * ACTIVE_ITEMS).astype(int), ACTIVE_ITEMS - 1)]
            base = sigmoid(theta[people] - difficulty[items])
            familiarity = 1 - np.exp(-seen[people, items] / 2)
            recency = np.exp(-(t - last_seen[people, items]) / 7)
            rehearsal = scenario.rehearsal * familiarity * recency
            leakage = 0.45 * (exposure[items] >= compromise_after[items])
            probability = 1 - (1 - base) * (1 - rehearsal) * (1 - leakage)
            y = (answer_u[t, r, people] < probability).astype(float)

            predicted = sigmoid(mu[people] - b_mean[items])
            info = predicted * (1 - predicted)
            weight = 1 / (1 + info * b_var[items])
            new_var = 1 / (1 / var[people] + weight * info)
            mu[people] += new_var * weight * (y - predicted)
            var[people] = new_var

            np.add.at(residual_sum, items, y - predicted)
            np.add.at(response_count, items, 1)
            np.add.at(information, items, info)
            if trace:
                for ix, i in enumerate(people):
                    responses.append(
                        dict(
                            seed=seed,
                            scenario=label,
                            period=t + 1,
                            round=r + 1,
                            learner=int(i),
                            item=int(items[ix]),
                            response=int(y[ix]),
                            true_theta_pre=theta[i],
                            probability=probability[ix],
                            rehearsal=rehearsal[ix],
                            leakage=leakage[ix],
                            predicted_probability=predicted[ix],
                        )
                    )
            seen[people, items] += 1
            last_seen[people, items] = t
            np.add.at(exposure, items, 1)
            theta[people] += scenario.diagnostic_gain * (1 - sigmoid(theta[people]))
            mu[people] += scenario.diagnostic_gain * (1 - sigmoid(mu[people]))

        q = sigmoid(theta)
        q_hat = sigmoid(mu)
        periods.append(
            dict(
                period=t + 1,
                mse=float(np.mean((q_hat - q) ** 2)),
                bias=float(np.mean(q_hat - q)),
                false_mastery=float(np.mean((q_hat >= 0.75) & (q < 0.75))),
                true_q=float(q.mean()),
                estimated_q=float(q_hat.mean()),
                replacements=replacements,
                calibration=calibration,
                diagnostic=diagnostic,
                reserve_left=scenario.reserve - replacements,
                decision_threshold=threshold,
                reserve_before=reserve_before,
                remaining_horizon=horizon - t,
                selected_slot=slot,
            )
        )

    mse = float(np.mean([p["mse"] for p in periods]))
    late_mse = float(np.mean([p["mse"] for p in periods[horizon // 3 :]]))
    early = sum(e["period"] <= horizon // 3 for e in events)
    endpoints = dict(
        mse=mse,
        rmse=np.sqrt(mse),
        late_mse=late_mse,
        late_rmse=np.sqrt(late_mse),
        bias=float(np.mean([p["bias"] for p in periods])),
        false_mastery=float(np.mean([p["false_mastery"] for p in periods])),
        true_q=float(np.mean([p["true_q"] for p in periods])),
        replacements=replacements,
        calibration=total_calibration,
        diagnostic=horizon * RESPONSES_PER_PERIOD - total_calibration,
        early_replacement_fraction=early / replacements if replacements else 0.0,
        first_exhaustion=next((p["period"] for p in periods if p["reserve_left"] == 0), horizon + 1),
        reserve_exhausted=int(replacements == scenario.reserve),
    )
    return WorldResult(endpoints, periods, events, responses)
