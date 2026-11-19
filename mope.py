"""Mixture of Probing Experts (MoPE).

MoPE selects, for each tenant of a multi-tenant cloud, one classical probing design (an "expert") for isolating faulty virtual machines
under noisy verification probes. The router uses only context that is observable at deployment time:

    x = [log m, log rho_hat, a1, a2, log(e1 + 1e-3), log(e2 + 1e-3)]

where m is the number of virtual machines of the tenant, rho_hat is a pilot-based prevalence estimate, and (a1, a2, e1, e2) are the calibrated
activation probabilities and false-alarm probabilities of the two probe readouts.

For every expert k, two ensembles of bootstrap gradient-boosted regressors predict the error rate e_k(x) and the executions per node q_k(x).
For an operator cost weight c the router returns the expert with the lowest risk-adjusted utility

    score_k(x; c) = e_k(x) + c * q_k(x) + beta * sqrt(s_e_k(x)^2 + c^2 * s_q_k(x)^2),

where s_e_k and s_q_k are the standard deviations of the ensembles. The regressors do not depend on c, so one fitted model serves every
cost weight.

Training and scoring tables are pandas DataFrames with one row per tenant and the columns

    m, rho_hat, a1, a2, e1, e2                 context of the tenant
    pilot_execs                                pilot executions per node (used by `realised_utility`)
    <expert>_err, <expert>_execs               error rate and executions per node of each expert (training and realised scoring only)

Dependencies: numpy, pandas, lightgbm.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd
import lightgbm as lgb

EXPERTS = ("Ind1", "Ind3", "Ind5", "Dorf1", "Dorf3", "Hwang", "Halving", "COMP", "SPRT10", "SPRT1", "SPRT01")
NEEDS_PREVALENCE = frozenset({"Dorf1", "Dorf3", "Hwang", "COMP"})   # experts that use the pilot prevalence estimate
CONTEXT_COLUMNS = ("m", "rho_hat", "a1", "a2", "e1", "e2")


# ---------------------------------------------------------------------------------------------------------------- calibration
def estimate_noise(readouts_bit1_faulty, readouts_bit2_faulty, readouts_clean):
    """Estimate probe noise from singleton probes on labelled nodes.

    Each argument is a boolean array of shape (n, 2) holding the two readouts of singleton probes on nodes that are
    faulty in bit 1 only, faulty in bit 2 only, and healthy. Returns (a1, a2, e1, e2): the activation probability and the
    false-alarm probability of each readout.
    """
    r1, r2, r0 = (np.asarray(r, float) for r in (readouts_bit1_faulty, readouts_bit2_faulty, readouts_clean))
    e1 = max(r0[:, 0].mean(), 1e-3)
    e2 = max(r0[:, 1].mean(), 1e-3)
    a1 = float(np.clip((r1[:, 0].mean() - e1) / (1 - e1), 0.3, 0.98))
    a2 = float(np.clip((r2[:, 1].mean() - e2) / (1 - e2), 0.3, 0.98))
    return a1, a2, float(e1), float(e2)


def estimate_prevalence(positive_rate, noise):
    """Fault prevalence from the share of positive singleton probes on a random pilot sample of nodes.

    `noise` is the tuple (a1, a2, e1, e2) returned by `estimate_noise`. The estimate is clipped to [0.003, 0.6].
    """
    a1, a2, e1, e2 = noise
    a = 0.5 * (a1 + a2)
    e = 1 - (1 - e1) * (1 - e2)
    return float(np.clip((positive_rate - e) / max(a * (1 - e), 0.3), 0.003, 0.6))


# ---------------------------------------------------------------------------------------------------------------------- model
def context_matrix(table: pd.DataFrame) -> np.ndarray:
    """Context features x = [log m, log rho_hat, a1, a2, log(e1 + 1e-3), log(e2 + 1e-3)] as a float32 matrix."""
    missing = [c for c in CONTEXT_COLUMNS if c not in table.columns]
    if missing:
        raise KeyError(f"missing context columns: {missing}")
    return np.stack([np.log(table["m"]), np.log(table["rho_hat"]), table["a1"], table["a2"],
                     np.log(table["e1"] + 1e-3), np.log(table["e2"] + 1e-3)], 1).astype(np.float32)


class MoPE:
    """Risk-aware top-1 router over a pool of probing designs.

    Parameters
    ----------
    experts : names of the probing designs; the training table must contain `<name>_err` and `<name>_execs` for each.
    n_boot : number of bootstrap regressors per expert and target.
    n_estimators : trees per regressor.
    beta : weight of the ensemble standard deviation in the routing score.
    seed : seed of the bootstrap resampling.
    """

    def __init__(self, experts: Sequence[str] = EXPERTS, n_boot: int = 5, n_estimators: int = 150, beta: float = 1.0, seed: int = 0):
        self.experts = tuple(experts)
        self.n_boot, self.n_estimators, self.beta, self.seed = n_boot, n_estimators, beta, seed
        self.models_ = None

    # -- fitting -----------------------------------------------------------------------------------------------------------
    def fit(self, table: pd.DataFrame) -> "MoPE":
        """Fit the error and execution ensembles of every expert on a table of tenant outcomes."""
        x = context_matrix(table)
        err, ex = self._outcomes(table)
        rng = np.random.default_rng(self.seed)
        self.models_ = []
        for k in range(len(self.experts)):
            per_target = []
            for target in (err[:, k], ex[:, k]):
                ensemble = []
                for _ in range(self.n_boot):
                    idx = rng.choice(len(x), len(x))
                    reg = lgb.LGBMRegressor(n_estimators=self.n_estimators, learning_rate=0.05, num_leaves=8, min_child_samples=20,
                                            subsample=0.8, subsample_freq=1, verbose=-1, n_jobs=1)
                    ensemble.append(reg.fit(x[idx], target[idx]))
                per_target.append(ensemble)
            self.models_.append(per_target)
        return self

    def _outcomes(self, table: pd.DataFrame):
        cols = [f"{e}_{s}" for e in self.experts for s in ("err", "execs")]
        missing = [c for c in cols if c not in table.columns]
        if missing:
            raise KeyError(f"missing outcome columns: {missing}")
        err = np.stack([table[f"{e}_err"] for e in self.experts], 1)
        ex = np.stack([table[f"{e}_execs"] for e in self.experts], 1)
        return err, ex

    # -- scoring -----------------------------------------------------------------------------------------------------------
    def predict(self, table: pd.DataFrame):
        """Predicted error and executions per node of every expert.

        Returns (mean, std), each of shape (n_tenants, n_experts, 2); the last axis is (error, executions).
        """
        if self.models_ is None:
            raise RuntimeError("call fit() first")
        x = context_matrix(table)
        mean = np.zeros((len(x), len(self.experts), 2))
        std = np.zeros_like(mean)
        for k in range(len(self.experts)):
            for j in range(2):
                p = np.stack([m.predict(x) for m in self.models_[k][j]])
                mean[:, k, j], std[:, k, j] = p.mean(0), p.std(0)
        return mean, std

    def score(self, table: pd.DataFrame, c: float, beta: float | None = None) -> np.ndarray:
        """Risk-adjusted utility of every expert for each tenant, shape (n_tenants, n_experts); lower is better."""
        mean, std = self.predict(table)
        beta = self.beta if beta is None else beta
        risk = np.sqrt(std[:, :, 0] ** 2 + (c * std[:, :, 1]) ** 2)
        return mean[:, :, 0] + c * mean[:, :, 1] + beta * risk

    def route(self, table: pd.DataFrame, c: float, beta: float | None = None) -> np.ndarray:
        """Index of the selected expert for each tenant."""
        return np.argmin(self.score(table, c, beta), 1)

    def route_names(self, table: pd.DataFrame, c: float, beta: float | None = None) -> list[str]:
        """Name of the selected expert for each tenant."""
        return [self.experts[i] for i in self.route(table, c, beta)]

    def realised_utility(self, table: pd.DataFrame, choice: np.ndarray, c: float, charge_pilot: bool = True) -> float:
        """Mean realised utility error + c * executions per node of a routing `choice` on a table with observed outcomes.

        The pilot executions (column `pilot_execs`) are charged to every routed tenant when `charge_pilot` is true.
        """
        err, ex = self._outcomes(table)
        rows = np.arange(len(table))
        cost = ex[rows, choice] + (table["pilot_execs"].to_numpy() if charge_pilot else 0.0)
        return float((err[rows, choice] + c * cost).mean())
