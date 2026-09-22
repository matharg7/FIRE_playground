"""Continual-learning metrics over the task-by-task score matrix.

R[t][i] is the score on task i after training on tasks 0..t (NaN if not
evaluated). Definitions follow the TRACE paper, after step t:

    OP_t  = mean_{i<=t} R[t][i]                     overall performance
    BWT_t = mean_{i<t} (R[t][i] - R[i][i])          backward transfer (< 0: forgetting)
    FWT_t = mean_{0<i<=t} (R[i-1][i] - b_i)          forward transfer, needs
                                                    zero-shot baselines b_i and
                                                    R[i-1][i] (task i evaluated
                                                    before it is trained)
The final-run values are those at t = T-1.
"""

import json
import math

import numpy as np


class ScoreMatrix:
    def __init__(self, tasks):
        self.tasks = list(tasks)
        self.R = np.full((len(self.tasks), len(self.tasks)), np.nan)
        self.baselines = {}       # zero-shot scores of the initial model, for FWT
        self.losses = {}          # {step: {task: eval loss}}

    def record(self, t, scores):
        """Store the scores after training step t; scores is {task: float}."""
        for task, value in scores.items():
            self.R[t, self.tasks.index(task)] = float(value)

    def _check_row(self, t, upto):
        missing = [self.tasks[i] for i in range(upto + 1) if math.isnan(self.R[t, i])]
        if missing:
            raise ValueError(f"step {t}: no score for {missing}")

    def op(self, t=None):
        t = len(self.tasks) - 1 if t is None else t
        self._check_row(t, t)
        return float(np.mean(self.R[t, : t + 1]))

    def bwt(self, t=None):
        t = len(self.tasks) - 1 if t is None else t
        if t == 0:
            return 0.0
        self._check_row(t, t)
        diag = np.diag(self.R)[:t]
        if np.isnan(diag).any():
            raise ValueError("BWT needs R[i][i] for every earlier task")
        return float(np.mean(self.R[t, :t] - diag))

    def fwt(self, baselines=None, t=None):
        """baselines: {task: zero-shot score of the initial model}; defaults to
        the ones recorded by record_baselines()."""
        baselines = self.baselines if baselines is None else baselines
        if not baselines:
            raise ValueError("FWT needs zero-shot baselines (--eval_zero_shot)")
        t = len(self.tasks) - 1 if t is None else t
        gaps = [self.R[i - 1, i] - baselines[self.tasks[i]] for i in range(1, t + 1)]
        if not gaps or any(math.isnan(g) for g in gaps):
            raise ValueError("FWT needs R[i-1][i] (evaluate task i before training it)")
        return float(np.mean(gaps))

    def can_fwt(self, t=None):
        """True when baselines exist and every task was scored before training it."""
        t = len(self.tasks) - 1 if t is None else t
        return bool(self.baselines) and t > 0 and not any(
            math.isnan(self.R[i - 1, i]) for i in range(1, t + 1))

    def summary(self, t):
        """OP_t, BWT_t and (when available) FWT_t, for logging after each step."""
        out = {"op": self.op(t), "bwt": self.bwt(t)}
        if self.can_fwt(t):
            out["fwt"] = self.fwt(t=t)
        return out

    def record_baselines(self, scores):
        """Zero-shot scores of the untrained model, one per task."""
        self.baselines = {task: float(v) for task, v in scores.items()}

    def record_losses(self, t, losses):
        """Teacher-forced eval loss per task after step t (t = -1: zero-shot)."""
        self.losses[str(t)] = {task: float(v) for task, v in losses.items()}

    def to_dict(self):
        rows = [[None if math.isnan(x) else float(x) for x in row] for row in self.R]
        return {"tasks": self.tasks, "R": rows, "baselines": self.baselines,
                "losses": self.losses}

    @classmethod
    def from_dict(cls, d):
        m = cls(d["tasks"])
        m.R = np.array([[np.nan if x is None else x for x in row] for row in d["R"]], dtype=float)
        m.baselines = d.get("baselines", {})
        m.losses = d.get("losses", {})
        return m

    def save(self, path):
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=1)

    @classmethod
    def load(cls, path):
        with open(path) as f:
            return cls.from_dict(json.load(f))
