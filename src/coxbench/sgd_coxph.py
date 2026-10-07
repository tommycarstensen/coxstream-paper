"""
Mini-batch SGD for Cox proportional hazards (BigSurvSGD algorithm).

Python reimplementation of Tarkhan & Simon (2021) "BigSurvSGD: Big Survival
Data Analysis via Stochastic Gradient Descent", Statistics and Computing.
Original R/C++ package: https://github.com/atarkhan/bigSurvSGD

Algorithm
---------
Data are partitioned into strata of fixed size S. For each stratum, the
partial-likelihood risk set is approximated by the S subjects within the
stratum (sorted by event time). The gradient is the Breslow partial-likelihood
gradient on this approximate risk set. Gradients from batch_size strata are
averaged before each parameter update.

This is an APPROXIMATE method: each risk set is restricted to the subject's
own stratum and the optimiser is stochastic, so the result is not the
maximiser of the full partial likelihood. tests/test_sgd_coxph.py checks the
stratum gradient against explicit risk sets.

Optimisers: SGD, ADAM (Kingma & Ba 2015), AMSGrad (Reddi et al. 2018).
Learning rate schedule: alpha_t = lr_const / t^lr_tau.
Coefficient: Polyak-Ruppert (iterate average) or last iterate.

Usage
-----
    model = BigSurvSGDCoxPH(strata_size=20, n_epochs=100)
    for chunk in chunks:
        model.partial_fit(chunk["time"], chunk["event"], chunk["X"])
    model.fit()
    print(model.coef_)          # Polyak-Ruppert averaged iterate
    print(model.coef_last_)     # last iterate
"""

import numpy as np

__all__ = ["BigSurvSGDCoxPH"]


class BigSurvSGDCoxPH:
    """
    Cox PH via mini-batch SGD with approximate within-stratum risk set.

    Parameters
    ----------
    strata_size : int
        Subjects per stratum (mini-batch). Default 20. Larger = less bias,
        more computation. Original paper uses 2-50.
    batch_size : int
        Strata to average per parameter update. Default 1.
    n_epochs : int
        Passes over the data. Default 100.
    opt_method : {"AMSGrad", "ADAM", "SGD"}
        Optimiser. Default "AMSGrad" (matches R package default).
    lr_const : float
        Learning-rate proportionality constant. Default 0.12.
    lr_tau : float
        Learning-rate decay exponent: lr_t = lr_const / t^lr_tau. Default 0.5.
    b1, b2, eps : float
        ADAM/AMSGrad hyperparameters. Defaults 0.9, 0.99, 1e-8.
    beta_type : {"averaged", "last"}
        "averaged": Polyak-Ruppert average (default, lower variance).
        "last": final iterate (faster convergence near optimum).
    standardize : bool
        Centre and scale X before fitting. Default True (matches R package).
    seed : int
        RNG seed for reproducibility.
    """

    def __init__(
        self,
        strata_size: int = 20,
        batch_size: int = 1,
        n_epochs: int = 100,
        opt_method: str = "AMSGrad",
        lr_const: float = 0.12,
        lr_tau: float = 0.5,
        b1: float = 0.9,
        b2: float = 0.99,
        eps: float = 1e-8,
        beta_type: str = "averaged",
        standardize: bool = True,
        seed: int = 42,
    ):
        if opt_method not in ("SGD", "ADAM", "AMSGrad"):
            raise ValueError("opt_method must be 'SGD', 'ADAM', or 'AMSGrad'")
        if beta_type not in ("averaged", "last"):
            raise ValueError("beta_type must be 'averaged' or 'last'")
        self.strata_size = strata_size
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.opt_method = opt_method
        self.lr_const = lr_const
        self.lr_tau = lr_tau
        self.b1 = b1
        self.b2 = b2
        self.eps = eps
        self.beta_type = beta_type
        self.standardize = standardize
        self.seed = seed
        self._chunks: list[tuple] = []

    def partial_fit(self, time, event, X):
        """Accumulate one data chunk. Returns self."""
        self._chunks.append((
            np.asarray(time, dtype=np.float64).ravel(),
            np.asarray(event, dtype=np.int32).ravel(),
            np.asarray(X, dtype=np.float64),
        ))
        return self

    def fit(self):
        """Run SGD to convergence. Sets coef_ and coef_last_."""
        t_all = np.concatenate([c[0] for c in self._chunks])
        e_all = np.concatenate([c[1] for c in self._chunks])
        X_all = np.vstack([c[2] for c in self._chunks])
        n, p = X_all.shape

        # Standardise (match R package default)
        if self.standardize:
            self._X_mean = X_all.mean(axis=0)
            self._X_std = X_all.std(axis=0)
            self._X_std[self._X_std == 0.0] = 1.0
            X_all = (X_all - self._X_mean) / self._X_std
        else:
            self._X_mean = np.zeros(p)
            self._X_std = np.ones(p)

        beta = np.zeros(p)
        beta_avg = np.zeros(p)
        m = np.zeros(p)
        v = np.zeros(p)
        v_hat = np.zeros(p)
        t_step = 0
        t_avg = 0

        S = self.strata_size
        B = self.batch_size
        rng = np.random.default_rng(self.seed)

        for _ in range(self.n_epochs):
            idx = rng.permutation(n)
            n_mini = n // (S * B)

            for i_mini in range(n_mini):
                t_step += 1
                lr = self.lr_const / (t_step ** self.lr_tau)
                gt = np.zeros(p)

                for i_batch in range(B):
                    start = (i_mini * B + i_batch) * S
                    si = idx[start: start + S]
                    t_s = t_all[si]
                    e_s = e_all[si]
                    X_s = X_all[si]

                    if e_s.sum() == 0:
                        continue

                    order = np.argsort(t_s, kind="stable")
                    t_s = t_s[order]
                    e_s = e_s[order]
                    X_s = X_s[order]

                    gt += self._stratum_gradient(t_s, e_s, X_s, beta)

                gt /= B

                # Parameter update
                if self.opt_method == "SGD":
                    beta = beta + lr * gt
                elif self.opt_method == "ADAM":
                    m = self.b1 * m + (1 - self.b1) * gt
                    v = self.b2 * v + (1 - self.b2) * gt ** 2
                    m_hat = m / (1 - self.b1 ** t_step)
                    v_hat = v / (1 - self.b2 ** t_step)
                    beta = beta + lr * m_hat / (np.sqrt(v_hat) + self.eps)
                else:  # AMSGrad
                    m = self.b1 * m + (1 - self.b1) * gt
                    v = self.b2 * v + (1 - self.b2) * gt ** 2
                    v_hat = np.maximum(v_hat, v)
                    beta = beta + lr * m / (np.sqrt(v_hat) + self.eps)

                # Polyak-Ruppert averaging
                t_avg += 1
                beta_avg = ((t_avg - 1) * beta_avg + beta) / t_avg

        self.coef_last_ = beta.copy()
        self.coef_ = beta_avg if self.beta_type == "averaged" else beta
        # Unstandardise
        if self.standardize:
            self.coef_ = self.coef_ / self._X_std
            self.coef_last_ = self.coef_last_ / self._X_std
        self.n_iter_ = t_step
        return self

    @staticmethod
    def _stratum_gradient(
        t_s: np.ndarray,
        e_s: np.ndarray,
        X_s: np.ndarray,
        beta: np.ndarray,
    ) -> np.ndarray:
        """Breslow partial-likelihood gradient on one stratum.

        The stratum is sorted by ascending time, so the risk set of row i is
        the rows from i to the end, and the sums run backwards from the end.
        (Until 2026-10-07 they ran forwards, over the rows that had already
        left the risk set; every result made with that version is void.)
        """
        del t_s  # the order carries the times
        exp_xb = np.exp(X_s @ beta)
        den = np.cumsum(exp_xb[::-1])[::-1]                          # (S,)
        num = np.cumsum((exp_xb[:, None] * X_s)[::-1], axis=0)[::-1]  # (S, p)
        mask = e_s == 1
        return (X_s[mask] - num[mask] / den[mask, None]).sum(axis=0)
