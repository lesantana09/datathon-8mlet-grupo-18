"""Instrumentação de MLflow para os experimentos de bandit da Etapa 3.

Assume que `mlflow.set_tracking_uri` já foi configurado pelo chamador
(notebook ou teste) — esta função só usa o tracking ativo no momento.
"""

from dataclasses import dataclass

import mlflow
import numpy as np
import pandas as pd

from datathon_mlet.policies import FixedPolicy, ThompsonSamplingPolicy
from datathon_mlet.replay import ReplayResult, run_replay


@dataclass(frozen=True)
class ExperimentResults:
    """Resultado do experimento, para o chamador reusar (ex. plotar)."""

    baseline: ReplayResult
    ts_conversion_rates: np.ndarray


def log_baseline_vs_thompson_sampling(
    action: pd.Series,
    reward: pd.Series,
    arms: list[str],
    baseline_arm: str,
    n_seeds: int,
    experiment_name: str = "baseline_vs_thompson_sampling",
) -> ExperimentResults:
    """Roda baseline (1x) e Thompson Sampling (`n_seeds` seeds) via replay.

    Loga um run para o baseline e, para o Thompson Sampling, um run pai com
    as métricas agregadas (média/desvio-padrão de `conversion_rate` entre
    seeds) e um run aninhado por seed — preserva a rastreabilidade
    individual exigida pela regra de reportar variabilidade em simulações
    estocásticas. Retorna os resultados pra evitar recomputar o replay só
    pra plotar ou analisar no notebook chamador.
    """
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name="baseline"):
        mlflow.log_param("policy", "fixed")
        mlflow.log_param("arm", baseline_arm)
        baseline_result = run_replay(
            action, reward, FixedPolicy(arm=baseline_arm), seed=0
        )
        mlflow.log_metric("rounds_used", baseline_result.rounds_used)
        mlflow.log_metric("cumulative_reward", baseline_result.cumulative_reward)
        mlflow.log_metric("conversion_rate", baseline_result.conversion_rate)

    prior_policy = ThompsonSamplingPolicy(arms=arms)
    prior_alpha = prior_policy.alpha[arms[0]]
    prior_beta = prior_policy.beta[arms[0]]

    with mlflow.start_run(run_name="thompson_sampling"):
        mlflow.log_param("policy", "thompson_sampling")
        mlflow.log_param("arms", ",".join(arms))
        mlflow.log_param("prior_alpha", prior_alpha)
        mlflow.log_param("prior_beta", prior_beta)
        mlflow.log_param("n_seeds", n_seeds)

        conversion_rates = []
        for seed in range(n_seeds):
            with mlflow.start_run(run_name=f"seed_{seed}", nested=True):
                mlflow.log_param("seed", seed)
                policy = ThompsonSamplingPolicy(arms=arms)
                result = run_replay(action, reward, policy, seed=seed)
                mlflow.log_metric("rounds_used", result.rounds_used)
                mlflow.log_metric("conversion_rate", result.conversion_rate)
                conversion_rates.append(result.conversion_rate)

        rates = np.array(conversion_rates)
        mlflow.log_metric("conversion_rate_mean", rates.mean())
        mlflow.log_metric("conversion_rate_std", rates.std())

    return ExperimentResults(baseline=baseline_result, ts_conversion_rates=rates)
