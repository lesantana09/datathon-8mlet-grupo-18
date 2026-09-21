"""Instrumentação de MLflow para os experimentos de bandit da Etapa 3.

Assume que `mlflow.set_tracking_uri` já foi configurado pelo chamador
(notebook ou teste) — esta função só usa o tracking ativo no momento.
"""

import tempfile
from dataclasses import dataclass
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd

from datathon_mlet.evaluation import (
    plot_cumulative_rewards,
    plot_posterior_distributions,
)
from datathon_mlet.policies import FixedPolicy, ThompsonSamplingPolicy
from datathon_mlet.replay import ReplayResult, run_replay
from infrastructure import RegistryClient

client = RegistryClient()

mlflow.autolog()


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
    seeds), artefatos gráficos (curva de convergência e distribuições posteriores)
    e um run aninhado por seed — preserva a rastreabilidade
    individual exigida pela regra de reportar variabilidade em simulações
    estocásticas. Retorna os resultados pra evitar recomputar o replay só
    pra plotar ou analisar no notebook chamador.
    """
    client.set_experiment(experiment_name)

    with client.start_run(run_name="baseline"):
        client.log_param("policy", "fixed")
        client.log_param("arm", baseline_arm)
        baseline_result = run_replay(
            action, reward, FixedPolicy(arm=baseline_arm), seed=0
        )
        client.log_metric("rounds_used", baseline_result.rounds_used)
        client.log_metric("cumulative_reward", baseline_result.cumulative_reward)
        client.log_metric("conversion_rate", baseline_result.conversion_rate)

    prior_policy = ThompsonSamplingPolicy(arms=arms)
    prior_alpha = prior_policy.alpha[arms[0]]
    prior_beta = prior_policy.beta[arms[0]]

    with client.start_run(run_name="thompson_sampling"):
        client.log_param("policy", "thompson_sampling")
        client.log_param("arms", ",".join(arms))
        client.log_param("prior_alpha", prior_alpha)
        client.log_param("prior_beta", prior_beta)
        client.log_param("n_seeds", n_seeds)

        conversion_rates: list[float] = []
        reward_histories: list[list[int]] = []
        last_policy: ThompsonSamplingPolicy | None = None

        for seed in range(n_seeds):
            with client.start_run(run_name=f"seed_{seed}", nested=True):
                client.log_param("seed", seed)
                policy = ThompsonSamplingPolicy(arms=arms)
                result = run_replay(action, reward, policy, seed=seed)
                client.log_metric("rounds_used", result.rounds_used)
                client.log_metric("conversion_rate", result.conversion_rate)
                conversion_rates.append(result.conversion_rate)
                reward_histories.append(result.reward_history)
                last_policy = policy

        rates = np.array(conversion_rates)
        client.log_metric("conversion_rate_mean", float(rates.mean()))
        client.log_metric("conversion_rate_std", float(rates.std()))

        # Geração e logging de artefatos visuais de avaliação
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)

            # 1. Curvas de recompensa cumulativa (Baseline vs Thompson Sampling)
            curves_path = tmp_path / "learning_curves.png"
            plot_cumulative_rewards(
                baseline_rewards=baseline_result.reward_history,
                ts_reward_histories=reward_histories,
                output_path=curves_path,
            )
            client.log_artifact(str(curves_path))

            # 2. Distribuições Beta posteriores dos braços
            if last_policy is not None:
                distributions_path = tmp_path / "posterior_distributions.png"
                plot_posterior_distributions(
                    alpha_params=last_policy.alpha,
                    beta_params=last_policy.beta,
                    output_path=distributions_path,
                )
                client.log_artifact(str(distributions_path))

    return ExperimentResults(baseline=baseline_result, ts_conversion_rates=rates)
