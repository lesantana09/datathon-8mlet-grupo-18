"""Módulo de avaliação e visualização de experimentos de Multi-Armed Bandit."""

from collections.abc import Mapping
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

# Configura backend headless para execução em servidores e pipelines sem interface gráfica
matplotlib.use("Agg")


def plot_cumulative_rewards(
    baseline_rewards: list[int],
    ts_reward_histories: list[list[int]],
    output_path: Path,
) -> Path:
    """Gera e salva o gráfico da recompensa cumulativa ao longo das rodadas de replay.

    Compara a trajetória de recompensa do Baseline com a média e intervalo de
    desvio-padrão obtidos através das sementes de Thompson Sampling.

    Args:
        baseline_rewards: Histórico de recompensas binárias do baseline determinístico.
        ts_reward_histories: Lista de históricos de recompensas obtidos por cada seed.
        output_path: Caminho no disco para salvar a figura gerada.

    Returns:
        Path: Caminho do arquivo de imagem gerado.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    # Baseline cumulative curve
    if baseline_rewards:
        baseline_cumsum = np.cumsum(baseline_rewards)
        ax.plot(
            np.arange(1, len(baseline_cumsum) + 1),
            baseline_cumsum,
            label="Baseline (Fixed)",
            color="#d9534f",
            linewidth=2,
            linestyle="--",
        )

    # Thompson Sampling trajectories
    if ts_reward_histories:
        min_length = min(len(history) for history in ts_reward_histories)
        if min_length > 0:
            aligned_cumsums = np.array(
                [np.cumsum(history[:min_length]) for history in ts_reward_histories]
            )
            rounds = np.arange(1, min_length + 1)
            mean_cumsum = aligned_cumsums.mean(axis=0)
            std_cumsum = aligned_cumsums.std(axis=0)

            ax.plot(
                rounds,
                mean_cumsum,
                label=f"Thompson Sampling (Mean across {len(ts_reward_histories)} seeds)",
                color="#0275d8",
                linewidth=2.5,
            )
            ax.fill_between(
                rounds,
                mean_cumsum - std_cumsum,
                mean_cumsum + std_cumsum,
                color="#0275d8",
                alpha=0.2,
                label="TS ± 1 std dev",
            )

    ax.set_title(
        "Recompensa Cumulativa: Baseline vs Thompson Sampling",
        fontsize=14,
        fontweight="bold",
    )
    ax.set_xlabel("Rodadas Replay Aproveitadas (Rounds Used)", fontsize=12)
    ax.set_ylabel("Recompensa Cumulativa (Conversões)", fontsize=12)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="upper left", fontsize=11)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return output_path


def plot_posterior_distributions(
    alpha_params: Mapping[str, float],
    beta_params: Mapping[str, float],
    output_path: Path,
) -> Path:
    """Gera e salva o gráfico das funções de densidade de probabilidade (PDF) Beta posteriores.

    Permite visualizar a estimativa da taxa de conversão e o grau de incerteza
    associado a cada braço ao final do aprendizado do bandit.

    Args:
        alpha_params: Dicionário contendo os valores de alpha para cada braço.
        beta_params: Dicionário contendo os valores de beta para cada braço.
        output_path: Caminho no disco para salvar a figura gerada.

    Returns:
        Path: Caminho do arquivo de imagem gerado.
    """
    from scipy.stats import beta as beta_dist

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.linspace(0.0, 0.35, 1000)

    palette = ["#0275d8", "#5cb85c", "#f0ad4e", "#d9534f", "#6f42c1"]

    for idx, (arm, alpha_val) in enumerate(alpha_params.items()):
        beta_val = beta_params.get(arm, 1.0)
        pdf_values = beta_dist.pdf(x, alpha_val, beta_val)
        mean_val = alpha_val / (alpha_val + beta_val)
        color = palette[idx % len(palette)]

        ax.plot(
            x,
            pdf_values,
            label=f"{arm} (Mean: {mean_val:.3f}, α={alpha_val:.0f}, β={beta_val:.0f})",
            color=color,
            linewidth=2,
        )
        ax.axvline(mean_val, color=color, linestyle=":", alpha=0.7)

    ax.set_title(
        "Distribuição a Posteriori por Canal (Thompson Sampling)",
        fontsize=14,
        fontweight="bold",
    )
    ax.set_xlabel("Taxa de Conversão Esperada (θ)", fontsize=12)
    ax.set_ylabel("Densidade de Probabilidade", fontsize=12)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="upper right", fontsize=11)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return output_path
