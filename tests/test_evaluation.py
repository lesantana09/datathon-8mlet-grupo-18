"""Testes unitários para o módulo de avaliação e geração de artefatos visuais."""

from pathlib import Path

from datathon_mlet.evaluation import (
    plot_cumulative_rewards,
    plot_posterior_distributions,
)


def test_plot_cumulative_rewards_generates_image_file(tmp_path: Path) -> None:
    """Valida se o gráfico de recompensa cumulativa é salvo no caminho especificado."""
    output_image = tmp_path / "test_curves.png"
    baseline_rewards = [0, 1, 0, 1, 0]
    ts_histories = [
        [0, 1, 1, 1, 0],
        [1, 1, 0, 1, 1],
    ]

    result_path = plot_cumulative_rewards(
        baseline_rewards=baseline_rewards,
        ts_reward_histories=ts_histories,
        output_path=output_image,
    )

    assert result_path.exists()
    assert result_path.stat().st_size > 0


def test_plot_posterior_distributions_generates_image_file(tmp_path: Path) -> None:
    """Valida se o gráfico de densidades Beta a posteriori é salvo com sucesso."""
    output_image = tmp_path / "test_posterior.png"
    alpha_params = {"cellular": 10.0, "telephone": 5.0}
    beta_params = {"cellular": 90.0, "telephone": 95.0}

    result_path = plot_posterior_distributions(
        alpha_params=alpha_params,
        beta_params=beta_params,
        output_path=output_image,
    )

    assert result_path.exists()
    assert result_path.stat().st_size > 0
