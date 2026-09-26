"""Políticas de decisão: baseline determinístico e Thompson Sampling."""

from typing import Protocol

import numpy as np

from integrations import RegistryClient

client = RegistryClient()


class Policy(Protocol):
    """Contrato mínimo de uma política de bandit."""

    def select_action(self, rng: np.random.Generator) -> str:
        """Escolhe um braço."""

    def update(self, arm: str, reward: int) -> None:
        """Atualiza a política com a recompensa observada para um braço."""

    def recommend(self) -> str:
        """Decide de forma determinística o braço a recomendar a um cliente."""


class FixedPolicy:
    """Baseline determinístico: sempre escolhe o mesmo braço."""

    def __init__(self, arm: str) -> None:
        self.arm = arm

    def select_action(self, rng: np.random.Generator) -> str:
        return self.arm

    def update(self, arm: str, reward: int) -> None:
        return None

    def recommend(self) -> str:
        return self.arm


class ThompsonSamplingPolicy:
    """Algoritmo Thompson Sampling para o problema Multi-Armed Bandit.

    Mantém uma distribuição Beta(alpha, beta) por braço. A cada passo,
    amostra um valor de cada distribuição e seleciona o braço com maior
    amostra. Os parâmetros são atualizados com base nas recompensas observadas.

    Attributes:
        n_arms: Número de braços disponíveis.
        alpha: Vetor de parâmetros alpha (sucessos + 1) da distribuição Beta.
        beta: Vetor de parâmetros beta (falhas + 1) da distribuição Beta.
    """

    def __init__(
        self, arms: list[str], prior_alpha: float = 1.0, prior_beta: float = 1.0
    ) -> None:
        # Inicializa as distribuições Beta (priors uniformes alpha=1, beta=1)
        self.alpha = {arm: prior_alpha for arm in arms}
        self.beta = {arm: prior_beta for arm in arms}

    def select_action(self, rng: np.random.Generator) -> str:
        """Seleciona um braço amostrando da distribuição Beta de cada braço.

        Args:
            rng: Gerador de números aleatórios numpy.

        Returns:
            Nome do braço com maior amostra Beta.
        """
        samples = {arm: rng.beta(self.alpha[arm], self.beta[arm]) for arm in self.alpha}
        return max(samples, key=samples.get)

    def update(self, arm: str, reward: int) -> None:
        """Atualiza os parâmetros da distribuição Beta do braço selecionado.

        Args:
            arm: Índice do braço que foi puxado.
            reward: Recompensa observada (0 = falha, 1 = sucesso).
        """
        if reward:
            self.alpha[arm] += 1
        else:
            self.beta[arm] += 1

    @client.trace(name="recommend_action", span_type="MODEL")
    def recommend(self) -> str:
        """Recomenda um braço com base na média posterior da distribuição Beta.

        Registra no span ativo do MLflow as métricas internas (alpha, beta e
        médias calculadas) para fins de observabilidade do modelo.

        Returns:
            Nome do braço que possui a maior média a posteriori.
        """
        import mlflow

        posterior_means = {
            arm: self.alpha[arm] / (self.alpha[arm] + self.beta[arm])
            for arm in self.alpha
        }
        best_arm = max(posterior_means, key=posterior_means.get)

        span = mlflow.get_current_active_span()
        if span is not None:
            span.set_attributes(
                {
                    "alpha_params": self.alpha,
                    "beta_params": self.beta,
                    "posterior_means": posterior_means,
                    "selected_arm": best_arm,
                }
            )

        return best_arm
