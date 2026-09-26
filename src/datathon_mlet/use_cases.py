from datathon_mlet.policies import Policy
from integrations import RegistryClient

client = RegistryClient()


@client.trace(name="recommend_channel", span_type="CHAIN")
def recommend_channel(policy: Policy) -> str:
    """Recomenda o canal ótimo a oferecer para um cliente utilizando a política informada.

    Args:
        policy: Objeto que implementa o protocolo de política de decisão.

    Returns:
        Identificador do canal recomendado pela política.
    """
    return policy.recommend()
