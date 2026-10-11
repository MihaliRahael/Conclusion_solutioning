from datetime import date

import pytest

from agents import build_graph
from services import DataStore, PolicyIndex, RefundRules, Services


class DownLLM:
    """Stands in for an unreachable Ollama server."""

    model = "down"

    def chat(self, *a, **k):
        from services import LLMError

        raise LLMError("connection refused")


@pytest.fixture(scope="session")
def store():
    return DataStore(mask_pii=True)


@pytest.fixture(scope="session")
def policy():
    return PolicyIndex(embeddings=False)  # keyword retrieval: no model download in tests


@pytest.fixture(scope="session")
def rules(policy):
    return RefundRules.from_policy(policy.full_text)


@pytest.fixture
def svc(store, policy, rules):
    # Orders are dated 2026-07-01..10; this reference date puts them inside the 30-day window.
    return Services(store, policy, rules, llm=None, today=date(2026, 7, 20))


@pytest.fixture
def graph(svc):
    return build_graph(svc)
