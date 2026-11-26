from care_rag.llm import LLM, Provider, parse_json, retry_wait


def test_retry_wait():
    assert retry_wait("429 You exceeded your current quota, check plan", 0) is None
    assert retry_wait("Rate limit reached. Please try again in 12.5s.", 0) == 13.5
    assert retry_wait("Rate limit reached. Please try again in 1m2s.", 0) == 63.0
    assert retry_wait("Rate limit reached. Please try again in 9m0s.", 0) is None
    assert retry_wait("503 UNAVAILABLE high demand", 1) == 30.0
    assert retry_wait("400 invalid request", 0) is None


class Fake(Provider):
    name = "fake"

    def __init__(self, model, fail=False):
        super().__init__(model)
        self.fail, self.calls = fail, 0

    def available(self):
        return True

    def generate(self, system, prompt, json_mode):
        self.calls += 1
        if self.fail:
            raise RuntimeError("400 bad request")
        return f"ok from {self.model}"


def test_fallback_to_next_provider():
    a, b = Fake("a", fail=True), Fake("b")
    resp = LLM([a, b], use_cache=False).generate("s", "p")
    assert resp.model == "b" and a.calls == 1  # non-retryable error: no retry, fall through


def test_parse_json_with_fences():
    assert parse_json('```json\n{"a": 1}\n```') == {"a": 1}
    assert parse_json('Here: {"claims": []} done') == {"claims": []}
