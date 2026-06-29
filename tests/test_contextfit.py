import json

from reduction import TokenOptimizer
from reduction.ccr import CompressionStore
from reduction.layers import contextfit
from reduction.metrics import estimate_tokens

BIG_JSON = json.dumps([{"id": i, "host": f"h-{i}", "open": True} for i in range(300)])


def test_includes_everything_when_budget_ample():
    chunks = ["alpha context", "beta context", "gamma context"]
    result = contextfit.fit_context(chunks, token_budget=10_000)
    assert result.included == 3
    assert result.dropped == 0
    assert result.chunks == chunks  # original order, verbatim


def test_drops_lowest_priority_when_over_budget():
    chunks = ["a " * 100, "b " * 100, "c " * 100]
    # budget fits ~2 of the 3 small chunks
    budget = estimate_tokens(chunks[0]) * 2 + 5
    result = contextfit.fit_context(chunks, token_budget=budget)
    assert result.tokens_used <= budget
    assert result.dropped >= 1
    assert result.included + result.dropped == 3


def test_query_relevance_prioritizes_matching_chunk():
    chunks = [
        "the quick brown fox jumps " * 20,  # irrelevant, large
        "deployment uses AWS_PROFILE=prod and region us-east-1",  # relevant, small
    ]
    # budget too small for the big irrelevant chunk; the relevant one must win
    result = contextfit.fit_context(
        chunks, token_budget=40, query="how do I deploy with which AWS profile"
    )
    joined = "\n".join(result.chunks)
    assert "AWS_PROFILE=prod" in joined


def test_compresses_to_fit():
    store = CompressionStore()
    chunks = ["small note", BIG_JSON]
    budget = estimate_tokens("small note") + 100  # forces compression of BIG_JSON
    result = contextfit.fit_context(chunks, token_budget=budget, store=store)
    assert result.compressed >= 1
    assert result.tokens_used <= budget
    assert result.refs
    assert store.get(result.refs[0]) == BIG_JSON


def test_truncates_with_reversible_marker():
    store = CompressionStore()
    text = "word " * 4000  # plain text won't content-compress much -> truncation path
    result = contextfit.fit_context([text], token_budget=60, ccr=True, store=store)
    assert result.tokens_used <= 60
    out = result.chunks[0]
    assert "reduction:" in out  # CCR marker present
    assert result.refs and store.get(result.refs[0]) == text


def test_zero_budget_drops_all():
    result = contextfit.fit_context(["a", "b"], token_budget=0)
    assert result.included == 0
    assert result.dropped == 2


def test_sdk_fit_context_records_metrics():
    opt = TokenOptimizer()
    budget = estimate_tokens("small note") + 100
    out = opt.fit_context(["small note", BIG_JSON], token_budget=budget)
    assert isinstance(out, list)
    assert opt.report()["input_tokens_saved"] > 0
