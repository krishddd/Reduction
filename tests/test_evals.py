from reduction import OptimizerConfig, TokenOptimizer
from reduction.evals import EvalCase, run_evals


def _model_reads_context(context: str, question: str) -> str:
    # A faithful model: answers correctly iff the needle survives in context.
    return "FAIL" if "FAIL" in context else "ok"


def test_preservation_when_compression_keeps_needle():
    # Needle is a log line; logcrush keeps ERROR/FAIL lines -> answer preserved.
    body = "\n".join(f"2026-01-01 INFO step {i} ok" for i in range(300))
    ctx = body + "\n2026-01-01 ERROR FAIL boom"
    cases = [EvalCase("nf", ctx, "fail?", lambda a: "FAIL" in a)]
    report = run_evals(cases, _model_reads_context)
    s = report.summary()
    assert s["raw_passes"] == 1
    assert s["compressed_passes"] == 1
    assert s["preservation_rate_pct"] == 100.0
    assert s["token_savings_pct"] > 0  # it did compress
    assert s["regressions"] == []


def test_detects_regression_when_compression_drops_needle():
    # Needle sits in the middle of a huge uniform JSON array -> sampled out,
    # so the (CCR-marked) compressed view no longer contains it inline.
    import json

    rows = [{"id": i, "status": "ok"} for i in range(300)]
    rows[150] = {"id": 150, "status": "FAIL"}
    ctx = json.dumps({"items": rows})
    cases = [EvalCase("jf", ctx, "fail?", lambda a: "FAIL" in a)]
    # Disable CCR marker noise; the point is the inline needle is gone.
    opt = TokenOptimizer(OptimizerConfig(ccr=False))
    report = run_evals(cases, _model_reads_context, optimizer=opt)
    s = report.summary()
    assert s["raw_passes"] == 1
    assert s["compressed_passes"] == 0
    assert s["preservation_rate_pct"] == 0.0
    assert "jf" in s["regressions"]


def test_render_mentions_preservation():
    cases = [EvalCase("x", "short ok text", "q", lambda a: True)]
    report = run_evals(cases, _model_reads_context)
    assert "Answer preservation" in report.render()
