from reduction import TokenOptimizer
from reduction.effort import route_effort


def test_routine_task_minimal_effort():
    d = route_effort("read the file config.py and show its contents")
    assert d.level == "minimal"
    assert d.thinking_budget == 0
    assert d.anthropic_thinking() is None  # thinking omitted


def test_analytical_task_high_effort():
    d = route_effort("debug why the deploy fails and find the root cause")
    assert d.level == "high"
    assert d.thinking_budget > 0
    assert d.anthropic_thinking() == {"type": "enabled", "budget_tokens": d.thinking_budget}


def test_complex_wins_over_routine():
    # contains both "show" (routine) and "analyze"/"why" (complex) -> complex wins
    d = route_effort("show me the logs and analyze why latency spiked")
    assert d.level == "high"


def test_neutral_task_uses_default():
    d = route_effort("write a haiku about the ocean")
    assert d.level == "medium"
    d2 = route_effort("write a haiku about the ocean", default="low")
    assert d2.level == "low"


def test_reasoning_effort_field_matches_level():
    for task in ("list files", "design the architecture", "say hello"):
        d = route_effort(task)
        assert d.reasoning_effort == d.level


def test_sdk_exposes_route_effort():
    opt = TokenOptimizer()
    d = opt.route_effort("grep for TODO markers")
    assert d.level == "minimal"
    assert d.rationale
