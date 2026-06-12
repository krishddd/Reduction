import pytest

from reduction import OptimizerConfig


def test_invalid_output_format_rejected():
    with pytest.raises(ValueError, match="output_format"):
        OptimizerConfig(output_format="xml")


def test_invalid_compression_rate_rejected():
    with pytest.raises(ValueError, match="compression_rate"):
        OptimizerConfig(compression_rate=0.0)
    with pytest.raises(ValueError, match="compression_rate"):
        OptimizerConfig(compression_rate=1.5)


def test_invalid_semantic_threshold_rejected():
    with pytest.raises(ValueError, match="semantic_threshold"):
        OptimizerConfig(semantic_threshold=1.2)


def test_valid_config_ok():
    cfg = OptimizerConfig(output_format="toon", compression_rate=0.5, semantic_threshold=0.92)
    assert cfg.output_format == "toon"


def test_with_overrides_preserves_validation():
    cfg = OptimizerConfig().with_overrides(output_format="yaml")
    assert cfg.output_format == "yaml"
    with pytest.raises(ValueError):
        OptimizerConfig().with_overrides(output_format="bogus")
