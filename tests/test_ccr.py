from reduction import ccr
from reduction.ccr import CompressionStore


def test_store_roundtrip_and_idempotent():
    store = CompressionStore()
    original = "x" * 5000
    ref1 = store.put(original)
    ref2 = store.put(original)
    assert ref1 == ref2  # content-addressed
    assert len(ref1) == 8
    assert store.get(ref1) == original
    assert len(store) == 1


def test_store_persists_across_instances(tmp_path):
    path = tmp_path / "ccr.json"
    s1 = CompressionStore(path=path)
    ref = s1.put("hello world original content")
    s2 = CompressionStore(path=path)  # fresh instance reads the file
    assert s2.get(ref) == "hello world original content"


def test_get_accepts_ref_prefix():
    store = CompressionStore()
    ref = store.put("data")
    assert store.get(f"ref={ref}") == "data"


def test_marker_format():
    marker = ccr.make_marker("json compressed 80%", "ab12cd34")
    assert marker == "[reduction: json compressed 80%, ref=ab12cd34]"


def test_retrieve_tool_definitions():
    anth = ccr.retrieve_tool_definition("anthropic")
    assert anth["name"] == "reduction_retrieve"
    assert "input_schema" in anth
    oai = ccr.retrieve_tool_definition("openai")
    assert oai["function"]["name"] == "reduction_retrieve"


def test_inject_retrieve_tool_no_duplicates():
    tools = ccr.inject_retrieve_tool(None, "anthropic")
    assert len(tools) == 1
    tools2 = ccr.inject_retrieve_tool(tools, "anthropic")
    assert len(tools2) == 1  # already present


def test_handle_retrieve_call():
    store = CompressionStore()
    ref = store.put("the original")
    got = ccr.handle_retrieve_call("reduction_retrieve", {"ref": ref}, store=store)
    assert got == "the original"
    assert ccr.handle_retrieve_call("other_tool", {"ref": ref}, store=store) is None
    missing = ccr.handle_retrieve_call("reduction_retrieve", {"ref": "deadbeef"}, store=store)
    assert "not found" in missing
