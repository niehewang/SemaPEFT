from semapeft.source_extract import extract_class, extract_func


def test_ast_extract(tmp_path):
    p = tmp_path / "x.py"
    p.write_text("@decorator\nclass A:\n    pass\n\ndef f():\n    return 1\n", encoding="utf-8")
    assert "class A" in extract_class(p, "A")
    assert "def f" in extract_func(p, "f")
