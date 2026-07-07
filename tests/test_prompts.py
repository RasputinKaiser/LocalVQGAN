from localvqgan.pipeline.prompts import parse_prompts


def test_single_prompt():
    p = parse_prompts("a sunset over the ocean")
    assert len(p) == 1
    assert p[0].text == "a sunset over the ocean"
    assert p[0].weight == 1.0


def test_pipe_and_weights():
    p = parse_prompts("a cat:1.5 | watercolor:0.5")
    assert [x.text for x in p] == ["a cat", "watercolor"]
    assert p[0].weight == 1.5 and p[1].weight == 0.5


def test_weight_and_stop():
    p = parse_prompts("dog:2:-0.5")
    assert p[0].weight == 2.0 and p[0].stop == -0.5


def test_empty_segments_skipped():
    assert len(parse_prompts("a | | b")) == 2
