from dataclasses import dataclass


@dataclass
class ParsedPrompt:
    text: str
    weight: float = 1.0
    stop: float = float("-inf")


def parse_prompts(s: str) -> list[ParsedPrompt]:
    out = []
    for chunk in s.split("|"):
        chunk = chunk.strip()
        if not chunk:
            continue
        vals = chunk.rsplit(":", 2)
        # A trailing :number is a weight; two are weight:stop. Bare colons
        # inside text (no numeric suffix) stay part of the text.
        text, weight, stop = vals[0], 1.0, float("-inf")
        try:
            if len(vals) == 3:
                weight, stop = float(vals[1]), float(vals[2])
            elif len(vals) == 2:
                weight = float(vals[1])
        except ValueError:
            text = chunk
        out.append(ParsedPrompt(text.strip(), weight, stop))
    return out
