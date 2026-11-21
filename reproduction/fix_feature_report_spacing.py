"""Repair typography while retaining exact inline code, links and commands."""
from pathlib import Path
import re

path = Path("outputs/full_reproduction/feature_audit/FEATURE_AUDIT.md")
text = path.read_text(encoding="utf-8")

def compact(value):
    previous = None
    while value != previous:
        previous = value
        value = re.sub(r"([A-Za-z]) (\d)", r"\1\2", value)
        value = re.sub(r"(\d) ([A-Za-z])", r"\1\2", value)
    return value

text = re.sub(r"`[^`]*`", lambda match: compact(match.group()), text)
text = re.sub(r"<https?[^>]+>", lambda match: compact(match.group()), text)
text = re.sub(r"(\d) e([+-]\d)", r"\1e\2", text)
for before, after in (("PINN 4 SOH", "PINN4SOH"), ("Fig 1 b", "Fig1b"), ("Fig 3 b", "Fig3b"),
                      ("Fig 3 a", "Fig3a"), ("FigS 4", "FigS4"), ("SHA 256", "SHA256"),
                      ("SHA 1", "SHA1"), ("MD 5", "MD5"), ("utf 8", "utf8")):
    text = text.replace(before, after)
text = text.replace("utf8reproduction/", "utf8 reproduction/")
for before, after in (("all262", "all 262"), ("132primary", "132 primary"),
                      ("262primary", "262 primary"), ("132boundary", "132 boundary"),
                      ("100%finite", "100% finite"), ("statedTJU", "stated TJU"),
                      ("public-readerHUST", "public-reader HUST"), ("exploratoryXJTU", "exploratory XJTU"),
                      ("sixdataset", "six dataset"), ("sixunit", "six unit"), ("sixfrozen", "six frozen"),
                      ("S 4", "S4"), ("physicalseconds", "physical seconds"),
                      ("sampleindex", "sample index"), ("normalizedtime", "normalized time"),
                      ("min-max physicaltime", "min-max physical time"), ("16 feature", "16-feature")):
    text = text.replace(before, after)
path.write_text(text, encoding="utf-8")
