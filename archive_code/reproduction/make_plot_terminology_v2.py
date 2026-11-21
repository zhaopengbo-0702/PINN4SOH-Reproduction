"""Create an exact string-literal terminology patch; never alter plot logic."""

import ast
import difflib
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "reproduction/plot_full_paper_results.py"
TARGET = ROOT / "reproduction/plot_full_paper_results_v2.py"
OUT = ROOT / "outputs/full_reproduction/plotter_terminology_v2_audit"

REPLACEMENTS = {
    "Exact-battery macro metrics; measured-SOH MAPE denominator. Mean +/- sample SD (ddof=1).\\n":
    "CSV-file macro; measured-SOH MAPE denominator. Mean +/- sample SD (ddof=1).\\n",
    "n=10 seeds per complete configuration; p = pending. Shared MLP/CNN. Lower is better.":
    "n=10 seeds; shared MLP/CNN; p=pending. MIT mixes cell identities in CSV5. Lower is better.",
    "Mean +/- sample SD (ddof=1); exact-battery macro RMSE. n=10 seeds per complete point.\\n":
    "Mean +/- sample SD (ddof=1); CSV-file macro RMSE. n=10 seeds per complete point.\\n",
    "Fixed test batteries; shared MLP/CNN; p = pending. Lower is better.":
    "Fixed test CSV files; shared MLP/CNN; p = pending. Lower is better.",
    "All available held-out batteries in each group.": "All available held-out CSV files in each group.",
    "Uniform display sampling (up to 256 points/battery)": "Uniform display sampling (up to 256 points/CSV)",
    'colLabels=["Group / battery"]': 'colLabels=["Group / CSV file"]',
    "Shared MLP/CNN; exact-battery macro metrics and measured-SOH MAPE denominator.":
    "CSV-file macro; measured-SOH MAPE. MIT CSV5 mixes cell identities; no physical-cell macro claim.",
    "Per-battery mean (sample SD, ddof=1), n=10; SOH fractions. Exact battery/date matching.\\n":
    "Per-CSV mean (sample SD, ddof=1), n=10; SOH fractions. Exact file/date matching.\\n",
    "Pending cells retain missing, ambiguous or incomplete groups.":
    "MIT CSV5 mixes cell identities. Pending = missing, ambiguous or incomplete groups.",
    "Signed per-battery metric differences in SOH fractions; no rounded identity guessing.":
    "Signed per-CSV metric differences in SOH fractions; no rounded identity guessing.",
    "真实电池等权宏平均": "发布 CSV 文件等权宏平均（CSV-file macro）",
    "MAPE 分母为实测 SOH。":
    "MAPE 分母为实测 SOH。MIT 的 CSV5 含两个物理身份，不能将该文件宏平均称为真实物理电池等权平均。",
}


def without_string_values(source):
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            node.value = "STRING_LITERAL"
    return ast.dump(tree, include_attributes=False)


def main():
    original = SOURCE.read_text(encoding="utf-8")
    updated = original
    for before, after in REPLACEMENTS.items():
        count = updated.count(before)
        if count != 1:
            raise ValueError("Expected one exact terminology occurrence: " + repr(before) + " found " + str(count))
        updated = updated.replace(before, after)
    if without_string_values(original) != without_string_values(updated):
        raise ValueError("A non-string AST node changed")
    with TARGET.open("w", encoding="utf-8", newline="") as stream:
        stream.write(updated)
    OUT.mkdir(parents=True, exist_ok=True)
    diff = "".join(difflib.unified_diff(original.splitlines(keepends=True), updated.splitlines(keepends=True),
        fromfile="plot_full_paper_results.py", tofile="plot_full_paper_results_v2.py"))
    (OUT / "terminology_only.diff").write_text(diff, encoding="utf-8")
    audit = {"original_file": str(SOURCE), "new_file": str(TARGET),
        "original_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        "v2_sha256": hashlib.sha256(TARGET.read_bytes()).hexdigest(),
        "exact_replacements": len(REPLACEMENTS), "all_non_string_AST_nodes_identical": True,
        "changes": "Display/caption strings only: CSV-file unit, MIT mixed identity, file/date matching. Numeric and selection logic unchanged.",
        "original_source_modified": False}
    (OUT / "terminology_only_audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(audit, ensure_ascii=False))


if __name__ == "__main__":
    main()
