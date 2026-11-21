"""Read-only satellite protocol/S1/S2/S3 supplement from the verified XJTU archive.

No training, data replacement, slope search, phase fitting, or physical aging.
The published 46-cycle schedule is transcribed before reading raw values.
"""

import argparse
import contextlib
import csv
from datetime import datetime, timezone
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import zipfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.text import Text
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
FIRST_HALF = [5, 20, 34, 41, 46, 50, 54, 56, 58, 60, 62, 64, 68, 69, 70, 71, 72, 72, 72, 72, 72, 72, 72]
SCHEDULE = FIRST_HALF + FIRST_HALF[::-1]
STYLE = Path("C:/Users/Administrator/.codex/skills/figure-style/kernel.py")


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def write_csv(path, rows):
    with Path(path).open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, list(dict.fromkeys(key for row in rows for key in row)))
        writer.writeheader()
        writer.writerows(rows)


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def finite(value):
    return float(value) if np.isfinite(value) else None


def render_check(fig, stem):
    fig.canvas.draw()
    for ax in fig.axes:
        for tick in ax.xaxis.get_major_ticks() + ax.xaxis.get_minor_ticks():
            if not min(ax.get_xlim()) <= tick.get_loc() <= max(ax.get_xlim()):
                tick.label1.set_visible(False)
                tick.label2.set_visible(False)
        for tick in ax.yaxis.get_major_ticks() + ax.yaxis.get_minor_ticks():
            if not min(ax.get_ylim()) <= tick.get_loc() <= max(ax.get_ylim()):
                tick.label1.set_visible(False)
                tick.label2.set_visible(False)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    texts = [(t, t.get_window_extent(renderer)) for t in fig.findobj(Text)
             if t.get_text().strip() and t.get_visible()]
    collisions = [[a.get_text(), b.get_text()] for i, (a, aa) in enumerate(texts)
                  for b, bb in texts[i + 1:] if aa.overlaps(bb)]
    outside = [t.get_text() for t, b in texts
               if b.x0 < -1 or b.y0 < -1 or b.x1 > fig.bbox.x1 + 1 or b.y1 > fig.bbox.y1 + 1]
    spines = [(s, s.get_window_extent(renderer)) for ax in fig.axes for s in ax.spines.values() if s.get_visible()]
    spine_hits = []
    for text, box in texts:
        for spine, spine_box in spines:
            ticks = set(spine.axes.get_xticklabels(which="both") + spine.axes.get_yticklabels(which="both"))
            if text not in ticks and box.overlaps(spine_box):
                spine_hits.append([text.get_text(), str(spine.spine_type)])
    qa = dict(text_collisions=collisions, text_outside_figure=outside, text_spine_collisions=spine_hits)
    write_json(stem.with_suffix(".layout.json"), qa)
    if collisions or outside or spine_hits:
        raise ValueError("Layout check failed: " + repr(qa))
    fig.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches=None)
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches=None)
    plt.close(fig)
    return qa


def extract(options):
    output = options.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError("Use an empty/new supplement output; never overwrite an existing audit")
    output.mkdir(parents=True, exist_ok=True)
    (output / "supplement_satellite_protocol_source.py").write_bytes(Path(__file__).read_bytes())
    audit = ROOT / "outputs/full_reproduction/feature_audit"
    archive_path = ROOT / "outputs/full_reproduction/raw_data/XJTU/Battery Dataset.zip"
    archive_status = read_json(archive_path.with_suffix(".status.json"))
    if archive_status.get("status") != "verified" or archive_status.get("official_checksum_passed") is not True:
        raise ValueError("Official archive has not passed verification")
    reader_path = audit / "preprocessing_source/XJTUBatteryClass.py"
    reader = load_module(reader_path, "satellite_official_reader")
    schedule_rows = [dict(protocol_cycle=i + 1, published_discharge_duration_min=value,
                          origin="Supplementary Information PDF p4 TableS1; intended schedule, not new measurement")
                     for i, value in enumerate(SCHEDULE)]
    write_csv(output / "TableS1_published_schedule_46cycles.csv", schedule_rows)
    rows, identities = [], []
    for battery_number in options.batteries:
        member = "Battery Dataset/Batch-6/Sim_satellite_battery-" + str(battery_number) + ".mat"
        with zipfile.ZipFile(archive_path) as archive:
            info = archive.getinfo(member)
            payload = archive.read(member)
        member_sha = hashlib.sha256(payload).hexdigest()
        with tempfile.TemporaryDirectory(prefix="pinn-satellite-") as directory:
            path = Path(directory) / Path(member).name
            path.write_bytes(payload)
            with contextlib.redirect_stdout(io.StringIO()):
                battery = reader.Battery(path.as_posix())
        del payload
        capacities = battery.get_capacity()
        identity = dict(battery=battery_number, member=member, member_crc32=info.CRC,
                        member_bytes=info.file_size, member_sha256=member_sha,
                        physical_cycles=int(battery.cycle_life))
        identities.append(identity)
        regular = 0
        for cycle in range(1, int(battery.cycle_life) + 1):
            description = battery.get_one_cycle_description(cycle)
            is_test = "test capacity" in description
            if not is_test:
                regular += 1
            t = battery.get_value(cycle, "relative_time_min")
            current = battery.get_value(cycle, "current_A")
            resets = np.flatnonzero(t == 0)
            row = dict(battery=battery_number, physical_cycle=cycle, description=description,
                       periodic_full_capacity_test=is_test, measured_discharge_capacity_Ah=float(capacities[cycle - 1]),
                       raw_samples=len(t), relative_time_zero_count=len(resets),
                       regular_cycle_index=None if is_test else regular,
                       prespecified_protocol_cycle=None if is_test else (regular - 1) % 46 + 1,
                       published_duration_min=None if is_test else SCHEDULE[(regular - 1) % 46],
                       phase_policy="Count non-test cycles from the first non-test record; no fitted phase or omitted cycles")
            negative_indices = np.flatnonzero(current < -.1)
            runs = []
            if len(negative_indices):
                split = np.flatnonzero((np.diff(negative_indices) != 1) | (np.diff(t[negative_indices]) < 0)) + 1
                for indices in np.split(negative_indices, split):
                    if len(indices):
                        runs.append(dict(start_raw_index=int(indices[0]), end_raw_index=int(indices[-1]),
                                         n=len(indices), start_min=float(t[indices[0]]), end_min=float(t[indices[-1]]),
                                         recorded_span_min=float(t[indices[-1]] - t[indices[0]]),
                                         current_median_A=float(np.median(current[indices]))))
            longest = max(runs, key=lambda record: record["recorded_span_min"]) if runs else None
            row.update(full_cycle_negative_current_gate_A=-.1, negative_current_run_count=len(runs),
                       negative_current_runs_json=json.dumps(runs, separators=(",", ":")),
                       negative_run_span_sum_min=sum(r["recorded_span_min"] for r in runs),
                       longest_negative_run_span_min=longest["recorded_span_min"] if longest else None,
                       longest_negative_run_samples=longest["n"] if longest else 0,
                       longest_negative_run_current_median_A=longest["current_median_A"] if longest else None)
            if not is_test and longest:
                row["longest_negative_run_minus_published_min"] = longest["recorded_span_min"] - row["published_duration_min"]
            try:
                tt = battery.get_partial_value(cycle, "relative_time_min", stage=3)
                ii = battery.get_partial_value(cycle, "current_A", stage=3)
                negative = ii < -.1
                row.update(stage3_samples=len(tt), stage3_negative_current_samples=int(negative.sum()),
                           stage3_start_min=finite(tt[0]) if len(tt) else None,
                           stage3_end_min=finite(tt[-1]) if len(tt) else None,
                           stage3_recorded_span_min=finite(tt[-1] - tt[0]) if len(tt) else None,
                           stage3_current_median_A=finite(np.median(ii)) if len(ii) else None,
                           stage3_negative_current_span_min=finite(tt[negative][-1] - tt[negative][0]) if negative.any() else None,
                           stage3_nondecreasing_time=bool(np.all(np.diff(tt) >= 0)), stage3_status="computed")
                if not is_test and row["stage3_recorded_span_min"] is not None:
                    row["span_minus_published_min"] = row["stage3_recorded_span_min"] - row["published_duration_min"]
            except (IndexError, ValueError) as error:
                row.update(stage3_status="unavailable", stage3_error=repr(error))
            rows.append(row)
        print(json.dumps(dict(battery=battery_number, raw_cycles=int(battery.cycle_life), regular_cycles=regular,
                              full_capacity_test_cycles=int(battery.cycle_life) - regular)), flush=True)
        del battery
    write_csv(output / "satellite_all_raw_cycle_capacity_and_stage3_duration.csv", rows)
    manifest = dict(created_utc=datetime.now(timezone.utc).isoformat(), status="raw_extraction_complete",
                    identities=identities, cycles=len(rows), schedule_cycles=46,
                    original_archive_sha256_saved=archive_status["sha256"],
                    archive_checksum_scope="Existing official archive verification plus this run ZIP member CRC and member SHA; full archive not rehashed",
                    official_checksum=archive_status["official_checksum"],
                    source_reader_sha256=digest(reader_path), script_sha256=digest(__file__),
                    supplement_pdf_sha256=digest(ROOT / "outputs/full_reproduction/supplementary_information.pdf"),
                    no_training=True, no_phase_fit=True, no_paper_result_changes=True,
                    boundaries=["Published schedule is an intended design", "Recorded endpoint span excludes unrecorded edge intervals",
                                "Periodic full-capacity tests are separate records", "No apparatus or physical aging repeated"])
    write_json(output / "SATELLITE_SUPPLEMENT.json", manifest)
    return rows, manifest


def plot(options, rows, manifest):
    output = options.output.resolve()
    style = load_module(STYLE, "satellite_figure_style")
    style.apply_figure_style(sizes=(10, 9, 8), font="Arial")
    plt.rcParams["savefig.bbox"] = None
    fig, ax = plt.subplots(figsize=(6.6, 7.7))
    ax.set_axis_off()
    ax.set_title("Table S1: published discharge duration", pad=16)
    table = ax.table(cellText=[[i + 1, SCHEDULE[i], 46 - i, SCHEDULE[45 - i]] for i in range(23)],
                     colLabels=["Cycle", "Duration (min)", "Paired cycle", "Duration (min)"],
                     cellLoc="center", loc="center", bbox=[0, 0, 1, .95])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    for (rr, cc), cell in table.get_celld().items():
        cell.set_edgecolor("#cbd5e1")
        cell.set_linewidth(.4)
        if rr == 0:
            cell.set_facecolor("#e8f1f5")
    qa = {"TableS1_published_schedule": render_check(fig, output / "TableS1_published_schedule")}
    fig, ax = plt.subplots(figsize=(6.6, 3.9), gridspec_kw=dict(left=.13, right=.97, bottom=.17, top=.88))
    ax.plot(range(1, 47), SCHEDULE, color="#0072b2", marker="o", ms=3, lw=1.2)
    ax.set(title="Published satellite discharge schedule", xlabel="Cycle within one 46-cycle block",
           ylabel="Discharge duration (min)", xlim=(-.5, 47.5), ylim=(-2, 78), xticks=[1, 10, 20, 30, 40, 46])
    qa["FigureS2_published_schedule"] = render_check(fig, output / "FigureS2_published_schedule")
    first = [row for row in rows if int(row["battery"]) == 1]
    if first:
        numeric = lambda row, name: float(row[name])
        is_test = lambda row: row["periodic_full_capacity_test"] in (True, "True")
        fig, ax = plt.subplots(figsize=(6.6, 4.4), gridspec_kw=dict(left=.13, right=.97, bottom=.29, top=.88))
        ax.scatter([numeric(r, "physical_cycle") for r in first], [numeric(r, "measured_discharge_capacity_Ah") for r in first],
                   color="#0072b2", s=3, label="Recorded discharge capacity")
        tests = [r for r in first if is_test(r)]
        ax.scatter([numeric(r, "physical_cycle") for r in tests], [numeric(r, "measured_discharge_capacity_Ah") for r in tests],
                   s=10, facecolor="white", edgecolor="#d55e00", linewidth=.6, zorder=4, label="Labelled full-capacity test")
        ax.set(title="Recorded capacity: first cell in satellite batch", xlabel="Physical raw cycle", ylabel="Discharge capacity (Ah)")
        ax.margins(x=.035, y=.08)
        ax.legend(loc="upper center", bbox_to_anchor=(.5, -.23))
        qa["FigureS3_first_cell_raw_capacity"] = render_check(fig, output / "FigureS3_first_cell_raw_capacity")
        regular = [r for r in first if not is_test(r)]
        available = [r for r in regular if r.get("longest_negative_run_span_min") not in (None, "")]
        differences = np.array([float(r["longest_negative_run_minus_published_min"]) for r in available])
        stage3_zero = [r for r in regular if float(r.get("stage3_recorded_span_min") or 0) == 0]
        all_medians = np.array([float(r["longest_negative_run_current_median_A"]) for r in available])
        fig, axes = plt.subplots(2, 1, figsize=(7.2, 6.3), gridspec_kw=dict(left=.12, right=.97, bottom=.12, top=.91, hspace=.58))
        axes[0].plot([int(r["regular_cycle_index"]) for r in regular],
                     [float(r["published_duration_min"]) for r in regular], lw=.7, color="#888888", label="Published block repeated from first non-test cycle")
        axes[0].scatter([int(r["regular_cycle_index"]) for r in available],
                        [float(r["longest_negative_run_span_min"]) for r in available],
                        s=4, color="#0072b2", label="Recorded longest continuous discharge segment", zorder=3)
        axes[0].set(title="Recorded spans and the stated 46-cycle schedule", xlabel="Non-test raw-cycle order", ylabel="Duration (min)")
        axes[0].margins(.035)
        axes[0].legend(loc="upper center", bbox_to_anchor=(.5, -.23), ncol=1)
        axes[1].scatter([int(r["regular_cycle_index"]) for r in available], differences, s=4, color="#0072b2")
        axes[1].axhline(0, color="#888888", lw=.65)
        axes[1].set(title="Signed difference under the fixed order convention", xlabel="Non-test raw-cycle order", ylabel="Recorded minus stated (min)")
        axes[1].margins(.035)
        qa["satellite_duration_order_audit"] = render_check(fig, output / "satellite_duration_order_audit")
        previous = ROOT / "outputs/full_reproduction/feature_audit/raw_curve_figures/figureS4_capacity_source.csv"
        with previous.open(encoding="utf-8-sig", newline="") as stream:
            previous_rows = [r for r in csv.DictReader(stream) if r["batch"] == "6"]
        previous_map = {int(r["physical_cycle"]): float(r["measured_discharge_capacity_Ah"]) for r in previous_rows}
        capacity_differences = [float(r["measured_discharge_capacity_Ah"]) - previous_map[int(r["physical_cycle"])] for r in first]
        summary = dict(battery=1, physical_cycles=len(first), labelled_capacity_test_records=len(tests),
                       non_test_records=len(regular), full_cycle_negative_current_span_available=len(available),
                       stage3_zero_span_records=len(stage3_zero), stage3_zero_physical_cycles=[int(r["physical_cycle"]) for r in stage3_zero],
                       capacity_vs_saved_S4_exact_matches=sum(d == 0 for d in capacity_differences),
                       capacity_vs_saved_S4_max_absolute_difference_Ah=max(abs(d) for d in capacity_differences),
                       signed_recorded_minus_schedule_min=dict(min=float(differences.min()), max=float(differences.max()),
                                                               mean=float(differences.mean()), mean_absolute=float(np.abs(differences).mean())),
                       observed_rounded_integer_minutes_equal_fixed_schedule=sum(
                           round(float(r["longest_negative_run_span_min"])) == int(r["published_duration_min"]) for r in available),
                       rounded_count_interpretation="Descriptive agreement at the printed integer-minute resolution, not a scientific tolerance or proven cycle alignment",
                       longest_negative_run_current_median_A_range=[float(all_medians.min()), float(all_medians.max())],
                       cycle_alignment_policy="All non-test records in original order; repeat the 46-cycle schedule from first non-test record. Phase/cycle correspondence is unreported and NOT established; do not attribute differences solely to schedule deviations.",
                       no_raw_cycles_excluded=True, no_phase_fit=True)
        first_raw_block = available[:43]
        write_csv(output / "first_43_non_test_raw_records_descriptive.csv", [dict(
            regular_cycle_index=int(r["regular_cycle_index"]), physical_cycle=int(r["physical_cycle"]),
            measured_longest_discharge_span_min=float(r["longest_negative_run_span_min"]),
            printed_integer_minute=round(float(r["longest_negative_run_span_min"]))) for r in first_raw_block])
        summary["exploratory_raw_pattern"] = dict(
            first_43_integer_minute_sequence=[round(float(r["longest_negative_run_span_min"])) for r in first_raw_block],
            first_block_72minute_records=sum(round(float(r["longest_negative_run_span_min"])) == 72 for r in first_raw_block),
            published_46block_72minute_records=SCHEDULE.count(72),
            observation="First non-test recorded sequence returns to 5 min at order43, then 5 min at44. This is a descriptive raw-record observation, not a corrected published schedule or a fitted reindexing.")
        write_json(output / "SATELLITE_INDEPENDENT_CHECKS.json", summary)
    manifest.update(figure_qa=qa, figure_style_sha256=digest(STYLE), render_script_sha256=digest(__file__), status="complete")
    write_json(output / "SATELLITE_SUPPLEMENT.json", manifest)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/full_reproduction/satellite_protocol_supplement_20261110")
    parser.add_argument("--batteries", type=int, nargs="+", default=[1])
    parser.add_argument("--render-only", action="store_true", help="Read this supplement's saved CSV and manifest; no raw extraction")
    args = parser.parse_args()
    if len(set(args.batteries)) != len(args.batteries) or any(i < 1 or i > 8 for i in args.batteries):
        raise ValueError("Expected unique satellite battery numbers in 1..8")
    if args.render_only:
        with (args.output / "satellite_all_raw_cycle_capacity_and_stage3_duration.csv").open(encoding="utf-8-sig", newline="") as stream:
            result_rows = list(csv.DictReader(stream))
        result_manifest = read_json(args.output / "SATELLITE_SUPPLEMENT.json")
    else:
        result_rows, result_manifest = extract(args)
    plot(args, result_rows, result_manifest)
