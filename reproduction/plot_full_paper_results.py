"""Render audited report tables and prespecified exported curves; never train."""

import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.text import Text
import numpy as np


LANES = {"published": "published_label_monotonicity", "paperloss": "paper_prediction_monotonicity"}
LANE_LABEL = {"published": "Published-code loss", "paperloss": "Paper-formula loss"}
GROUPS = ["XJTU/" + str(i) for i in range(1, 7)] + ["TJU/1", "TJU/2", "TJU/3", "MIT", "HUST"]
DOMAINS = ["XJTU/1", "TJU/3", "MIT", "HUST"]
MODELS = ["PINN", "MLP", "CNN"]
COLORS = {"PINN": "#197c80", "MLP": "#b96537", "CNN": "#655897", "Measured": "#252525"}
MARKERS = {"PINN": "o", "MLP": "s", "CNN": "^"}
SOURCE_MODES = {"supplied_author_checkpoint": "Author weights", "reproduced_checkpoint": "Reproduced weights"}
TABLES = ["repeat_summaries", "per_repeat", "per_battery_summaries", "published_comparisons", "comparison_effects", "task_audit"]


def apply_figure_style():
    # Mechanics adapted from the installed Apache-2.0 figure-style helper.
    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"],
        "font.size": 9, "axes.labelsize": 9, "axes.titlesize": 9, "legend.fontsize": 8,
        "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.linewidth": .6,
        "xtick.direction": "out", "ytick.direction": "out", "xtick.major.size": 3,
        "ytick.major.size": 3, "axes.spines.top": False, "axes.spines.right": False,
        "legend.frameon": False, "lines.linewidth": 1.15, "savefig.dpi": 300,
        "figure.dpi": 120, "pdf.fonttype": 42, "ps.fonttype": 42,
        "axes.titlelocation": "left", "axes.titleweight": "normal", "axes.unicode_minus": False})


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def scalar(value):
    if not isinstance(value, str):
        return value
    if value == "":
        return None
    if value in {"True", "False"}:
        return value == "True"
    try:
        result = float(value)
        if math.isfinite(result):
            return int(result) if re.fullmatch(r"-?\d+", value) else result
    except ValueError:
        pass
    return value


def read_csv(path):
    if not Path(path).is_file():
        return []
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        return [{key: scalar(value) for key, value in row.items()} for row in csv.DictReader(stream)]


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def write_csv(path, rows):
    columns = list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows({key: json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list)) else value
                         for key, value in row.items()} for row in rows)


def short_group(group):
    return group.replace("XJTU/", "XJTU ").replace("TJU/", "TJU ")


def source_group(dataset, batch):
    if dataset == "XJTU":
        names = ["2C", "3C", "R2.5", "R3", "RW", "satellite"]
        index = names.index(str(batch)) if str(batch) in names else int(batch)
        return "XJTU/" + str(index + 1)
    if dataset == "TJU":
        return "TJU/" + str(int(batch) + 1)
    return dataset


class Inputs:
    def __init__(self, report, evidence):
        self.report, self.evidence = report, evidence
        self.hashes, self.parts, self.curves, self.truths = {}, defaultdict(list), [], {}
        self.synthetic = False
        document = {}
        if report and (report / "suite_report.json").is_file():
            document = read_json(report / "suite_report.json")
            if document.get("metadata_only"):
                raise ValueError("Figures require the independently recomputed report")
            self.synthetic = bool(document.get("synthetic_layout_fixture"))
            self.hashes[str(report / "suite_report.json")] = digest(report / "suite_report.json")
        if evidence:
            index_path = evidence / "suite_evidence_index.json"
            index = read_json(index_path)
            if index.get("format") != "pinn4soh-suite-evidence-export-v1":
                raise ValueError("Unsupported evidence index format")
            self.synthetic |= bool(index.get("synthetic_layout_fixture"))
            self.hashes[str(index_path)] = digest(index_path)
            for record in index["files"]:
                path = (evidence / record["path"]).resolve()
                if evidence.resolve() not in path.parents or not path.is_file() or digest(path) != record["sha256"]:
                    raise ValueError("Evidence part is outside export, missing or hash-inconsistent: " + record["path"])
                self.hashes[str(path)] = record["sha256"]
                self.parts[record["kind"]].append(path)
            self.curves = self.json_parts("curve_predictions")
            self.truths = {row["truth_id"]: row for row in self.json_parts("curve_truth")}
            self.schemas = index.get("table_schemas", {})
        self.tables = {}
        for name in TABLES:
            path = report / (name + ".csv") if report else None
            if path and path.is_file():
                self.hashes[str(path)] = digest(path)
                self.tables[name] = read_csv(path)
                copied = self.parts.get("csv_" + name, [])
                if copied and digest(path) != digest(copied[0]):
                    raise ValueError("Report CSV differs from the supplied export: " + name)
            elif self.parts.get("csv_" + name):
                self.tables[name] = read_csv(self.parts["csv_" + name][0])
            else:
                schema = getattr(self, "schemas", {}).get(name, [])
                exported = [dict(zip(schema, row)) for row in self.json_parts("table_" + name)]
                fallback = {"repeat_summaries": "summaries", "task_audit": "tasks"}.get(name, name)
                self.tables[name] = exported or document.get(fallback, [])
        if document.get("summaries"):
            fields = ["comparison_group", "evaluation", "aggregation", "convention", "test_split_sha256",
                      "n_repeats", "expected_repeats", "complete_repeat_set"]
            fields += [metric + suffix for metric in ["MAE", "MAPE", "RMSE", "MSE"]
                       for suffix in ["_mean", "_sd_ddof1", "_n"]]
            signature = lambda rows: sorted(json.dumps({key: row.get(key) for key in fields}, sort_keys=True) for row in rows)
            if signature(document["summaries"]) != signature(self.tables["repeat_summaries"]):
                raise ValueError("Report JSON summaries differ from selected CSV/export summaries")
        self.summaries = [row for row in self.tables["repeat_summaries"]
                          if row.get("aggregation") == "battery_macro" and row.get("convention") == "standard"]
        metadata = {row["comparison_group"]: row for row in self.summaries}
        self.repeats = [{**metadata.get(row.get("comparison_group"), {}), **row} for row in self.tables["per_repeat"]
                        if row.get("aggregation") == "battery_macro" and row.get("convention") == "standard"]
        self.by_task = {(row["task_id"], row.get("evaluation", "final")): row for row in self.repeats}
        self.by_curve = defaultdict(list)
        for row in self.curves:
            meta = self.by_task.get((row["task_id"], row["evaluation"]))
            if meta and int(meta.get("repeat", 0)) == 1:
                self.by_curve[(row["task_id"], row["evaluation"])].append(row)

    def json_parts(self, kind):
        return [row for path in self.parts.get(kind, []) for row in read_json(path)]


class Renderer:
    def __init__(self, options):
        self.options, self.output = options, options.output.resolve()
        self.data = Inputs(options.report, options.evidence)
        self.protocol = read_json(options.protocol)
        self.references = options.references or options.protocol.parent / "source_data"
        self.per_battery_paper = read_csv(self.references / "paper_per_battery_tables.csv")
        s7 = self.references / "paper_table_s7.json"
        self.s7 = read_json(s7) if s7.is_file() else {}
        self.qa, self.inventory, self.values, self.curve_exports, self.ambiguities = [], [], [], [], []
        self.output.mkdir(parents=True, exist_ok=True)
        (self.output / "figures").mkdir(exist_ok=True)
        (self.output / "tables").mkdir(exist_ok=True)

    def select(self, lane, family, group, model="PINN", size=None, evaluation="final", **extra):
        loss = LANES[lane] if model == "PINN" else "author_current"
        rows = [row for row in self.data.summaries if row.get("family") == family and row.get("paper_group") == group
                and row.get("model") == model and row.get("sample_size") == size and row.get("loss_variant") == loss
                and row.get("evaluation") == evaluation and row.get("protocol_id") == self.options.protocol_id
                and row.get("validation_protocol") == self.options.validation_protocol
                and all(row.get(key) == value for key, value in extra.items())]
        if len(rows) > 1:
            self.ambiguities.append({"lane": lane, "family": family, "group": group, "model": model,
                                     "sample_size": size, "evaluation": evaluation, "filters": extra,
                                     "comparison_groups": [row["comparison_group"] for row in rows]})
            return None, "ambiguous"
        if not rows:
            return None, "pending"
        row = rows[0]
        if model != "PINN":
            anchor, _ = self.select(lane, family, group, "PINN", size, evaluation, **extra)
            if not anchor or not row.get("test_split_sha256") or row.get("test_split_sha256") != anchor.get("test_split_sha256"):
                return row, "test_split_unmatched"
        valid = (row.get("complete_repeat_set") is True and int(row.get("n_repeats", 0)) == 10
                 and int(row.get("expected_repeats", 0)) == 10)
        return row, "complete" if valid else "pending"

    def cell(self, lane, family, group, model, metric, size=None, published=None, **extra):
        row, status = self.select(lane, family, group, model, size, **extra)
        observed = row.get(metric + "_mean") if row and status == "complete" else None
        sd = row.get(metric + "_sd_ddof1") if row and status == "complete" else None
        if status == "complete" and (row.get(metric + "_n") != 10 or observed is None or sd is None):
            observed, sd, status = None, None, "metric_pending"
        result = {"lane": lane, "family": family, "group": group, "model": model, "sample_size": size,
                  "metric": metric, **extra, "status": status, "comparison_group": row.get("comparison_group") if row else None,
                  "n_repeats": row.get("n_repeats", 0) if row else 0, "expected_repeats": row.get("expected_repeats", 10) if row else 10,
                  "observed": observed, "sample_sd_ddof1": sd, "partial_mean": row.get(metric + "_mean") if row and status != "complete" else None,
                  "published": published, "difference": observed - published if observed is not None and published is not None else None}
        self.values.append(result)
        return result

    def figure(self, nrows, ncols, title, caption, width=7.2, panel_height=1.8):
        height = nrows * panel_height + 1.4
        fig, axes = plt.subplots(nrows, ncols, figsize=(width, height), squeeze=False)
        top = "SYNTHETIC LAYOUT FIXTURE | " if self.data.synthetic else ""
        fig.suptitle(top + title, x=.08, y=.99, ha="left", fontsize=9)
        fig.text(.08, .025, caption, fontsize=8, va="bottom", linespacing=1.3)
        fig.subplots_adjust(left=.085, right=.975, bottom=.9 / height,
                            top=1 - .8 / height, wspace=.32, hspace=.55)
        return fig, axes

    def save(self, fig, name, table=False):
        directory = self.output / ("tables" if table else "figures")
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        suppressed = set()
        for ax in fig.axes:
            if not ax.get_visible():
                suppressed.update(id(item) for item in ax.findobj(Text))
                continue
            for axis in [ax.xaxis, ax.yaxis]:
                low, high = sorted(axis.get_view_interval())
                for tick in axis.get_major_ticks() + axis.get_minor_ticks():
                    if not ax.axison or not axis.get_visible() or not low <= tick.get_loc() <= high:
                        suppressed.update([id(tick.label1), id(tick.label2)])
        text_items = [(text, text.get_window_extent(renderer)) for text in fig.findobj(Text)
                      if text.get_visible() and text.get_text().strip() and id(text) not in suppressed]
        outside = [text.get_text() for text, box in text_items if not fig.bbox.contains(box.x0, box.y0) or not fig.bbox.contains(box.x1, box.y1)]
        overlaps = []
        for index, (first, first_box) in enumerate(text_items):
            for second, second_box in text_items[index + 1:]:
                if first_box.overlaps(second_box):
                    overlaps.append([first.get_text(), second.get_text()])
        for extension in ["png", "pdf"]:
            fig.savefig(directory / (name + "." + extension), dpi=300, facecolor="white")
        overflowing_cells = []
        for ax in fig.axes:
            for table_object in ax.tables:
                for key, cell in table_object.get_celld().items():
                    text_box = cell.get_text().get_window_extent(renderer)
                    cell_box = cell.get_window_extent(renderer)
                    if text_box.width > cell_box.width - 3 or text_box.height > cell_box.height - 2:
                        overflowing_cells.append({"cell": list(key), "text": cell.get_text().get_text()})
        self.qa.append({"figure": name, "outside_text": outside, "text_bbox_overlaps": overlaps,
                        "overflowing_table_cells": overflowing_cells,
                        "geometry_passed": not outside and not overlaps and not overflowing_cells})
        self.inventory.append({"name": name, "kind": "table" if table else "figure",
                               "png": str((directory / (name + ".png")).relative_to(self.output)),
                               "pdf": str((directory / (name + ".pdf")).relative_to(self.output))})
        plt.close(fig)

    def legend(self, fig, measured=False):
        names = (["Measured"] if measured else []) + MODELS
        handles = [Line2D([0], [0], color=COLORS[name], marker=MARKERS.get(name),
                          linewidth=1.3, markersize=3, label=name) for name in names]
        fig.legend(handles=handles, loc="upper right", bbox_to_anchor=(.985, .955), ncol=len(names), fontsize=8)

    def regular_error(self, lane):
        fig, axes = self.figure(3, 1, "Fig. 4b | " + LANE_LABEL[lane],
            "Exact-battery macro metrics; measured-SOH MAPE denominator. Mean +/- sample SD (ddof=1).\n"
            "n=10 seeds per complete configuration; p = pending. Shared MLP/CNN. Lower is better.", panel_height=1.55)
        self.legend(fig)
        for ax, metric, ylabel in zip(axes[:, 0], ["MAE", "MAPE", "RMSE"], ["MAE (SOH pp)", "MAPE (%)", "RMSE (SOH pp)"]):
            for mi, model in enumerate(MODELS):
                for gi, group in enumerate(GROUPS):
                    cell = self.cell(lane, "regular", group, model, metric)
                    x = gi + (mi - 1) * .2
                    if cell["observed"] is not None and cell["sample_sd_ddof1"] is not None:
                        ax.errorbar(x, cell["observed"] * 100, yerr=cell["sample_sd_ddof1"] * 100,
                                    color=COLORS[model], marker=MARKERS[model], markersize=3,
                                    capsize=2, linewidth=.8, linestyle="none")
                    else:
                        ax.text(x, .04, "p", transform=ax.get_xaxis_transform(), color="#777777", ha="center", fontsize=7)
            ax.set_xticks(range(len(GROUPS)))
            ax.set_xticklabels([short_group(group) for group in GROUPS])
            ax.set_xlim(-.6, len(GROUPS) - .4)
            ax.set_ylim(bottom=0)
            ax.set_ylabel(ylabel)
            ax.grid(axis="y", alpha=.18)
        self.save(fig, "Fig4b_" + lane)

    def small(self, lane, groups, name, title):
        columns = 2 if len(groups) == 2 else 3
        rows = math.ceil(len(groups) / columns)
        fig, axes = self.figure(rows, columns, title + " | " + LANE_LABEL[lane],
            "Mean +/- sample SD (ddof=1); exact-battery macro RMSE. n=10 seeds per complete point.\n"
            "Fixed test batteries; shared MLP/CNN; p = pending. Lower is better.", panel_height=2.0)
        self.legend(fig)
        for ax, group in zip(axes.flat, groups):
            pending_points = defaultdict(list)
            for model in MODELS:
                values, errors = [], []
                for size in range(1, 5):
                    cell = self.cell(lane, "small_sample", group, model, "RMSE", size)
                    values.append(cell["observed"] * 100 if cell["observed"] is not None else np.nan)
                    errors.append(cell["sample_sd_ddof1"] * 100 if cell["sample_sd_ddof1"] is not None else np.nan)
                    if cell["observed"] is None:
                        pending_points[size].append(model)
                ax.errorbar(range(1, 5), values, yerr=errors, color=COLORS[model], marker=MARKERS[model], markersize=3, capsize=2)
            for size, models in pending_points.items():
                if len(models) == len(MODELS):
                    ax.text(size, .05, "p", transform=ax.get_xaxis_transform(), color="#777777", fontsize=7, ha="center")
                else:
                    for model in models:
                        ax.text(size + (MODELS.index(model) - 1) * .18, .05, "p", transform=ax.get_xaxis_transform(),
                                color=COLORS[model], fontsize=7, ha="center")
            ax.set(title=short_group(group), xlabel="Training batteries", ylabel="RMSE (SOH pp)", xticks=[1, 2, 3, 4], xlim=(.7, 4.3))
            ax.set_ylim(bottom=0)
            ax.grid(axis="y", alpha=.18)
        for ax in list(axes.flat)[len(groups):]:
            ax.set_visible(False)
        self.save(fig, name + "_" + lane)

    def selected_curve(self, lane, group, model):
        row, status = self.select(lane, "regular", group, model)
        if row is None or status in {"ambiguous", "test_split_unmatched"}:
            return [], "pending"
        repeats = [item for item in self.data.repeats if item.get("comparison_group") == row["comparison_group"]
                   and item.get("evaluation") == "final" and int(item.get("repeat", 0)) == 1]
        if len(repeats) != 1:
            return [], "pending" if not repeats else "ambiguous"
        return self.data.by_curve.get((repeats[0]["task_id"], "final"), []), status

    def regular_curves(self, lane):
        fig, axes = self.figure(4, 3, "Fig. 4a | " + LANE_LABEL[lane],
            "Prespecified repeat 1; no error-based selection. All available held-out batteries in each group.\n"
            "Uniform display sampling (up to 256 points/battery); x is retained-point order, not original cycle ID.", panel_height=1.55)
        self.legend(fig, measured=True)
        for ax, group in zip(axes.flat, GROUPS):
            collections, pending = {}, []
            for model in MODELS:
                records, status = self.selected_curve(lane, group, model)
                if not records:
                    pending.append(model)
                else:
                    collections[model] = records
            reference, offset, references = None, 0, {}
            for records in collections.values():
                reference = sorted(records, key=lambda item: self.data.truths[item["truth_id"]]["battery"])
                break
            expected_batteries = {self.data.truths[item["truth_id"]]["battery"] for item in reference or []}
            if reference and len(expected_batteries) != len(reference):
                raise ValueError("Duplicate battery in display curve collection")
            for record in reference or []:
                truth = self.data.truths.get(record["truth_id"])
                if truth is None:
                    raise ValueError("Curve refers to a missing measured series")
                indices, measured = np.array(truth["relative_retained_point_indices"]), np.array(truth["measured_SOH"])
                references[truth["battery"]] = (truth, offset)
                ax.plot(offset + indices, measured * 100, color=COLORS["Measured"], linewidth=.8)
                if offset:
                    ax.axvline(offset - .5, color="#bbbbbb", linewidth=.4)
                offset += int(truth["n_full_retained_points"])
            for model, records in collections.items():
                actual_batteries = [self.data.truths[item["truth_id"]]["battery"] for item in records]
                if len(actual_batteries) != len(expected_batteries) or set(actual_batteries) != expected_batteries:
                    raise ValueError("Model curves have incomplete or different test battery sets")
                for record in records:
                    truth = self.data.truths[record["truth_id"]]
                    known = references.get(truth["battery"])
                    if known is None or known[0] != truth:
                        raise ValueError("Model curves have unmatched battery identities or measured sampling")
                    x = known[1] + np.array(truth["relative_retained_point_indices"])
                    prediction = np.array(record["predicted_SOH"])
                    if len(prediction) != len(x) or not np.isfinite(prediction).all():
                        raise ValueError("Invalid display prediction array")
                    ax.plot(x, prediction * 100, color=COLORS[model], linewidth=.7, alpha=.9)
                    self.curve_exports.extend({"lane": lane, "group": group, "model": model,
                        "task_id": record["task_id"], "repeat": 1, "battery": truth["battery"],
                        "retained_point_index": int(index), "display_x": int(point),
                        "measured_SOH": float(measured), "predicted_SOH": float(predicted)}
                        for index, point, measured, predicted in zip(truth["relative_retained_point_indices"], x, truth["measured_SOH"], prediction))
            if not collections:
                ax.text(.5, .5, "pending", transform=ax.transAxes, ha="center", color="#777777", fontsize=8)
            elif pending:
                ax.text(.98, .04, "pending: " + "/".join(pending), transform=ax.transAxes, ha="right", color="#777777", fontsize=7)
            ax.set_title(short_group(group))
            ax.margins(.04)
            if ax.get_subplotspec().is_first_col():
                ax.set_ylabel("SOH (%)")
            if ax.get_subplotspec().is_last_row():
                ax.set_xlabel("Retained test point")
        axes.flat[-1].set_visible(False)
        self.save(fig, "Fig4a_" + lane)

    def table_image(self, name, title, columns, rows, labels, note):
        limit = 18
        for start in range(0, len(rows), limit):
            current = rows[start:start + limit]
            current_labels = ["\n".join(textwrap.wrap(label, 25, break_long_words=True, break_on_hyphens=False))
                              for label in labels[start:start + limit]]
            row_lines = [max([len(label.splitlines())] + [len(str(value).splitlines()) for value in row])
                         for label, row in zip(current_labels, current)]
            height = max(2.5, (sum(row_lines) + 2) * .13 + 1.25)
            fig = plt.figure(figsize=(7.2, height))
            watermark = "SYNTHETIC FIXTURE | " if self.data.synthetic else ""
            fig.text(.03, .98, "\n".join(textwrap.wrap(watermark + title, 100)), ha="left", va="top", fontsize=9)
            bottom, top = .68 / height, 1 - .5 / height
            ax = fig.add_axes([.03, bottom, .94, top - bottom])
            ax.axis("off")
            values = [[label] + row for label, row in zip(current_labels, current)]
            widths = [.31] + [.69 / len(columns)] * len(columns)
            table = ax.table(cellText=values, colLabels=["Group / battery"] + columns, cellLoc="center",
                             colLoc="center", bbox=[0, 0, 1, 1], colWidths=widths)
            table.auto_set_font_size(False)
            table.set_fontsize(7)
            for (ri, ci), cell in table.get_celld().items():
                cell.set_edgecolor("#d8d8d8")
                cell.set_linewidth(.35)
                cell.set_facecolor("#eef0f1" if ri == 0 else "#ffffff" if ri % 2 else "#f7f8f8")
                if ci == 0:
                    cell.set_text_props(ha="left")
            for ri, lines in enumerate([2] + row_lines):
                for ci in range(len(columns) + 1):
                    table[ri, ci].set_height(lines / (sum(row_lines) + 2))
            fig.text(.03, .04, note, fontsize=8, va="bottom")
            suffix = "_p" + str(start // limit + 1) if len(rows) > limit else ""
            self.save(fig, name + suffix, table=True)

    def metric_tables(self, lane, label, table, family):
        columns = table.get("column_order", table.get("columns", []))
        grid, delta, labels, long = [], [], [], []
        for key, published_values in table.get("rows", {}).items():
            group, size = key, None
            if family == "small_sample":
                group, size = key.rsplit("/n", 1)
                size = int(size)
            current, differences = [], []
            for index, column in enumerate(columns):
                model, metric = column.split("_")
                cell = self.cell(lane, family, group, model, metric, size, published_values[index])
                long.append({"table": label, **cell})
                current.append(format_cell(cell))
                differences.append(format_number(cell["difference"], signed=True) if cell["difference"] is not None else "pending")
            grid.append(current)
            delta.append(differences)
            labels.append(short_group(group) + (" / n=" + str(size) if size else ""))
        if not grid:
            return
        compact_columns = [column.replace("_", "\n") for column in columns]
        base = label.replace(" ", "_") + "_" + lane
        self.table_image(base, label + " | " + LANE_LABEL[lane], compact_columns, grid, labels,
                         "SOH fractions. Cells: mean (sample SD, ddof=1), n=10; pending = incomplete/missing.\nShared MLP/CNN; exact-battery macro metrics and measured-SOH MAPE denominator.")
        self.table_image(base + "_difference", label + " difference | reproduced - published", compact_columns, delta, labels,
                         "Signed differences in SOH fractions; no best-convention selection.\nRounded numeric agreement does not establish protocol equivalence.")
        write_csv(self.output / "tables" / (base + ".csv"), long)

    def transfer(self, lane, mode):
        table = self.protocol.get("published_table_4", {})
        long = []
        for evaluation, paper_key in [("final", "fine_tuning"), ("source_only", "source_only")]:
            grid, differences, labels = [], [], []
            for ti, target in enumerate(DOMAINS):
                current, delta = [], []
                for si, source in enumerate(DOMAINS):
                    if source == target:
                        current.append("--")
                        delta.append("--")
                        continue
                    dataset = source.split("/")[0]
                    batch = {"XJTU": 0, "TJU": 2}.get(dataset, -1)
                    published = table.get(paper_key, [[None] * 4] * 4)[ti][si]
                    cell = self.cell(lane, "transfer", target, "PINN", "RMSE", published=published,
                                     evaluation=evaluation, source=dataset, source_batch=batch,
                                     source_mode=mode, transfer_design=self.options.transfer_design)
                    long.append({"table": "Table 4", "target": target, "source_group": source, **cell})
                    current.append(format_cell(cell))
                    delta.append(format_number(cell["difference"], signed=True) if cell["difference"] is not None else "pending")
                for size in [1, 2]:
                    published = table.get("target_only_n" + str(size), [None] * 4)[ti]
                    cell = self.cell(lane, "small_sample", target, "PINN", "RMSE", size, published,
                                     evaluation="final")
                    long.append({"table": "Table 4", "target": target, "source_group": "target_only_n" + str(size), **cell})
                    current.append(format_cell(cell))
                    delta.append(format_number(cell["difference"], signed=True) if cell["difference"] is not None else "pending")
                labels.append(short_group(target))
                grid.append(current)
                differences.append(delta)
            base = "Table_4_" + lane + "_" + mode + "_" + evaluation
            columns = [short_group(item).replace(" ", "\n") for item in DOMAINS] + ["Target\nn=1", "Target\nn=2"]
            note = "RMSE fraction: mean (sample SD, ddof=1), n=10 runs. Rows: target; columns: source.\n"
            note += "Author source-only can reuse identical weights; SD=0 does not show independent pretraining."
            self.table_image(base, "Table 4 | " + LANE_LABEL[lane] + " | " + SOURCE_MODES[mode] + " | " + evaluation,
                             columns, grid, labels, note)
            self.table_image(base + "_difference", "Table 4 | reproduced - published | " + evaluation,
                             columns, differences, labels, "Signed RMSE difference in SOH fractions. Target-only results are shared across source modes.")
        write_csv(self.output / "tables" / ("Table_4_" + lane + "_" + mode + ".csv"), long)

    def battery_tables(self, lane):
        for label, dataset in [("Table S3", "XJTU"), ("Table S4", "TJU"), ("Table S5", "HUST"), ("Table S6", "MIT")]:
            paper_rows = [row for row in self.per_battery_paper if row["group"].split("/")[0] == dataset]
            long, grid, deltas, labels = [], [], [], []
            for published in paper_rows:
                group, current, delta = published["group"], [], []
                for model in MODELS:
                    selected, status = self.select(lane, "regular", group, model)
                    batteries = [row for row in self.data.tables["per_battery_summaries"]
                                 if selected and row.get("comparison_group") == selected["comparison_group"]
                                 and row.get("convention") == "standard" and row.get("evaluation") == "final"
                                 and battery_matches(str(row.get("battery", "")), published)]
                    valid = (status == "complete" and len(batteries) == 1
                             and batteries[0].get("complete_repeat_set") is True
                             and int(batteries[0].get("n_repeats", 0)) == 10
                             and int(batteries[0].get("expected_repeats", 0)) == 10)
                    row = batteries[0] if valid else None
                    for metric in ["MAPE", "RMSE"]:
                        observed = row.get(metric + "_mean") if row else None
                        sd = row.get(metric + "_sd_ddof1") if row else None
                        metric_valid = valid and observed is not None and sd is not None
                        if not metric_valid:
                            observed, sd = None, None
                        reference = published.get(model + "_" + metric)
                        cell = {"table": label, "lane": lane, "group": group, "battery": published["battery"],
                                "date": published.get("date"), "model": model, "metric": metric,
                                "status": "complete" if metric_valid else "pending", "observed": observed,
                                "sample_sd_ddof1": sd, "published": reference,
                                "difference": observed - reference if observed is not None and reference is not None else None,
                                "comparison_group": selected.get("comparison_group") if selected else None}
                        long.append(cell)
                        current.append(format_cell(cell))
                        delta.append(format_number(cell["difference"], signed=True) if cell["difference"] is not None else "pending")
                labels.append(short_group(group) + " / " + (str(published.get("date")) + " / " if published.get("date") else "") + str(published["battery"]))
                grid.append(current)
                deltas.append(delta)
            if not grid:
                continue
            columns = [model + "\n" + metric for model in MODELS for metric in ["MAPE", "RMSE"]]
            base = label.replace(" ", "_") + "_" + lane
            self.table_image(base, label + " | " + LANE_LABEL[lane], columns, grid, labels,
                             "Per-battery mean (sample SD, ddof=1), n=10; SOH fractions. Exact battery/date matching.\nPending cells retain missing, ambiguous or incomplete groups.")
            self.table_image(base + "_difference", label + " | reproduced - published", columns, deltas, labels,
                             "Signed per-battery metric differences in SOH fractions; no rounded identity guessing.")
            write_csv(self.output / "tables" / (base + ".csv"), long)

    def efficiency(self):
        document = read_json(self.options.efficiency) if self.options.efficiency else {}
        observed = document.get("rows", [])
        config = document.get("config", {})
        formal = bool(document.get("status") == "complete" and not document.get("smoke")
                      and config.get("table_s8_iteration_protocol") and config.get("batch_size") == 1000
                      and config.get("forwards") == 1000 and config.get("device") == "cpu"
                      and config.get("eval") is True and config.get("no_grad") is True)
        experiment = next(item for item in self.protocol["experiments"] if item["id"] == "E_inference_efficiency")
        rows, grid, labels = [], [], []
        for index, model in enumerate(["F", "MLP", "CNN"]):
            for threads in [1, 2]:
                found = [row for row in observed if row.get("model") == model and row.get("threads") == threads]
                current = found[0] if len(found) == 1 else None
                published = experiment["reported_seconds_per_1000_samples"][index]
                valid = (current and formal and current.get("independent_process_repeats", 0) >= 2
                         and current.get("independent_process_repeats") == config.get("repeats")
                         and current.get("sample_sd_seconds_per_1000_samples") is not None)
                value = current.get("mean_seconds_per_1000_samples") if valid else None
                sd = current.get("sample_sd_seconds_per_1000_samples") if valid else None
                row = {"table": "Table S8", "model": model, "threads": threads,
                       "published_parameters": experiment["reported_parameters"][index],
                       "observed_parameters": current.get("parameters") if current else None,
                       "published_seconds_per_1000": published, "observed_seconds_per_1000": value,
                       "sample_sd_seconds_per_1000": sd,
                       "difference_seconds_per_1000": value - published if value is not None else None,
                       "n_independent_process_repeats": current.get("independent_process_repeats") if current else None,
                       "status": "complete" if value is not None else "pending", "input_smoke": document.get("smoke")}
                rows.append(row)
                labels.append(model + " / " + str(threads) + " thread" + ("s" if threads == 2 else "")
                              + "\nn=" + str(current.get("independent_process_repeats")) if valid
                              else model + " / " + str(threads) + " thread" + ("s" if threads == 2 else "") + "\nn=pending")
                grid.append([str(row["published_parameters"]), str(row["observed_parameters"]) if current else "pending",
                             format_number(published, scientific=True),
                             format_number(value, scientific=True) + "\n(" + format_number(sd, scientific=True) + ")" if value is not None else "pending",
                             format_number(row["difference_seconds_per_1000"], scientific=True, signed=True) if value is not None else "pending"])
        self.table_image("Table_S8", "Table S8 | CPU inference efficiency", ["Paper\nparams", "Measured\nparams", "Paper\nseconds", "Measured\nmean +/- SD", "Difference\nseconds"], grid, labels,
                         "Batch 1000; 1000 forwards; mean (sample SD, ddof=1); n independent processes per row.\nExplicit threads, warmup, eval/no_grad. Hardware/software differences limit timing comparison.")
        write_csv(self.output / "tables" / "Table_S8.csv", rows)

    def schematics_and_coverage(self):
        schematic_dir = self.output / "schematics"
        schematic_dir.mkdir(exist_ok=True)
        diagrams = {
            "Fig1_computational_framework": '''flowchart LR
    A["Provided feature CSVs: 16 descriptors and retained cycle input"] --> B["Loader normalization and declared split protocol"]
    B --> C["Adjacent retained-point pairs: x1,t1,y1 and x2,t2,y2"]
    C --> F["Solution network F: predicted SOH u"]
    F --> D["Autograd first derivatives: u_t and u_x"]
    B --> G["Dynamics network G: receives x,t,u,u_x,u_t"]
    F --> G
    D --> G
    D --> R["PDE residual r = u_t - G"]
    G --> R
    C --> L["Data MSE at both pair endpoints"]
    F --> L
    R --> T["Total loss: L_data + alpha L_PDE + beta L_mono"]
    L --> T
    C --> P["Publication-era code: sum ReLU(y2-y1); parameter gradient is zero"]
    F --> Q["Paper Eq.8 lane: sum ReLU(u2-u1)"]
    P --> T
    Q --> T
''',
            "Fig6_transfer_framework": '''flowchart LR
    A["Source-domain data"] --> B["Pretrain solution F and dynamics G"]
    W["Author-supplied F/G checkpoint; separate source mode"] --> C["Frozen source checkpoint identity"]
    B --> C
    C --> D["Source-only prediction on exact target test batteries"]
    C --> E["Clone F; freeze G parameters"]
    T["Prespecified target adaptation battery and validation protocol"] --> E
    E --> F["Update F through data, PDE and lane-specific monotonicity losses"]
    F --> H["Target test predictions after fine-tuning"]
    D --> J["Paired RMSE comparison by repeat, target and source mode"]
    H --> J
    C --> K["Artifact audit: unchanged G hash; changed F; checkpoint lineage"]
    F --> K
    N["Independent target-only n=1 and n=2 models; exact protocol matching"] --> J
'''}
        for name, content in diagrams.items():
            (schematic_dir / (name + ".mmd")).write_text(content, encoding="utf-8")
        notes = """# 计算框架示意图

这两幅 Mermaid 源图可编辑，依据当前训练封装及作者网络前向过程核对；不表示重新进行了电池老化、设备搭建或原始特征提取。

- Fig. 1 只重画可核对的计算路径。实测电压、电流、时间波形与特征窗口由独立原始数据审计提供；CSV 训练不能替代原始特征提取验证。
- published 使用发表时期代码的 `sum ReLU(y2-y1)`，对模型参数梯度为零；paperloss 使用论文 Eq.8 的 `sum ReLU(u2-u1)`，两者分开训练、统计、出图。
- 当前仓库另有 `sum ReLU((u2-u1)*(y1-y2))`，与以上两式均不同；本图未把它混入这两条实验通道。
- G 的输入包含 u_t。图中的 PDE 残差表达不构成电化学动力学可辨识性的证明。
- Fig. 6 表示经代码核对的迁移流程。只有实际运行产物中的冻结 G 哈希、F 更新、精确电池划分及重复结果才可证明相应方向完成；示意图本身不是数值复现结果。
- Fig. S1 的设备照片与 Fig. S2/S3 的原始实验协议属于原文材料审计；Fig. S4 需独立完整/不完整放电及插值证据，不能用云端训练替代。
"""
        (schematic_dir / "计算框架说明.md").write_text(notes, encoding="utf-8")
        rules_path = Path(__file__).with_name("suite_evidence_rules.json")
        rules = read_json(rules_path)
        coverage = []
        for entry in rules["figure_table_inventory"]:
            identifier = entry["id"]
            prefix = identifier.replace("Fig. ", "Fig").replace(" ", "_")
            artifacts = [item for item in self.inventory if item["name"] == prefix or item["name"].startswith(prefix + "_")]
            status = "pending_external_dataset_audit"
            lane_status = {}
            for lane in self.options.lanes:
                if identifier == "Fig. 4a":
                    groups = {(row["group"], row["model"]) for row in self.curve_exports if row["lane"] == lane}
                    lane_status[lane] = {"status": "complete_repeat1_display" if len(groups) == 33 else "partial_repeat1_display" if groups else "pending",
                                         "model_group_curves": len(groups), "expected_model_groups": 33}
                elif artifacts:
                    if identifier.startswith("Table"):
                        paths = list((self.output / "tables").glob(prefix + "_" + lane + "*.csv"))
                        if identifier == "Table S8":
                            paths = [self.output / "tables/Table_S8.csv"]
                        cells = [row for path in paths for row in read_csv(path)]
                    else:
                        cells = [row for row in self.values if row["lane"] == lane and row["family"] in entry["families"]
                                 and (not entry.get("sample_groups") or row["group"] in entry["sample_groups"])]
                    counts = Counter(row.get("status", "pending") for row in cells)
                    lane_status[lane] = {"status": "complete_numeric_cells" if cells and set(counts) == {"complete"}
                                         else "partial_numeric_cells" if counts.get("complete") else "pending",
                                         "cell_counts": dict(counts)}
            if artifacts:
                status = "rendered_with_explicit_evidence_status"
            if identifier in {"Fig. 1", "Fig. 6"}:
                status = "code_audited_editable_schematic; numerical_or_physical_claims_separate"
            if identifier in {"Fig. S1", "Fig. S2", "Fig. S3"}:
                status = "publication_material_audit; no_independent_hardware_reproduction"
            coverage.append({**entry, "render_status": status, "synthetic_layout_fixture": self.data.synthetic,
                             "lane_status": lane_status, "rendered_artifacts": artifacts,
                             "independent_physical_experiment": False})
        write_json(self.output / "figure_table_render_status.json", coverage)
        write_csv(self.output / "figure_table_render_status.csv", coverage)
        source_paths = [Path(__file__).with_name("run_extended.py"), Path(__file__).with_name("run_transfer.py"),
                        Path(__file__).resolve().parents[1] / "PINN4SOH/Model/Model.py", rules_path]
        write_json(schematic_dir / "source_audit_sha256.json", {str(path.resolve()): digest(path) for path in source_paths if path.is_file()})

    def render(self):
        for lane in self.options.lanes:
            self.regular_curves(lane)
            self.regular_error(lane)
            self.small(lane, ["XJTU/1", "HUST"], "Fig5", "Fig. 5 | Small-sample estimation")
            self.small(lane, ["XJTU/" + str(i) for i in range(1, 7)], "FigS5", "Fig. S5 | All six XJTU batches")
            self.metric_tables(lane, "Table 2", self.protocol.get("published_table_2", {}), "regular")
            self.metric_tables(lane, "Table 3", self.protocol.get("published_table_3", {}), "small_sample")
            self.metric_tables(lane, "Table S7", self.s7, "small_sample")
            self.battery_tables(lane)
            for mode in SOURCE_MODES:
                self.transfer(lane, mode)
        self.efficiency()
        self.schematics_and_coverage()
        write_csv(self.output / "plotted_metric_cells.csv", self.values)
        write_csv(self.output / "repeat1_display_curves.csv", self.curve_exports)
        write_json(self.output / "figure_qa.json", self.qa)
        write_json(self.output / "ambiguous_groups.json", self.ambiguities)
        for name in ["comparison_effects", "published_comparisons"]:
            write_csv(self.output / "tables" / (name + "_audited.csv"), self.data.tables[name])
        self.brief()
        inputs = dict(self.data.hashes)
        for path in [self.options.protocol, self.references / "paper_per_battery_tables.csv", self.references / "paper_table_s7.json", self.options.efficiency]:
            if path and path.is_file():
                inputs[str(path.resolve())] = digest(path)
        manifest = {"format": "pinn4soh-rendered-paper-results-v1", "created_at": datetime.now(timezone.utc).isoformat(),
                    "synthetic_layout_fixture": self.data.synthetic, "training_performed": False,
                    "script_sha256": digest(__file__), "input_sha256": inputs, "figures": self.inventory,
                    "metric_cell_counts": dict(Counter(row["status"] for row in self.values)),
                    "ambiguous_group_count": len(self.ambiguities),
                    "all_text_geometry_passed": all(item["geometry_passed"] for item in self.qa),
                    "figure_style": "figure-style rules; apply_figure_style; PNG 300 dpi, PDF TrueType",
                    "statistic": "Report-provided mean and sample SD ddof=1; no metric recomputation or cross-configuration pooling",
                    "artifact_sha256": {str(path.relative_to(self.output)): digest(path) for path in self.output.rglob("*")
                                        if path.is_file() and path.name != "manifest.json"}}
        write_json(self.output / "manifest.json", manifest)
        return manifest

    def brief(self):
        counts = Counter(row["status"] for row in self.values)
        title = "# PINN4SOH 结果图表简报" if not self.data.synthetic else "# 合成布局验证，非实验结果"
        text = [title, "", "本目录由独立审计报告与导出证据重绘，未运行训练或重新计算指标。",
                "published 与 paperloss 分栏输出，CNN/MLP 使用同一份已审计基线。",
                "", "- 主图统计口径：真实电池等权宏平均；MAPE 分母为实测 SOH。",
                "- 误差线：十次完整重复的均值与样本标准差（ddof=1）。",
                "- 曲线：事先固定 repeat 1，均匀抽样显示，横轴为保留测试点顺序。",
                "- 完整数值格：" + str(counts.get("complete", 0)) + "；待完成或不匹配格：" + str(sum(value for key, value in counts.items() if key != "complete")) + "。",
                "- Table 4 分开列出作者权重与独立复训权重、微调、source-only 及 target-only n=1/2。",
                "- 作者权重 source-only 可重复使用相同检查点，SD=0 不代表独立预训练稳定性。",
                "- 缺失或不足十次的配置保留 pending，不以零值补齐；结果不预设 PINN 优势。",
                "", "PNG/PDF 位于 figures 与 tables；各表 CSV 保留观测值、样本标准差、论文值及差值。",
                "几何检查记录在 figure_qa.json，输入与输出哈希记录在 manifest.json。", ""]
        (self.output / "结果图表简报.md").write_text("\n".join(text), encoding="utf-8")


def format_number(value, signed=False, scientific=False):
    if value is None:
        return "pending"
    return format(value, "+.2e" if signed and scientific else ".2e" if scientific else "+.4f" if signed else ".4f")


def format_cell(cell):
    if cell["observed"] is None:
        return "pending"
    return format_number(cell["observed"]) + "\n(" + format_number(cell.get("sample_sd_ddof1")) + ")"


def battery_matches(path, published):
    value = path.replace("\\", "/")
    stem = value.rsplit("/", 1)[-1].removesuffix(".csv")
    expected = str(published["battery"])
    if published["group"].startswith("XJTU/"):
        return stem.endswith("_" + expected)
    if stem != expected:
        return False
    if published["group"] == "MIT" and published.get("date"):
        year, month, day = (int(part) for part in str(published["date"]).split("/"))
        return ("/" + str(year) + "-" + str(month).zfill(2) + "-" + str(day).zfill(2) + "/") in value
    return True


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--evidence", type=Path)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--references", type=Path)
    parser.add_argument("--efficiency", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--lanes", nargs="+", choices=list(LANES), default=list(LANES))
    parser.add_argument("--protocol-id", default="author_current")
    parser.add_argument("--validation-protocol", default="author_sample_split")
    parser.add_argument("--transfer-design", default="paper_fixed12")
    options = parser.parse_args(argv)
    if options.report is None and options.evidence is None:
        parser.error("--report and/or --evidence is required")
    for source in [options.report, options.evidence]:
        if source and (options.output.resolve() == source.resolve() or source.resolve() in options.output.resolve().parents):
            raise ValueError("Plot output must be separate from report/evidence inputs")
    if options.output.exists() and any(options.output.iterdir()):
        raise ValueError("Use a new or empty plot output directory")
    apply_figure_style()
    manifest = Renderer(options).render()
    print(json.dumps({key: manifest[key] for key in ["synthetic_layout_fixture", "metric_cell_counts", "ambiguous_group_count", "all_text_geometry_passed"]}))
    return manifest


if __name__ == "__main__":
    main()
