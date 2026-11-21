"""Redraw original XJTU curve examples without substituting candidate features.

The cell/cycle selections are fixed before plotting: batch-1 battery-1 cycle 10
for Fig1b, and battery-1 in every batch for Supplementary Fig4.
"""

import argparse
import contextlib
import gc
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import time
import zipfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
from matplotlib.ticker import MaxNLocator, EngFormatter
import numpy as np
import pandas as pd


CELLS = ["2C_battery-1", "3C_battery-1", "R2.5_battery-1", "R3_battery-1", "RW_battery-1", "Sim_satellite_battery-1"]
STYLE = Path("C:/Users/Administrator/.codex/skills/figure-style/kernel.py")


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def figure_check(fig, stem):
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for ax in fig.axes:
        for tick in ax.xaxis.get_major_ticks() + ax.xaxis.get_minor_ticks():
            if not min(ax.get_xlim()) <= tick.get_loc() <= max(ax.get_xlim()):
                tick.label1.set_visible(False)
                tick.label2.set_visible(False)
        for tick in ax.yaxis.get_major_ticks() + ax.yaxis.get_minor_ticks():
            if not min(ax.get_ylim()) <= tick.get_loc() <= max(ax.get_ylim()):
                tick.label1.set_visible(False)
                tick.label2.set_visible(False)
    texts = [(item, item.get_window_extent(renderer)) for item in fig.findobj(matplotlib.text.Text)
             if item.get_text().strip() and item.get_visible()]
    overlaps = []
    outside = []
    for i, (a, first) in enumerate(texts):
        if first.x0 < -1 or first.y0 < -1 or first.x1 > fig.bbox.x1 + 1 or first.y1 > fig.bbox.y1 + 1:
            outside.append(a.get_text())
        for b, second in texts[i + 1:]:
            if first.overlaps(second):
                overlaps.append([a.get_text(), b.get_text()])
    result = dict(overlapping_text_pairs=overlaps, text_outside_figure=outside)
    stem.with_suffix(".layout.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    if overlaps or outside:
        raise ValueError("Figure text overlaps: " + str(result))
    fig.savefig(stem.with_suffix(".png"), dpi=300)
    fig.savefig(stem.with_suffix(".pdf"))
    plt.close(fig)
    return result


def decimate_xy(values, max_points):
    size = len(values)
    indices = np.unique(np.linspace(0, max(0, size - 1), min(size, max_points), dtype=int))
    return np.column_stack([indices, values[indices]])


def add_collection(ax, segments, cycles, norm, cmap):
    collection = LineCollection(segments, cmap=cmap, norm=norm, linewidth=.35, rasterized=True)
    collection.set_array(np.array(cycles))
    ax.add_collection(collection)
    ax.autoscale_view()
    ax.margins(.035)
    return collection


def raw_cell(archive, batch, name, source):
    member = "Battery Dataset/Batch-" + str(batch) + "/" + name + ".mat"
    from resume_boundary_memory_guard import available_gib
    budget = max(3.0, min(4.5, 2.0 + archive.getinfo(member).file_size * 18 / 1024 ** 3))
    while available_gib() < budget:
        print(json.dumps(dict(status="waiting_raw_plot_memory", member=member,
                              available_gib=available_gib(), required_gib=budget)), flush=True)
        time.sleep(10)
    payload = archive.read(member)
    with tempfile.TemporaryDirectory(prefix="pinn_curve_") as temporary:
        target = Path(temporary) / (name + ".mat")
        target.write_bytes(payload)
        with contextlib.redirect_stdout(io.StringIO()):
            battery = source.Battery(target.as_posix())
    identity = dict(batch=batch, battery=name, member=member, crc32=archive.getinfo(member).CRC,
                    uncompressed_bytes=len(payload), sha256=hashlib.sha256(payload).hexdigest())
    return battery, identity


def figure1b(battery, output, style):
    cycle = 10
    v = battery.get_partial_value(cycle, "voltage_V")
    current = battery.get_partial_value(cycle, "current_A")
    time = battery.get_partial_value(cycle, "relative_time_min") * 60
    voltage_mask = (v >= 4.0) & (v <= 4.2) & (current >= 3.9)
    current_mask = (current >= .1) & (current <= .5)
    charge = pd.DataFrame(dict(raw_cycle=cycle, charge_sample_index=np.arange(len(v)), elapsed_seconds=time,
                               voltage_V=v, current_A=current, paper_voltage_window=voltage_mask,
                               paper_current_window=current_mask,
                               supplemental_voltage_4p195_window=voltage_mask & (v <= 4.195)))
    charge.to_csv(output / "figure1b_cycle10_charge_source.csv", index=False)
    fig, axes = plt.subplots(2, 2, figsize=(8.0, 6.1), gridspec_kw=dict(left=.105, right=.87, bottom=.10,
                                                                       top=.91, wspace=.38, hspace=.44))
    norm, cmap = Normalize(1, int(battery.cycle_life)), plt.get_cmap("viridis")
    for ax, values, mask, title, label, limits in [
        (axes[0, 0], v, voltage_mask, "Cycle 10: charge voltage", "Voltage (V)", [4.0, 4.2]),
        (axes[0, 1], current, current_mask, "Cycle 10: charge current", "Current (A)", [.1, .5]),
    ]:
        samples = np.arange(len(values))
        ax.plot(samples, values, color="#6b7280", linewidth=1.0, label="Charge curve")
        ax.plot(samples[mask], values[mask], color="#0072b2", linewidth=1.6, label="Selected window")
        for boundary in limits:
            ax.axhline(boundary, color="#d55e00", ls="--", linewidth=.6)
        ax.set(title=title, xlabel="Charge sample index", ylabel=label)
        ax.margins(.04)
        ax.set_xlim(0, max(samples) * 1.035)
    axes[0, 0].legend(loc="lower right")
    curve_counts, rows = {}, []
    for col, variable, lower, upper, title, label in [
        (0, "voltage_V", 4.0, 4.2, "Selected voltage over cell life", "Voltage (V)"),
        (1, "current_A", .1, .5, "Selected current over cell life", "Current (A)"),
    ]:
        segments, cycles = [], []
        for raw_cycle in range(1, int(battery.cycle_life) + 1):
            values = battery.get_partial_value(raw_cycle, variable)
            amperes = battery.get_partial_value(raw_cycle, "current_A")
            selected = (values >= lower) & (values <= upper)
            if col == 0:
                selected &= amperes >= 3.9
            chosen = values[selected]
            if len(chosen) < 2:
                continue
            segments.append(np.column_stack([np.arange(len(chosen)), chosen]))
            cycles.append(raw_cycle)
            rows.append(dict(curve=variable, raw_cycle=raw_cycle, selected_points=len(chosen),
                             first_value=chosen[0], last_value=chosen[-1]))
        curve_counts[variable] = len(cycles)
        add_collection(axes[1, col], segments, cycles, norm, cmap)
        axes[1, col].set(title=title, xlabel="Selected sample index", ylabel=label)
        axes[1, col].set_xlim(0, max(segment[-1, 0] for segment in segments) * 1.035)
        axes[1, col].set_ylim(lower - .02, upper + .02)
    cax = fig.add_axes([.91, .12, .018, .72])
    fig.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax, label="Physical cycle")
    pd.DataFrame(rows).to_csv(output / "figure1b_full_life_window_counts.csv", index=False)
    for label, ax in zip("abcd", axes.flat):
        style.panel_letter(ax, label, dx=-.24, dy=1.03)
    check = figure_check(fig, output / "figure1b_raw_window_example")
    return dict(selected_battery=CELLS[0], selected_physical_cycle=10,
                full_life_physical_cycles=int(battery.cycle_life), plotted_window_curves=curve_counts,
                voltage_window="4.0<=V<=4.2 V during I>=3.9 A (paper boundary; public reader CC gate)",
                current_window="0.1<=I<=0.5 A during charge stage",
                supplemental_4p195_boundary_used_for_plot=False, layout_check=check)


def figure_s4(archive, output, source, style, max_points):
    fig, axes = plt.subplots(6, 4, figsize=(11.8, 14.6),
                             gridspec_kw=dict(left=.075, right=.985, bottom=.055, top=.92,
                                             wspace=.42, hspace=.49))
    records, identities, capacities = [], [], []
    palette = plt.get_cmap("viridis")
    for batch, name in enumerate(CELLS, 1):
        print(json.dumps(dict(figure="S4", batch=batch, status="loading", battery=name)), flush=True)
        battery, identity = raw_cell(archive, batch, name, source)
        identities.append(identity)
        life = int(battery.cycle_life)
        norm = Normalize(1, life)
        tests = []
        for raw_cycle in range(1, life + 1):
            if "test capacity" in battery.get_one_cycle_description(raw_cycle):
                tests.append(raw_cycle)
        measured = battery.get_capacity()
        interpolate_capacity = batch >= 4 and len(tests) >= 2
        if interpolate_capacity:
            trajectory = battery.get_degradation_trajectory()
            if not np.all(np.isfinite(trajectory)):
                raise ValueError("Nonfinite periodic capacity interpolation in " + name)
        else:
            trajectory = measured
        identity.update(physical_cycles=life, labelled_capacity_tests=len(tests),
                        trajectory_is_interpolated=interpolate_capacity,
                        capacity_basis="Periodic test-capacity interpolation" if interpolate_capacity else "Raw per-cycle discharge capacity")
        for raw_cycle in range(1, life + 1):
            capacities.append(dict(batch=batch, battery=name, physical_cycle=raw_cycle,
                                   measured_discharge_capacity_Ah=measured[raw_cycle - 1],
                                   periodic_full_discharge_capacity_test=raw_cycle in tests,
                                   trajectory_capacity_Ah=trajectory[raw_cycle - 1],
                                   trajectory_is_interpolated=interpolate_capacity))
        for col, variable, ylabel in [(0, "voltage_V", "Voltage (V)"), (1, "current_A", "Current (A)"),
                                      (2, "temperature_C", "Temperature (deg C)")]:
            segments, cycles = [], []
            for raw_cycle in range(1, life + 1):
                values = battery.get_value(raw_cycle, variable)
                if len(values) < 2:
                    continue
                segments.append(decimate_xy(values, max_points))
                cycles.append(raw_cycle)
                records.append(dict(batch=batch, battery=name, raw_cycle=raw_cycle, variable=variable,
                                    raw_sample_points=len(values), plotted_sample_points=len(segments[-1])))
            ax = axes[batch - 1, col]
            add_collection(ax, segments, cycles, norm, palette)
            ax.set_xlim(0, max(segment[-1, 0] for segment in segments) * 1.035)
            ax.xaxis.set_major_locator(MaxNLocator(4))
            ax.xaxis.set_major_formatter(EngFormatter(places=0, sep=""))
            ax.set_ylabel(ylabel)
            if batch == 6:
                ax.set_xlabel("Within-cycle sample index")
            if batch == 1:
                ax.set_title(["Voltage curves", "Current curves", "Temperature curves"][col])
        ax = axes[batch - 1, 3]
        x = np.arange(1, life + 1)
        ax.scatter(x, measured, c=x, cmap=palette, norm=norm, s=1.3, linewidths=0,
                   rasterized=True, label="Per-cycle discharge")
        if interpolate_capacity:
            ax.plot(x, trajectory, color="#d55e00", lw=1.0, label="Interpolated trajectory")
            ax.scatter(tests, measured[np.array(tests) - 1], facecolor="white", edgecolor="#d55e00",
                       s=9, linewidth=.5, zorder=4, label="Capacity tests")
        ax.set_ylabel("Discharge capacity (Ah)")
        ax.margins(.04)
        if batch == 1:
            ax.set_title("Measured / interpolated capacity")
        if batch == 4:
            ax.legend(loc="lower left")
        if batch == 6:
            ax.set_xlabel("Physical cycle")
        axes[batch - 1, 0].text(-.29, 1.24, "Batch " + str(batch), transform=axes[batch - 1, 0].transAxes,
                                 fontweight="bold", fontsize=9, ha="left", va="bottom")
        del battery
        gc.collect()
        print(json.dumps(dict(figure="S4", batch=batch, status="curves_added", physical_cycles=life)), flush=True)
    pd.DataFrame(records).to_csv(output / "figureS4_display_sampling.csv", index=False)
    pd.DataFrame(capacities).to_csv(output / "figureS4_capacity_source.csv", index=False)
    cax = fig.add_axes([.39, .966, .30, .008])
    colorbar = fig.colorbar(matplotlib.cm.ScalarMappable(norm=Normalize(0, 1), cmap=palette),
                           cax=cax, orientation="horizontal", ticks=[0, 1])
    colorbar.ax.set_xticklabels(["Early", "Late"])
    colorbar.ax.set_xlabel("Relative cycle within each cell")
    colorbar.ax.xaxis.set_label_position("top")
    check = figure_check(fig, output / "figureS4_raw_six_batch_curves")
    return dict(cells=identities, all_physical_cycles_plotted=True,
                display_sampling="Uniform index sampling, including endpoints, max " + str(max_points) + " points per variable per cycle",
                capacity_interpolation="Public reader cubic interpolation of labelled test-capacity cycles with linear fallback, batches 4-6. Batch3 has only an initial labelled test and is fully discharged per Note1.3; raw discharge capacity is retained.",
                layout_check=check)


def main(options):
    options.output.mkdir(parents=True, exist_ok=True)
    root = options.root.resolve()
    archive_path = root.parent / "raw_data" / "XJTU" / "Battery Dataset.zip"
    archive_status = json.loads(archive_path.with_suffix(".status.json").read_text(encoding="utf-8"))
    if archive_status["status"] != "verified" or not archive_status["official_checksum_passed"]:
        raise ValueError("Original XJTU archive is not checksum verified")
    style = load_module(STYLE, "raw_curve_figure_style")
    style.apply_figure_style(sizes=(9, 8, 7), font="Arial")
    source_path = root / "preprocessing_source" / "XJTUBatteryClass.py"
    source = load_module(source_path, "raw_curve_official_reader")
    with zipfile.ZipFile(archive_path) as archive:
        battery, identity = raw_cell(archive, 1, CELLS[0], source)
        first = figure1b(battery, options.output, style)
        del battery
        gc.collect()
        result = dict(figure1b=first,
                      figureS4=figure_s4(archive, options.output, source, style, options.max_display_points),
                      figure1b_source=identity)
    result.update(archive_sha256=archive_status["sha256"], official_checksum=archive_status["official_checksum"],
                  reader_sha256=hashlib.sha256(source_path.read_bytes()).hexdigest(),
                  script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  figure_style_sha256=hashlib.sha256(STYLE.read_bytes()).hexdigest(),
                  scope="Data-grounded reconstruction of figure concepts; exact original display cell/cycle identity is unreported",
                  paper_results_modified=False)
    (options.output / "RAW_CURVE_FIGURES.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    (options.output / "CAPTIONS.md").write_text(
        "# Original Raw Curve Figures\n\n"
        "Figure1b: XJTU batch-1 battery-1, physical cycle10, fixed before rendering. Upper panels show the complete charge stage; blue points identify the paper voltage/current windows. The voltage panel uses the public reader's I>=3.9A CC gate. Lower panels overlay every valid physical-cycle window with physical cycle as color. Sample index is the within-charge or within-selected-window sample index, not elapsed time. This figure deliberately uses the stated paper4.2V limit; the exploratory4.195V feature-audit boundary is exported as a separate source column and is not substituted in this plot. The original paper's exact example-cell identity is unreported, so this is an independent raw-data illustration, not pixel identity.\n\n"
        "FigureS4: The first battery of each of the six XJTU batches, matching the declared selection in Supplementary FigS4. All physical cycles are drawn and color runs from early (purple) to late (yellow) within each battery. Voltage, current, and temperature use complete cycles, including charge/rest/discharge stages, with uniform within-cycle display sampling only; source counts are retained in figureS4_display_sampling.csv. The long initial capacity-test cycle is deliberately retained; its full sample-index extent differs from the cropped regular-cycle range visible in the original paper. Discharge capacity is plotted for every physical cycle. In batches4-6, incomplete discharges are not full-capacity measurements: white markers identify labelled periodic capacity tests and the orange line is the public reader's interpolated trajectory. Batch3 is described as fully discharged to2.5V in Note1.3 and its first cell has only one labelled capacity test; its raw discharge values are shown without an invented interpolation. This resolves a discrepancy with the broader batches3-6 statement in the S4 caption, and agrees with the absence of a red line on the original S4 batch3 panel. The exported capacity CSV distinguishes each quantity. No laboratory hardware or new cell tests are claimed.\n",
        encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("outputs/full_reproduction/feature_audit"))
    parser.add_argument("--output", type=Path, default=Path("outputs/full_reproduction/feature_audit/raw_curve_figures"))
    parser.add_argument("--max-display-points", type=int, default=1024)
    main(parser.parse_args())
