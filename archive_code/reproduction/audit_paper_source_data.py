"""Read official source workbook and audit its published numerical summaries."""

import csv
import hashlib
import json
from pathlib import Path
import re
import statistics

import openpyxl
from openpyxl.utils import get_column_letter


ROOT = Path(__file__).resolve().parents[1] / "outputs" / "full_reproduction"
OUT = ROOT / "source_data"


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def group_name(value):
    value = value.replace(" dataset", "")
    if " batch " in value:
        dataset, batch = value.split(" batch ")
        return dataset + "/" + batch
    return value


def six_number_rows(text):
    pattern = re.compile(r"^\s*(.*?)\s+" + r"\s+".join([r"(\d+\.\d{4})"] * 6) + r"\s*$")
    result = []
    for line in text.splitlines():
        match = pattern.match(line)
        if match:
            result.append((match.group(1).strip(), [float(value) for value in match.groups()[1:]]))
    return result


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    source = ROOT / "source_data.xlsx"
    source_sha = hashlib.sha256(source.read_bytes()).hexdigest()
    book = openpyxl.load_workbook(source, read_only=True, data_only=True)
    normalized = {"source_workbook": str(source), "source_sha256": source_sha,
                  "units": "SOH fraction; MAPE relative fraction", "sheets": {}}
    inventory = []
    repeat_rows, small_rows, s5_rows = [], [], []
    for sheet in book:
        header = [cell.value for cell in next(sheet.iter_rows(min_row=1, max_row=1))]
        rows = []
        counters = {}
        for row_number, values in enumerate(sheet.iter_rows(min_row=2, values_only=True), start=2):
            if sheet.title == "Figure 4a":
                for column in range(0, len(header), 2):
                    truth, prediction = values[column:column + 2]
                    if truth is None and prediction is None:
                        continue
                    if truth is None or prediction is None:
                        raise ValueError("Incomplete truth/prediction pair")
                    group = group_name(header[column].replace(" true", ""))
                    rows.append({"group": group, "sample_index": row_number - 1,
                                 "true": truth, "prediction": prediction,
                                 "true_cell": f"'{sheet.title}'!{get_column_letter(column + 1)}{row_number}",
                                 "prediction_cell": f"'{sheet.title}'!{get_column_letter(column + 2)}{row_number}"})
            elif sheet.title == "Figure 4b":
                model, metric = values[:2]
                model = "PINN" if model == "Ours" else model
                key = (model, metric)
                counters[key] = counters.get(key, 0) + 1
                for column in range(2, len(header)):
                    rows.append({"group": group_name(header[column]), "model": model,
                                 "metric": metric, "repeat": counters[key], "value": values[column],
                                 "cell": f"'{sheet.title}'!{get_column_letter(column + 1)}{row_number}"})
                repeat_rows.extend(rows[-(len(header) - 2):])
            elif sheet.title in {"Figure 5a", "Figure 5b"}:
                repeat, value, model, n_batteries = values
                row = {"group": "XJTU/1" if sheet.title == "Figure 5a" else "HUST",
                       "model": "PINN" if model == "Ours" else model,
                       "train_batteries": n_batteries, "repeat": repeat, "metric": "RMSE",
                       "value": value, "cell": f"'{sheet.title}'!B{row_number}"}
                rows.append(row)
                small_rows.append(row)
            elif sheet.title == "Figure S5":
                n_batteries, model, repeat = values[:3]
                for column in range(3, len(header)):
                    row = {"group": "XJTU/" + str(column - 2),
                           "model": "PINN" if model == "Ours" else model,
                           "train_batteries": n_batteries, "repeat": repeat,
                           "metric": "RMSE", "value": values[column],
                           "cell": f"'{sheet.title}'!{get_column_letter(column + 1)}{row_number}"}
                    rows.append(row)
                    s5_rows.append(row)
            else:
                raise ValueError("Unexpected source sheet " + sheet.title)
        filename = sheet.title.replace(" ", "_").lower() + ".csv"
        write_csv(OUT / filename, rows)
        normalized["sheets"][sheet.title] = {"source_header": header, "records": rows}
        inventory.append({"sheet": sheet.title, "records": len(rows), "csv": filename})
    book.close()
    write_json(OUT / "source_data_normalized.json", normalized)

    supplement = (ROOT / "supplementary" / "paper_text.txt").read_text(encoding="utf-8")
    pages = {int(page): text for page, text in re.findall(
        r"PDF PAGE (\d+) =+\n(.*?)(?=\n=+ PDF PAGE|\Z)", supplement, flags=re.S)}
    per_battery = []
    for page, table, dataset, batches in [
        (8, "S3", "XJTU", [(1, 2), (2, 3), (3, 2), (4, 2), (5, 2), (6, 2)]),
        (9, "S4", "TJU", [(1, 13), (2, 11), (3, 2)]),
        (10, "S5", "HUST", [(None, 20)]),
        (11, "S6", "MIT", [(None, 23)])]:
        raw_rows = six_number_rows(pages[page])
        assert len(raw_rows) == sum(count for _, count in batches), (table, len(raw_rows))
        cursor = 0
        for batch, count in batches:
            for local_index in range(count):
                identity, numbers = raw_rows[cursor]
                cursor += 1
                identity = re.sub(r"^\d\s+(?=(?:battery|CY))", "", identity)
                identity = re.sub(r"^201[78]/\d+/\d+\s+", "", identity)
                identity = identity.replace("–", "-").replace("−", "-")
                date = None
                if dataset == "MIT":
                    date = "2017/5/12" if local_index < 9 else "2017/6/30" if local_index < 17 else "2018/4/12"
                row = {"group": dataset + ("/" + str(batch) if batch else ""),
                       "battery": identity, "date": date,
                       "source": f"Supplement PDF p{page}, Table {table}, row {cursor}"}
                row.update(dict(zip(["PINN_MAPE", "PINN_RMSE", "MLP_MAPE", "MLP_RMSE", "CNN_MAPE", "CNN_RMSE"], numbers)))
                per_battery.append(row)
    write_csv(OUT / "paper_per_battery_tables.csv", per_battery)
    write_json(OUT / "paper_per_battery_tables.json", per_battery)

    table_s7 = {}
    raw_s7 = six_number_rows(pages[12])
    assert len(raw_s7) == 24, len(raw_s7)
    for index, (_, numbers) in enumerate(raw_s7):
        table_s7[f"XJTU/{index // 4 + 1}/n{index % 4 + 1}"] = numbers
    write_json(OUT / "paper_table_s7.json", {
        "source": "Supplement PDF p12, Table S7", "units": "fraction",
        "columns": ["PINN_MAPE", "PINN_RMSE", "MLP_MAPE", "MLP_RMSE", "CNN_MAPE", "CNN_RMSE"],
        "rows": table_s7})

    protocol = json.loads((ROOT / "paper_protocol.json").read_text(encoding="utf-8"))
    comparisons, summaries = [], []
    def summarize_rows(records, family, table_rows, small=False):
        keys = sorted({(row["group"], row["model"], row.get("train_batteries"), row["metric"]) for row in records})
        for group, model, n, metric in keys:
            selected = [row for row in records if (row["group"], row["model"], row.get("train_batteries"), row["metric"]) == (group, model, n, metric)]
            assert len(selected) == 10 and len({row["repeat"] for row in selected}) == 10, (family, group, model, n, metric)
            values = [row["value"] for row in selected]
            entry = {"family": family, "group": group, "model": model, "train_batteries": n,
                     "metric": metric, "n_runs": 10, "mean": statistics.mean(values),
                     "std_ddof0": statistics.pstdev(values), "std_ddof1": statistics.stdev(values),
                     "source_cells": [row["cell"] for row in selected]}
            summaries.append(entry)
            if metric not in {"MAPE", "RMSE"}:
                continue
            key = group + (f"/n{n}" if small else "")
            columns = protocol["published_table_2"]["column_order"]
            published = table_rows[key][columns.index(model + "_" + metric)]
            difference = entry["mean"] - published
            comparisons.append({**entry, "published_rounded_mean": published,
                                "difference": difference, "within_rounding_tolerance": abs(difference) <= 0.0000500001})
    summarize_rows(repeat_rows, "Figure4b_vs_Table2", protocol["published_table_2"]["rows"])
    summarize_rows(small_rows, "Figure5_vs_Table3", protocol["published_table_3"]["rows"], True)
    summarize_rows(s5_rows, "FigureS5_vs_TableS7", table_s7, True)

    battery_averages = []
    columns = protocol["published_table_2"]["column_order"]
    for group, numbers in protocol["published_table_2"]["rows"].items():
        selected = [row for row in per_battery if row["group"] == group]
        for column, published in zip(columns, numbers):
            mean = statistics.mean(row[column] for row in selected)
            battery_averages.append({"group": group, "metric": column, "n_batteries": len(selected),
                                     "mean_of_rounded_battery_means": mean,
                                     "Table2_rounded_mean": published,
                                     "difference": mean - published,
                                     "compatible_with_two_stage_rounding": abs(mean - published) <= 0.0001000001})
    overlap = []
    for row in small_rows:
        if row["group"] != "XJTU/1":
            continue
        match = next(item for item in s5_rows if all(item[key] == row[key]
                     for key in ["group", "model", "train_batteries", "repeat"]))
        overlap.append(row["value"] == match["value"])
    write_csv(OUT / "source_repeat_summaries.csv", summaries)
    write_csv(OUT / "published_mean_comparisons.csv", comparisons)
    write_csv(OUT / "battery_double_average_checks.csv", battery_averages)
    audit = {"source_workbook_sha256": source_sha, "sheets": inventory,
             "per_repeat_groups_all_have_ten_distinct_repeats": True,
             "published_mean_comparisons": len(comparisons),
             "published_mean_matches_to_rounding": sum(row["within_rounding_tolerance"] for row in comparisons),
             "published_mean_mismatches": [row for row in comparisons if not row["within_rounding_tolerance"]],
             "double_average_comparisons": len(battery_averages),
             "double_average_compatible_with_rounding": sum(row["compatible_with_two_stage_rounding"] for row in battery_averages),
             "double_average_mismatches": [row for row in battery_averages if not row["compatible_with_two_stage_rounding"]],
             "Figure5a_vs_FigureS5_batch1_exact_matches": sum(overlap),
             "Figure5a_overlap_values": len(overlap),
             "interpretation_limits": ["Numerical compatibility does not prove a unique aggregation algorithm",
                                        "Source Data does not include Table4 per-repeat transfer results",
                                        "Figure4a predictions are a representative result, not all ten checkpoints",
                                        "SD ddof remains unspecified in paper; both ddof0 and ddof1 are exported"]}
    write_json(OUT / "source_data_audit.json", audit)
    write_json(OUT / "source_repeat_summaries.json", summaries)
    print(json.dumps({key: value for key, value in audit.items() if not key.endswith("mismatches")}, ensure_ascii=False, indent=2))
    if audit["published_mean_mismatches"]:
        print("MEAN_MISMATCHES", json.dumps(audit["published_mean_mismatches"], ensure_ascii=False))


if __name__ == "__main__":
    main()
