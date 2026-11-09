"""Effectiveness reporting: fills the LaTeX effectiveness tables from the CSVs.

For every table in config.yml -> reporting.effectiveness.tables (list):
  1. ensure the .tex skeleton exists (generate one from config when missing)
  2. empty_table(): rewrite the .tex keeping rows/columns/headers but stripping
     all value cells
  3. fill_table(): walk the .tex blocks, match each dataset block to its config
     dataset and each model row to the CSV row, and write formatted values:
       - digits_after_decimal
       - bold the better number of each pair in bold_values ("X vs Y, ...")
       - bold_red in pairs of bold_red_values: best > base -> red bold,
         best == base -> plain bold, best < base -> untouched
       - optional star on the best nn/plm model per dataset (star_indicator_method)
CSVs come from reporting_effectiveness.py (wandb online data).
"""

import csv
import re

import yaml
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# dataset-name token -> text used in the LaTeX block headers
BLOCK_NAME_ALIASES = {"duben": "deepurlbench", "dibhran": "dibhran"}

PRETTY_NAMES = {
    "threatfox": "ThreatFox",
    "macformer": "MACFormer",
    "deepurlbench": "DeepURLBench",
    "dibhran": "DiBHran",
}

STAR = r"$^{\star}$"
MISSING = "--"
RUN_ORDER = ["baseline", "MIAFF_base", "MIAFF_best_macro_f1"]

METRIC_TEX_NAMES = {
    "macro_f1": "Macro-F1",
    "binary_f1": "Binary-F1",
    "weighted_f1": "Weighted-F1",
    "accuracy": "Accuracy",
    "balanced_accuracy": "Balanced Acc.",
    "precision": "Precision",
    "recall": "Recall",
    "mcc": "MCC",
    "ap": "AP",
    "auroc": "AUROC",
}


def load_values(config, table):
    """(model, dataset, run_name) -> {metric: value} from the effectiveness CSV."""
    eff = config["reporting"]["effectiveness"]
    path = ROOT / eff["output"]["file"].replace("{table.title}", table["title"])
    values = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            model, ds = row["model"], row["dataset"]
            model = model.split(":", 1)[-1]
            for col, val in row.items():
                m = re.match(r"(.+)_(baseline|MIAFF_base|MIAFF_best_macro_f1)$", col)
                if not m:
                    continue
                values.setdefault((model, ds, m.group(2)), {})[m.group(1)] = val
    return values


def dataset_of_block(table, block_words):
    tokens = {re.sub(r"\W", "", w.lower()) for w in block_words}
    for ds in table["datasets"]:
        alias = BLOCK_NAME_ALIASES.get(ds["name"].split("_")[0], ds["name"].split("_")[0])
        if alias in tokens or ds["name"].split("_")[0] in tokens:
            return ds["name"]
    return None


def split_cells(line):
    parts = line.split("&")
    tail = parts[-1].rsplit(r"\\", 1)
    cells = parts[:-1] + [tail[0]]
    ending = (r" \\" + tail[1]) if len(tail) > 1 and tail[1].strip() == "" else r" \\"
    return cells, ending


FORMAT_COMMAND = re.compile(r"\\textcolor\{[^}]*\}|\\textbf\{[^}]*\}|\$\^\{?\\star\}?\$")


def is_value_cell(cell):
    stripped = FORMAT_COMMAND.sub(" ", cell)
    stripped = stripped.replace("{", "").replace("}", "").replace("$", "").strip()
    if stripped == "":
        return True
    return bool(re.fullmatch(r"(--|[\d.]+)", stripped))


def is_data_row(line):
    if "&" not in line or "\\multicolumn" in line or "\\textbf{Method}" in line:
        return False
    cells, _ = split_cells(line)
    return all(is_value_cell(c) for c in cells[1:])


def is_subheader_row(line):
    """The ' &  & base & best & ...' third header line (or its blanked remnant)."""
    if "&" not in line or "\\multicolumn" in line or "\\textbf{Method}" in line:
        return False
    cells, _ = split_cells(line)
    return not cells[0].strip()


def parse_metric_groups(lines):
    """Map tex metric header name (e.g. 'Macro-F1') -> cell index of its first column."""
    groups = {}
    for line in lines:
        spans = re.findall(r"\\multicolumn\{3\}\{c\}\{\\textbf\{([^}]+)\}", line)
        for k, name in enumerate(spans):
            groups[name] = 1 + 3 * k
    return groups


def tex_metric_name(metric):
    return metric.replace("_", "-")


def full_metric_positions(lines):
    """'Macro-F1' -> 'macro_f1' : first cell index of the group."""
    return {tex_metric_name(k).replace("-", "_").lower(): v
            for k, v in parse_metric_groups(lines).items()}


def generate_skeleton(config, table):
    """Generate a .tex skeleton (empty values) mirroring the Malware_Binary layout."""
    metrics = table["metrics"]
    n = len(metrics)
    span = 1 + 3 * n

    tex = [
        r"\begin{table}[!t]",
        r"    \centering",
        rf"    \caption{{{table['title'].replace('_', ' ')} across benchmarks. Bold indicates the better result.}}",
        rf"    \label{{tab:{table['title'].lower()}_effectiveness}}",
        r"    \scriptsize",
        r"    \setlength{\tabcolsep}{1.5pt}",
        r"    \renewcommand{\arraystretch}{0.88}",
        rf"    \begin{{tabular}}{{@{{}}l*{{{n}}}{{ccc}}@{{}}}}",
        r"    \toprule",
        "     & " + " & ".join(
            rf"\multicolumn{{3}}{{c}}{{\textbf{{{METRIC_TEX_NAMES.get(m, m)}}}}}" for m in metrics
        ) + " \\\\",
        "    " + " ".join(rf"\cmidrule(lr){{{2 + 3 * k}-{4 + 3 * k}}}" for k in range(n)),
        "    \\textbf{Method} & " + " & ".join(
            r"\textbf{Baseline} & \multicolumn{2}{c}{\textbf{MIAFF}}" for _ in metrics
        ) + " \\\\",
        "    " + " ".join(rf"\cmidrule(lr){{{3 + 3 * k}-{4 + 3 * k}}}" for k in range(n)),
        "     " + " & ".join(
            cells for cells in ([""]
            + [c for _ in range(n) for c in ([" ", r"\textbf{base}", r"\textbf{best}"])])
            ) + " \\\\",
        "    \\midrule",
    ]

    scenarios = config.get("scenarios", {})
    scenario_of_ds = {ds["name"]: sc_name for sc_name, sc in scenarios.items()
                      for ds in sc.get("datasets", [])}

    first_block = True
    for ds in table["datasets"]:
        sc_name = scenario_of_ds.get(ds["name"], "")
        token = ds["name"].split("_")[0]
        pretty = BLOCK_NAME_ALIASES.get(token, token)
        pretty = PRETTY_NAMES.get(pretty, pretty[0].upper() + pretty[1:])
        if ds["name"].endswith("family"):
            word = "Malware Family"
        else:
            word = "Phishing" if "phishing" in sc_name else "Malware"
        if not first_block:
            tex.append("    \\midrule")
            tex.append("    \\addlinespace[2pt]")
        else:
            first_block = False
        tex.append(rf"    \multicolumn{{10}}{{@{{}}l}}{{\textbf{{{pretty} {word}}}}} \\")
        tex.append("    \\addlinespace[1pt]")
        for m in table["models"]["nn"]:
            cells = [m["name"]] + [""] * (3 * n)
            tex.append("    " + " & ".join(c.strip() for c in cells).rstrip() + " \\\\")
        tex.append(rf"    \cmidrule(lr){{1-{span}}}")
        for m in table["models"]["plm"]:
            cells = [m["name"]] + [""] * (3 * n)
            tex.append("    " + " & ".join(c.strip() for c in cells).rstrip() + " \\\\")

    tex.append("    \\bottomrule")
    tex.append("    \\end{tabular}")
    tex.append("    \\end{table}")
    return "\n".join(tex) + "\n"


def empty_table(lines):
    out = []
    changed = 0
    for line in lines:
        if is_data_row(line) and not is_subheader_row(line):
            indent = line[: len(line) - len(line.lstrip())]
            cells, ending = split_cells(line)
            for i, c in enumerate(cells[1:], start=1):
                cells[i] = " "
            line = indent + " & ".join(c.strip() for c in cells).rstrip() + ending
            changed += 1
        out.append(line)
    return out, changed


def fmt(value, digits):
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return None


def bold_cell(text):
    return r"\textbf{" + text + "}"


def bold_red_cell(text):
    return r"\textcolor{red}{\textbf{" + text + "}}"


def parse_pairs(spec):
    """'baseline vs MIAFF_base, baseline vs best, if same ...' -> [(a, b), ...]."""
    pairs = []
    for seg in (spec or "").split(","):
        m = re.fullmatch(r"\s*(\w+)\s+vs\s+(\w+)\s*", seg)
        if m:
            pairs.append((m.group(1), m.group(2)))
    return pairs


def cell_idx(run, pos):
    return pos + RUN_ORDER.index(run)


def fill_table(config, table, lines, values):
    details = config["reporting"]["effectiveness"]["output"]["latex_table_details"]
    digits = details.get("digits_after_decimal", 3)
    bold_pairs = parse_pairs(details["bold_values"]) if details.get("bold_indicates_better", True) else []
    red_pairs = parse_pairs(details["bold_red_values"]) if details.get("bold_red_indicates_better", False) else []
    metric_names = table["metrics"]

    group_pos = full_metric_positions(lines)  # metric -> first cell index
    # restore the 'base / best' sub-header line if it was emptied by a previous run
    for i, line in enumerate(lines):
        if is_subheader_row(line):
            if "base" not in line:
                cells, _ = split_cells(line)
                for _, pos in group_pos.items():
                    cells[pos] = " "
                    cells[pos + 1] = r"\textbf{base}"
                    cells[pos + 2] = r"\textbf{best}"
                indent = line[: len(line) - len(line.lstrip())]
                lines[i] = indent + " & ".join(c.strip() for c in cells).rstrip() + " \\\\"
            continue

    current_ds = None
    block_models = {}  # ds -> [(line_idx, model)]
    for i, line in enumerate(lines):
        m = re.search(r"\\multicolumn\{10\}\{@\{\}l\}\{\\textbf\{(.+?)\}\}", line)
        if m:
            current_ds = dataset_of_block(table, m.group(1).split())
        elif is_data_row(line):
            model_name = split_cells(line)[0][0].strip()
            block_models.setdefault(current_ds, []).append((i, model_name))

    filled = 0
    for ds, rows in block_models.items():
        if ds is None:
            continue
        # star: best nn + best plm per dataset, scored by the first metric of the best/base run
        star_cls = {}
        if details.get("star_indicator_method", False):
            for group_name, models in table["models"].items():
                best_model, best_score = None, float("-inf")
                for m in models:
                    v = (values.get((m["name"], ds, "MIAFF_best_macro_f1"), {}).get(metric_names[0])
                         or values.get((m["name"], ds, "MIAFF_base"), {}).get(metric_names[0]))
                    try:
                        v = float(v)
                    except (TypeError, ValueError):
                        continue
                    if v > best_score:
                        best_model, best_score = m["name"], v
                star_cls[best_model] = group_name

        for line_idx, _model_hint in rows:
            cells, ending = split_cells(lines[line_idx])
            model = re.sub(r"\s*\$\^\{?\s*\\star\s*\}?\$\s*$", "", cells[0].strip()).strip()
            if model in star_cls:
                cells[0] = model + " " + STAR
            else:
                cells[0] = model
            for metric, pos in group_pos.items():
                if metric not in metric_names:
                    continue
                triple = {}
                for run in RUN_ORDER:
                    v = values.get((model, ds, run), {}).get(metric)
                    triple[run] = fmt(v, digits) if v not in (None, "") else None
                cells[pos] = triple["baseline"] if triple["baseline"] else MISSING
                for run in ("MIAFF_base", "MIAFF_best_macro_f1"):
                    cells[cell_idx(run, pos)] = triple[run] if triple[run] is not None else MISSING

                def pair_numbers(a, b):
                    try:
                        return float(triple[a]), float(triple[b])
                    except (TypeError, KeyError, ValueError):
                        return None

                # bold pass: bold the better value of each pair
                for a, b in bold_pairs:
                    nums = pair_numbers(a, b)
                    if nums and nums[1] > nums[0]:
                        cells[cell_idx(b, pos)] = bold_cell(triple[b])

                # bold-red pass: best vs base -> red if strictly better,
                # keep plain bold if equal, leave untouched if worse
                for a, b in red_pairs:
                    nums = pair_numbers(a, b)
                    if not nums:
                        continue
                    if nums[1] > nums[0]:
                        cells[cell_idx(b, pos)] = bold_red_cell(triple[b])
                    elif nums[1] == nums[0]:
                        cells[cell_idx(b, pos)] = bold_cell(triple[b])
            indent = lines[line_idx][: len(lines[line_idx]) - len(lines[line_idx].lstrip())]
            lines[line_idx] = indent + " & ".join(c.strip() for c in cells).rstrip() + ending
            filled += 1
    return lines, filled


def process_table(config, table):
    eff = config["reporting"]["effectiveness"]
    tex_path = ROOT / eff["output"]["latex_table"].replace("{table.title}", table["title"])
    if not tex_path.exists():
        tex_path.parent.mkdir(parents=True, exist_ok=True)
        tex_path.write_text(generate_skeleton(config, table))
        print(f"Generated skeleton {tex_path.name}")

    lines = tex_path.read_text().splitlines()
    lines, emptied = empty_table(lines)
    print(f"[{table['title']}] Emptied {emptied} data rows in {tex_path.name}")

    values = load_values(config, table)
    lines, filled = fill_table(config, table, lines, values)
    print(f"[{table['title']}] Filled {filled} model rows from CSV")

    tex_path.write_text("\n".join(lines) + "\n")


def main():
    config = yaml.safe_load((ROOT / "config.yml").read_text())
    eff = config["reporting"]["effectiveness"]
    full_tables = eff["tables"] if isinstance(eff["tables"], list) else [eff["tables"]]
    for table in full_tables:
        process_table(config, table)


if __name__ == "__main__":
    main()
