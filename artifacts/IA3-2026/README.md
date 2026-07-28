# IA3-2026 GraphSAGE Artifact

This directory contains the records used for the validation-accuracy,
training-loop-throughput, cache, and activation-input analyses reported in
_GraphSAGE Training on Cerebras CS-3: Execution Characteristics of a High-Level
Workflow_.

The artifact covers the reported records. The repository's GraphSAGE
implementation and configuration profiles cover fresh executions.

## Contents

- `logs/`: twelve sanitized H100 and CS-3 run logs, one per reported system,
  dataset, and measurement condition.
- `configs/`: the six H100 configuration files at the recorded source
  revision and the six resolved CS-3 `trainer_params.yaml` files.
- `wio/`: the Grafana/Loki request and response chunks captured for the two
  uncached CS-3 jobs on June 17, 2026. These are the complete
  `rt_iter_perf` query responses, serialized without row projection. The
  reproduction script selects activation rows and reads the raw `timestamp`,
  `replica_id`, `it`, and `is` fields directly from those responses. The
  normalization below is an analysis convention; the public records do not
  independently define the semantics of `is`.
- `MANIFEST.json`: condition-to-record mapping, job IDs, source and public
  hashes, WIO normalization, and row counts.
- `reported_metrics.json`: values reported by the paper and compared with
  recomputed results.
- `scripts/`: numeric reproduction, plotting, and preparation tools.

## Reported-run provenance

Paths in the H100 and CS-3 record columns are relative to `logs/`.

| Dataset | Measurement | H100 record | CS-3 record | CS-3 execute job |
| --- | --- | --- | --- | --- |
| ogbn-arxiv | validation accuracy | `arxiv_graphsage_1gpu_accuracy_eval.log` | `arxiv_graphsage_wse_accuracy_eval.log` | `wsjob-mswfcbcppuzmiuapx9psr6` |
| ogbn-products | validation accuracy | `products_graphsage_1gpu_accuracy_eval.log` | `products_graphsage_wse_accuracy_eval.log` | `wsjob-ndve7gpbywsrd33jccuxnb` |
| ogbn-arxiv | cached throughput | `arxiv_graphsage_1gpu_cache.log` | `arxiv_graphsage_wse_cache.log` | `wsjob-v7j6tkhq7nfcrdburs7sui` |
| ogbn-products | cached throughput | `products_graphsage_1gpu_cache.log` | `products_graphsage_wse_cache.log` | `wsjob-kqpue4uyt2s3wobgpkpw2c` |
| ogbn-arxiv | uncached throughput and WIO | `arxiv_graphsage_1gpu_not.log` | `arxiv_graphsage_wse_not.log` | `wsjob-dw436fv2uoch2jqzzmfvco` |
| ogbn-products | uncached throughput and WIO | `products_graphsage_1gpu_not.log` | `products_graphsage_wse_not.log` | `wsjob-s4ghbzereuamjqqdqtzg7w` |

## Reproduce reported metrics

From the repository root:

```bash
python artifacts/IA3-2026/scripts/reproduce_metrics.py --check
```

The command recomputes:

- final validation accuracy from the last evaluation record;
- H100 throughput from the final cumulative throughput record;
- CS-3 throughput from the first and last training-loop progress timestamps;
- the median of `is / 83 / it` over all complete activation records for each
  uncached CS-3 job.

The `--check` option is a reported-value consistency check: it compares these
recomputed values with `reported_metrics.json` at the precision used in the
paper: four decimal places for validation accuracy, the nearest sample/s for
throughput, and one decimal place for the normalized fraction in percent.

The runtime reported 103 allocated activation WIOs and a saturation capacity
of 83. The analysis convention uses 83 as the WIO count in `is / 83 / it`.
The resulting normalized value is reported only as a supporting input-path
signal, not as a wall-clock stall fraction or per-stage timing.

The arxiv capture contains 4,824 `rt_iter_perf` rows, including 4,221
activation rows. Of those, 4,179 have both `it` and `is` and enter the median.
The products capture contains 9,624 rows, including 8,421 activation rows and
8,400 complete rows used by the median. The remaining 42 and 21 startup rows
do not contain `is`; they remain in the raw responses. Every request was split
into a five-minute interval, and no response reached its 5,000-line limit.
Each `query_manifest.json` records the exact query, interval, request and
response filenames, row counts, and hashes.

## Regenerate figures

Install the plotting dependency from `requirements.txt`, then run:

```bash
python artifacts/IA3-2026/scripts/plot_accuracy.py \
  artifacts/IA3-2026/logs \
  --steps-only \
  --include-log arxiv_graphsage_1gpu_accuracy_eval.log \
  --include-log arxiv_graphsage_wse_accuracy_eval.log \
  --output /tmp/ia3-arxiv-accuracy.png

python artifacts/IA3-2026/scripts/plot_accuracy.py \
  artifacts/IA3-2026/logs \
  --steps-only \
  --include-log products_graphsage_1gpu_accuracy_eval.log \
  --include-log products_graphsage_wse_accuracy_eval.log \
  --output /tmp/ia3-products-accuracy.png

python artifacts/IA3-2026/scripts/plot_breakdown.py \
  artifacts/IA3-2026/logs \
  --exclude-log '*wse*' \
  --output /tmp/ia3-result.png
```

The last command writes `/tmp/ia3-result_breakdown.png`.

## Provenance and sanitization

CS-3 execute job IDs are retained because they identify the reported runs.
The H100 logs identify source revision
`8d1457914cfb8edfad40734500c663a483d13d90`.

The public records replace operational details that are not needed to
recompute the results:

- ALCF user and absolute repository paths;
- compute-node hostnames and PBS job identifiers;
- internal Cerebras dashboard URLs, workdirs, and namespace.

These replacements apply to the run logs and resolved configurations. The WIO
request and response chunks are copied byte for byte from the June 17 capture;
they contain neither session credentials nor the request headers used to
authenticate to Grafana. The preparation script reconstructs the original
captured JSONL from the response DataFrames and requires its SHA-256 hash to
match before writing `MANIFEST.json`.
