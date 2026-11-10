# Reproducing ACE on FiNER

This repository supports the FTEC5660 reproducibility report. I reproduced the direction of the FiNER result in the ACE paper using DeepSeek Flash and a smaller held-out sample. I then changed the Generator's response format to return only the four required tags. The model, questions, playbooks, decoding settings, output cap and scorer were shared between the two output formats.

## Results in the report

| Comparison | Control | Treatment | Paired difference |
| --- | --- | --- | --- |
| Empty playbook vs. 100-batch playbook on 64 held-out test batches | 620/768 (80.73%) | 658/768 (85.68%) | +4.95 pp, 95% CI [+0.26, +10.16] |
| Original format vs. four tags on 64 previously unused validation batches | 616/768 (80.21%) | 620/768 (80.73%) | +0.52 pp, 95% CI [−0.52, +1.69] |

Both comparisons use three frozen playbooks (seeds 17, 29 and 43), with four tag positions per batch and per seed. Intervals resample whole batches across the three trials. The paper reports 70.7% for the base model and 78.3% for offline ACE with labels using DeepSeek-V3.1-671B. The model, sample and baseline prompt differ from this reproduction. [ACE paper, Table 2](https://arxiv.org/html/2510.04618v3#S4.T2)

Returning four tags reduced completion tokens from **47,501 to 9,033** (80.98%) and truncated responses from **2 to 0**. The two formats each scored **616/760** on pairs where both responses completed. The estimated cost at peak rates fell from CNY 0.720 to CNY 0.544 using the recorded token counts and cache hits.

## Verify the archived results

Clone or download the repository and run the command from its root with Python 3.12. It uses only the Python standard library and the archived evidence.

```sh
python verify_report.py
```

| Evidence | What it contains |
| --- | --- |
| [Training-size comparison](evidence/training_scale/summary.json) | Scores at 24, 48 and 100 batches, with the frozen playbooks in the trial directories |
| [100-batch confirmation](evidence/confirm_100/summary.json) | Paired predictions and usage for the empty and learned playbooks |
| [Four-tag comparison](evidence/answer_only_100/summary.json) | Paired predictions and usage for the original and four-tag formats |

The three directories also contain `run_manifest.json` with sample IDs and input hashes, `paired_rows.jsonl` with predictions, and `usage.jsonl` with request and token records. The earlier runs and their original commands are described in [Early experiments](EARLY_EXPERIMENTS.md).

## Run the comparisons on the frozen inputs

Install the pinned dependencies in a Python 3.12 virtual environment. The commands below use PowerShell from the repository root.

```powershell
py -3.12 -m venv .venv
$python = ".\.venv\Scripts\python.exe"
& $python -m pip install -r requirements-repro.txt
$env:TIKTOKEN_CACHE_DIR = Join-Path (Get-Location).Path '.venv\tiktoken-cache'
& $python -c "import tiktoken; print(tiktoken.get_encoding('cl100k_base').name)"
& $python experiments/retest_finer.py --confirm-100
& $python experiments/run_answer_only_100.py
```

The tokenizer setup downloads a public vocabulary on first use. The last two commands check the holdouts, prompt and playbook hashes and report the planned 384 completions per experiment.

To make paid API calls, set `DEEPSEEK_API_KEY` in your local environment and explicitly pass `--run`. The runners use `deepseek-flash` at temperature 0 with thinking disabled, JSON mode and a 1,536-token output cap. They apply a per-run spending ceiling based on the remaining headroom in the archived CNY 50 budget. Each invocation uses its own output directory under `runs/`. Hosted model responses and billing can vary.

```powershell
& $python experiments/retest_finer.py --confirm-100 --run
& $python experiments/run_answer_only_100.py --run
```

The archived [training continuation script](experiments/run_training_scale.py) can also plan a run from the saved 24-batch checkpoints using the command below. The report's evaluations start from the already trained 100-batch playbooks.

```powershell
& $python experiments/run_training_scale.py --out runs/training_scale
```

The original ACE code is pinned to [82709de050e1](https://github.com/ace-agent/ace/tree/82709de050e1db6e6ef2f07bcb0393560b94992a). The original documentation and license are in [README_UPSTREAM.md](README_UPSTREAM.md) and [LICENSE.txt](LICENSE.txt).
