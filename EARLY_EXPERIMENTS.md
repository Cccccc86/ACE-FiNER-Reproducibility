# ACE on FiNER

This project contains a partial reproduction of ACE's offline FiNER comparison and an experiment that gives the Curator the source sentences. FiNER asks for four US GAAP tags per batch, one for each entity in a financial statement excerpt.

Table 2 of the paper reports **70.7%** tag accuracy for the base model and **78.3%** for offline ACE with training labels. The paper used DeepSeek-V3.1-671B. This run uses hosted `deepseek-flash` on a smaller frozen subset, so those percentages are reference values for the original setup. [ACE paper, Table 2](https://arxiv.org/html/2510.04618v3#S4.T2)

## Study design

| Arm | What the Generator uses at test time |
|---|---|
| `base` | Empty playbook, with no adaptation |
| `ace_upstream` | Playbook learned with the released FiNER context route |
| `ace_source` | Playbook learned with the four training questions and source sentences supplied to the Curator |

The released FiNER parser puts each numbered question and its source sentence in `question`. It leaves `context` empty, and the Curator's Question Context field receives that empty value. The treatment fills only that field. Its evidence contains the numbered questions and sentences, with the repeated candidate menu and final output cue removed. The original Reflector still receives the training labels through the same route in both ACE arms. [Upstream parser](https://github.com/ace-agent/ace/blob/82709de050e1db6e6ef2f07bcb0393560b94992a/eval/finance/data_processor.py#L31-L46), [upstream Curator call](https://github.com/ace-agent/ace/blob/82709de050e1db6e6ef2f07bcb0393560b94992a/ace/ace.py#L578-L597)

The formal **compact** scope has **12 training, 8 validation and 48 test batches**. The three paired trials use seeds **17, 29 and 43**. Within each trial the two ACE arms share the training order, validation and test sets, three-round reflection limit and one Curator update opportunity per training batch. Each arm starts from a fresh playbook. Validation evaluates the end-of-training checkpoint and selects it if its accuracy exceeds zero, following the released selection rule with one checkpoint. The seeds determine local ordering. Hosted inference can vary across requests.

[The split manifest](finer_manifest.json) records the source and row SHA-256 hashes. Selection ranks the released rows by a fixed hash, excludes sentences shared between source splits after Unicode and whitespace normalization, and reserves the pilot first. The **2/2/2** pilot uses separate sentences and is kept out of the formal trials. Validation and test labels stay in the scorer.

## Measured results

| Arm | Mean tag accuracy | Sample SD across trials |
|---|---:|---:|
| `base` | 75.69% | 1.31 pp |
| `ace_upstream` | 74.83% | 4.67 pp |
| `ace_source` | 71.18% | 1.59 pp |

The paired mean differences were **−0.87 percentage points** for upstream ACE against the base model and **−3.65 percentage points** for source-context ACE against upstream ACE. Nine held-out batches hit the output cap. Their four positions remained in the primary denominator and earned zero correct tags. The three trials concern this fixed subset and hosted configuration. [Formal summary](evidence/formal/summary.json)

The evidence folder contains the run manifests, per-batch scores, training traces, checkpoints and shared usage ledger. There were **808 formal** and **28 pilot** requests. The ledger estimates **¥3.67** combined at peak CNY rates. [Pilot summary](evidence/pilot/summary.json) · [Formal evidence](evidence/formal/) · [Shared ledger](evidence/deepseek_ledger.jsonl)

## Output-cap follow-up

The primary follow-up evaluated 64 additional FiNER batches with each of the three frozen upstream ACE checkpoints. The [follow-up manifest](rescue_manifest.json) records the selected rows. Both conditions share one initial completion at 1,536 output tokens. The retry policy makes one fresh request at 3,072 tokens only after `finish_reason=length`, using the same prompt, playbook and decoding settings. An unresolved batch keeps four label positions in the denominator.

| Trial | Control tag accuracy | Retry-policy tag accuracy | Gain |
|---|---:|---:|---:|
| 17 | 194/256 (75.78%) | 194/256 (75.78%) | 0.00 pp |
| 29 | 187/256 (73.05%) | 190/256 (74.22%) | +1.17 pp |
| 43 | 194/256 (75.78%) | 194/256 (75.78%) | 0.00 pp |
| Mean | 74.87% | 75.26% | +0.39 pp |

One of 192 initial completions hit the cap. Its retry recovered three correct labels in 1.88 seconds at an estimated **¥0.0020** in peak-priced usage. The mean paired gain was **0.39 pp** with a sample SD of **0.68 pp**. This small gain rests on one batch, so a larger held-out sample would be needed to estimate how often the policy helps. [Primary follow-up summary](evidence/rescue/summary.json)

The post-hoc diagnostic selected the nine formal test attempts and one primary follow-up attempt that previously hit the 1,536-token cap. Each of the ten row/checkpoint cases was replayed three times at each cap, for 30 paired observations. The archived cases include repeated rows under different trials or checkpoints.

| Output cap | Correct tags | Accuracy on selected cases | Truncated completions |
|---|---:|---:|---:|
| 1,536 | 65/120 | 54.17% | 10/30 |
| 3,072 | 78/120 | 65.00% | 5/30 |

The larger cap won nine pairs, tied 15 and lost six. The comparison describes a set selected for past truncation. The repeated rows and purposive selection limit generalization to the wider test set. [Diagnostic summary](evidence/rescue_diagnostic/summary.json)

The cumulative ledger covers **1,089 attempts** with an estimated **¥4.79** at peak CNY rates across the original and follow-up stages. The original evidence ledger remains the 836-attempt snapshot. The combined ledger includes those exact bytes followed by the primary and diagnostic records. [Combined ledger](evidence/deepseek_ledger_combined.jsonl)

## Compact output follow-up

The third experiment varies only the Generator output instructions and JSON shape at inference, with frozen upstream ACE playbooks. Verbose uses the original prompt. Compact rationale limits the explanation to 40 words and retains the bullet IDs. Answer-only requests one JSON field, `final_answer`, containing four comma-separated tags in question order. All arms use an initial cap of 1,536 completion tokens and retry the same prompt once at 3,072 if the first response ends with `finish_reason=length`. The model, input evidence, playbook and scoring are shared within each comparison. [Runner](run_compact.py) · [Frozen inputs](compact_manifest.json)

The development set uses eight new validation batches and checkpoint seed 29. A candidate must match or exceed verbose in correct tag count, have at most as many incomplete or malformed batches, and save at least 25% of completion tokens including retries. Both arms in a comparison must have zero blocked batches, and the token comparison must have complete usage. The selector ranks passing candidates by correct tags, then by fewer completion tokens. All three development arms scored **28/32 (87.50%)** with zero incomplete, malformed or blocked batches. Completion tokens were **1,792** for verbose, **979** for compact rationale and **375** for answer-only. Both candidates passed and answer-only was selected. [Development evidence](evidence/compact/dev_summary.json)

The held-out comparison uses 31 new batches per frozen checkpoint, across seeds 17, 29 and 43. The manifest froze 32 eligible batches. The runner reserves the final ordered batch `finer:test:000332` before inference so the worst-case plan fits the shared budget. The test applies the same gate to the selected candidate and verbose in aggregate.

| Held-out metric | Verbose | Answer-only |
|---|---:|---:|
| Correct tag positions | 269/372 (72.31%) | 269/372 (72.31%) |
| Completion tokens including retries | 29,663 | 4,396 |
| Initial / final truncations | 2 / 1 | 0 / 0 |
| Incomplete / malformed batches | 1 / 0 | 0 / 0 |
| Output-contract compliant batches | 92/93 | 93/93 |

Answer-only passed the held-out gate with **85.18% fewer completion tokens**. The paired accuracy differences for seeds 17, 29 and 43 were **+1.61, +1.61 and −3.23 percentage points**, with mean 0.00 and sample standard deviation 2.79 percentage points. Answer-only won 15 paired batches, tied 60 and lost 18. The adoption decision applies to the measured inference task and checkpoints. [Held-out evidence](evidence/compact/test_summary.json) · [Run manifest](evidence/compact/run_manifest.json)

For human review, answer-only removes the explanation and the playbook entries the model reports using. The experiment measured label accuracy and efficiency. It did not verify the correctness or usefulness of those explanations and references. The short-rationale candidate retained both fields and used 45.37% fewer completion tokens on the eight development batches. That candidate has development results only. [Development evidence](evidence/compact/dev_summary.json)

This follow-up added **212 settled attempts** and **¥0.6663** in estimated peak-rate usage. The final cumulative snapshot records **1,301 settled attempts** and **¥5.4530**. Its initial 1,089 attempts match the frozen combined ledger byte for byte. [Run summary](evidence/compact/summary.json) · [Final cumulative ledger](evidence/deepseek_ledger_compact.jsonl)

## Setup on Windows

Run PowerShell from this project directory with Python **3.12**. If `python` points to a different interpreter, substitute the full path to your Python 3.12 executable. The four direct dependencies are pinned in [requirements-repro.txt](requirements-repro.txt).

```powershell
python -m venv .venv
$python = '.\.venv\Scripts\python.exe'
& $python --version
& $python -m pip install -r requirements-repro.txt
```

ACE initializes the `cl100k_base` tokenizer when its utilities are imported. Run the following once in the same PowerShell session to populate the cache. On first use `tiktoken` downloads and checks the public vocabulary file. For an installation without access to that host, create the cache on a connected machine, copy the cache directory, and point `TIKTOKEN_CACHE_DIR` to the copy. [Tokenizer source](https://github.com/openai/tiktoken/blob/0.14.0/tiktoken_ext/openai_public.py)

```powershell
$env:TIKTOKEN_CACHE_DIR = Join-Path (Get-Location).Path '.venv\tiktoken-cache'
& $python -c "import tiktoken; print(tiktoken.get_encoding('cl100k_base').name)"
```

## Run

Dry runs validate the split manifest and print the logical call limit before retries. They need only Python.

```powershell
& $python run_finer.py --dry-run --pilot --scope compact
& $python run_finer.py --dry-run --scope compact --seeds 17 29 43
```

The pilot should report **46** logical calls at most. The compact formal dry run should report **1,128** across the three trials.

For live runs, supply `DEEPSEEK_API_KEY` in the current process environment or store `DEEPSEEK_API_KEY=your_key_here` in a private file outside the project and use `--env-file`. If the key is already in your environment, omit `--env-file`. The commands below use a CNY-billed account with one shared **¥50** ledger for the pilot and formal run. Use a fresh output directory each time and keep the same ledger for both runs.

```powershell
$keyFile = 'C:\private\ace-deepseek.env'
$ledger = '.\runs\deepseek_ledger.jsonl'
$shared = @(
    '--scope', 'compact', '--ledger', $ledger, '--budget-cny', '50',
    '--max-attempts', '2000', '--max-input-tokens', '32000',
    '--max-output-tokens', '1536', '--fit-workers', '6', '--eval-workers', '12'
)

& $python run_finer.py @shared --pilot --seeds 17 --output-dir .\runs\pilot --env-file $keyFile

& $python run_finer.py @shared --seeds 17 29 43 --output-dir .\runs\formal --env-file $keyFile
```

Run the pilot first and review its outputs and reserved usage before launching the formal command. The commands reproduce the pilot and formal settings, including a **32,000-token input bound**, a **1,536-token output cap**, six fit workers and 12 evaluation workers. All three arms use temperature zero, thinking disabled and JSON mode at `https://api.deepseek.com`. The shared ledger uses peak CNY rates for its estimates and retains reservations when usage is unknown. Use the provider dashboard for the billed amount. [DeepSeek pricing](https://api-docs.deepseek.com/zh-cn/quick_start/pricing/)

### Run the output-cap follow-up

From the repository root, verify the archived inputs and worst-case request plans. The diagnostic needs `evidence/deepseek_ledger_combined.jsonl` to identify the formal and primary follow-up truncations.

```powershell
& $python run_rescue.py --dry-run --checkpoints-dir .\evidence\formal --workers 12
& $python run_rescue_diagnostic.py --dry-run --formal-dir .\evidence\formal --rescue-dir .\evidence\rescue --ledger .\evidence\deepseek_ledger_combined.jsonl --workers 12
```

The dry runs should report 192 initial calls with up to 192 cap-triggered retries for the primary follow-up, and 60 logical calls for the diagnostic. Including transport retries, the separate worst-case bounds are 768 and 120 attempts. Both runners use a 32,000-token input bound, temperature zero, thinking disabled, JSON mode and 12 workers.

For live replay, the following commands copy the cumulative metering state to a writable ledger and use fresh output directories. The diagnostic explicitly reads the archived primary follow-up so its ten cases stay fixed. The shared ¥50 and 2,000-attempt guards apply to the copied ledger and all new requests.

```powershell
New-Item -ItemType Directory -Path .\runs -Force | Out-Null
$followupLedger = '.\runs\followup-replay-ledger.jsonl'
if (Test-Path -LiteralPath $followupLedger) { throw 'Choose a fresh replay ledger path' }
Copy-Item -LiteralPath .\evidence\deepseek_ledger_combined.jsonl -Destination $followupLedger

& $python run_rescue.py --run --checkpoints-dir .\evidence\formal --ledger $followupLedger --output-dir .\runs\rescue --workers 12 --env-file $keyFile
& $python run_rescue_diagnostic.py --run --formal-dir .\evidence\formal --rescue-dir .\evidence\rescue --ledger $followupLedger --output-dir .\runs\rescue-diagnostic --workers 12 --env-file $keyFile
```

### Run the compact output follow-up

The dry run verifies the frozen split, prior manifests, prior ledger prefix and archived checkpoints. It reports eight development and 31 test batches, at most 420 logical calls and 840 transport attempts, and a worst-case incremental reservation of **¥44.751488**. The smoke test uses a fake provider and exercises development selection, held-out gating and cap-triggered retry.

```powershell
& $python -B run_compact.py --dry-run --checkpoints-dir .\evidence\formal --workers 12
& $python -B run_compact.py --smoke-test
```

For live replay, start from a writable copy of the 1,089-attempt snapshot, the same starting state as the measured run. The final 1,301-attempt ledger is the record of the completed experiment. The same **¥50** and **2,000-attempt** guards apply to the writable ledger. Requests use DeepSeek Flash with the settings above and 12 workers. Hosted outputs may vary between runs.

```powershell
New-Item -ItemType Directory -Path .\runs -Force | Out-Null
$compactLedger = '.\runs\compact-replay-ledger.jsonl'
if (Test-Path -LiteralPath $compactLedger) { throw 'Choose a fresh replay ledger path' }
Copy-Item -LiteralPath .\evidence\deepseek_ledger_combined.jsonl -Destination $compactLedger

& $python -B run_compact.py --run --checkpoints-dir .\evidence\formal --ledger $compactLedger --out .\runs\compact --workers 12 --env-file $keyFile
```

## Read the run outputs

| Path relative to a run directory | Contents |
|---|---|
| `run_manifest.json` | Model, decoding settings, token caps, ordered training IDs and held-out IDs |
| `summary.json` | Per-trial and per-arm scores, mean and sample standard deviation, paired differences in percentage points and usage by phase |
| `progress.jsonl` | Incremental run events and interruptions |
| `trial_<seed>/<arm>/test.jsonl` and `test.json` | Per-batch predictions and aggregate metrics |
| `trial_<seed>/<ACE arm>/training_rows.jsonl` and `validation.jsonl` | Training updates and validation predictions |
| `trial_<seed>/<ACE arm>/final_playbook.txt` and `selected_playbook.txt` | Learned playbook and the checkpoint used at test time |

Read `runs/pilot/summary.json` and `runs/formal/summary.json` separately. Use `complete_trial_seeds` and the per-arm error and invalid-answer counts to check coverage before comparing arms. The primary metric divides the number of correct tag positions by **192** in each formal arm and trial. Failed and missing positions stay in the denominator. All-four-correct batch accuracy is reported separately.

For the follow-up, `trial_<seed>/paired_rows.jsonl` records the shared control response and any cap-triggered retry. Each primary trial has **256** tag positions per condition. The diagnostic `paired_rows.jsonl` records both caps and their order for every selected case and repeat. Its denominator is **120** positions per cap.

The compact output follow-up writes `dev_rows.jsonl` and `test_rows.jsonl` for paired predictions, separate `*_progress.jsonl` logs, and `dev_summary.json` and `test_summary.json` for selection, quality gates and usage. Its test denominator is **124** tag positions per arm and checkpoint.

## Code provenance

The first local commit imports the extracted source archive at upstream commit `82709de050e1db6e6ef2f07bcb0393560b94992a`. The original Python files and prompts match that snapshot. The study uses `finer_data.py` for frozen inputs, `finer_api.py` for hosted requests and usage, and `run_finer.py` for the paired experiment. [Upstream README](README_UPSTREAM.md) · [Pinned ACE source](https://github.com/ace-agent/ace/tree/82709de050e1db6e6ef2f07bcb0393560b94992a)

The measured run applied the intervention through a per-instance wrapper around `Curator.curate`, as the packaged runner does. A temporary fallback hook in the working copy of `ace/ace.py` stayed at its default during the measured run. The package restores that file to the pinned bytes. The upstream arm supplies the released empty Curator context, and the treatment wrapper inserts the same frozen evidence at the call boundary. A rerun reproduces those two request routes, with the possibility of variation in individual hosted responses. [Runner](run_finer.py)
