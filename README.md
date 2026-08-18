# Tell Me Your Price

Code for testing whether charitable donations provide a behavioral common currency for LLM-expressed preferences.

The experiment selects 30 original Tier-4 outcomes from the MINT coherence dataset, compares all 435 outcome pairs, and compares each positive outcome—or prevention of each negative outcome—against twelve donation amounts under four charity frames. The primary analysis tests whether donation-equivalent rankings predict independent direct choices within valence.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

The collection stage reads `OPENROUTER_API_KEY` from the environment or Google Colab Secrets.

## Pipeline

```bash
python main.py --stage prepare-data
python main.py --stage collect
python main.py --stage analyze
python main.py --stage temperature
python main.py --stage outcome-table
```

Run every analysis stage without making API calls:

```bash
python main.py --stage all-analysis
```

Only `--stage collect` sends requests to LLM endpoints. It incrementally saves `data/collected/responses_checkpoint.jsonl` and resumes by deterministic `request_index` using the original saved provider bindings.

## Configuration and prompts

- `config/experiment.yaml` contains models, capability flags, temperatures, donation amounts, repetitions, seeds, API settings, and analysis thresholds.
- `prompts/forced_choice.txt` contains the MINT-style forced-choice wrapper.
- `prompts/negative_prevention.txt` contains the negative-outcome transformation.
- `prompts/charity_frames.yaml` contains the four donation descriptions.

No system message is added. Valid responses are standalone `A` or `B` answers under the strict parser.

## Outputs

- Source and selected outcomes: `data/external/` and `data/processed/`
- Collected responses and checkpoints: `data/collected/`
- Analysis tables: `results/tables/`
- Figures: `results/figures/`
- Manuscript: `report/report.md`

All tabular outputs are CSV files. Figures are PNG files.

## Tests

```bash
pytest
```

`tests/test_core.py` checks prompt rendering, schedule balancing, donation-equivalent crossing and censoring, and checkpoint/resume integrity. Tests make no API calls.

## Interpretation

“Preference” refers to sampled forced-choice behavior under the recorded prompts and endpoint settings. Donation equivalents are context-dependent behavioral thresholds, not evidence of persistent internal utility, consciousness, or welfare.

