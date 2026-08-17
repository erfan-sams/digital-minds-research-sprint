# Tell Me Your Price

**Testing charitable donations as a behavioral common currency for LLM-expressed preferences**

## Overview

This project tests whether heterogeneous outcomes can be assigned donation-equivalent values that predict independent model choices. Using 30 Tier-4 outcomes from [Ajayi et al. (2026)](https://arxiv.org/abs/2606.21102), originally introduced by [Mazeika et al. (2025)](https://arxiv.org/abs/2502.08640), the experiment collected 475,800 forced-choice response records from ten LLMs.

The study includes:

* Direct comparisons of all 435 outcome pairs.
* Transitivity and cycle analysis.
* Donation trade-offs across 12 amounts and four charity frames.
* Estimation of donation-equivalent (d_{50}) values.
* Within-valence cross-instrument validation using permutation tests.

## Main result

Quality-eligible models showed low direct-choice cycle rates, but donation sensitivity varied across models, valence, and charity framing. Under the primary World Food Programme frame, donation-equivalent rankings significantly predicted direct choices for Gemma-4 31B and GPT-5.6 Terra after Holm correction.

## Repository contents

* Experiment notebook for OpenRouter data collection.
* Analysis notebook for statistics, tables, and figures.
* Final research report.

## Reproduction

1. Open the collection notebook in Google Colab.
2. Provide an OpenRouter API key without committing it to GitHub.
3. Inspect the model list, request schedule, and estimated cost.
4. Run the collection notebook, which saves incremental checkpoints.
5. Run the analysis notebook using the completed response CSV.

**Warning:** the full schedule makes 475,800 API requests. The analysis notebook makes no API calls.

## Interpretation

These results describe prompt-elicited model choices. They do not establish persistent internal preferences, consciousness, or welfare.
