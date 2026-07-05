# neuroprobe-exp

Experiments for the Neuroprobe benchmark, focused on cross-subject decoding with clean, reproducible training pipelines.

This repository is the working area for trying models, running baselines, and testing ideas such as learnable electrode embeddings against the official Neuroprobe benchmark and splits. Neuroprobe is built on BrainTreebank, a large intracranial recording dataset with about 40 hours of recordings from 10 subjects watching naturalistic stimuli.

## Goal

The current target is strong performance on the **cross-subject** split, which Neuroprobe describes as the hardest setting. The benchmark requires using the package-provided train/validation/test splits for leaderboard submissions.

## Repository layout

```text
neuroprobe-dev/
├── neuroprobe/          # Upstream Neuroprobe repo clone
├── braintreebank/       # Downloaded BrainTreebank dataset
└── neuroprobe-exp/      # This repo
    ├── README.md
    ├── .gitignore
    ├── configs/
    ├── notebooks/
    ├── scripts/
    ├── src/
    └── results/
```

Recommended conventions:
- Keep the upstream `neuroprobe/` clone mostly untouched.
- Put custom training code, notebooks, configs, and evaluation logic here.
- Do not commit dataset files, checkpoints, or bulky outputs.

## Environment

This project is intended to be used with **PyCharm + Conda**.

Example setup:

```bash
conda create -n neuro python=3.11 -y
conda activate neuro
pip install --upgrade pip
pip install neuroprobe requests beautifulsoup4 pandas matplotlib seaborn scikit-learn jupyter
```

Neuroprobe is installed from PyPI with `pip install neuroprobe` 

## Data setup

The BrainTreebank dataset should live outside this repo, typically as a sibling folder:

```text
neuroprobe-dev/
├── neuroprobe/
├── braintreebank/
└── neuroprobe-exp/
```

For benchmark-focused work, the maintainers recommend the lighter download path because it removes unnecessary files for benchmark-only use. The full BrainTreebank release is much larger and is mainly useful for custom raw-data pipelines or extra neuroscience analyses beyond the standard Neuroprobe workflow.

## First run

Set the dataset root before creating datasets:

```python
import os
os.environ['ROOT_DIR_BRAINTREEBANK'] = r'../braintreebank'
```

Minimal smoke test:

```python
import os
import torch
from neuroprobe import BrainTreebankSubject, BrainTreebankSubjectTrialBenchmarkDataset

os.environ['ROOT_DIR_BRAINTREEBANK'] = r'../braintreebank'

subject = BrainTreebankSubject(
    subject_id=1,
    cache=True,
    dtype=torch.float32,
    coordinates_type='mni'
)

dataset = BrainTreebankSubjectTrialBenchmarkDataset(
    subject,
    trial_id=2,
    dtype=torch.float32,
    eval_name='gpt2_surprisal'
)

sample = dataset[0]
print(type(sample), sample['data'].shape if isinstance(sample, dict) else 'ok')
```

The package exposes dataset objects with 1-second neural windows aligned to task labels, and example outputs include electrode labels, electrode coordinates, and metadata.

## Official splits

Use the official Neuroprobe split generators for any leaderboard-relevant experiment:

```python
from neuroprobe import generate_splits_cross_subject

splits = generate_splits_cross_subject(
    test_subject=1,
    test_trial_id=2,
    eval_name='gpt2_surprisal',
    output_indices=False,
)
```

Leaderboard submissions must use the exact provided train/validation/test splits.

## Experiment priorities

Current priorities:
- Reproduce a clean baseline.
- Focus on the cross-subject setting.
- Test ideas that improve robustness across brains and electrode placements.
- Try learnable electrode embeddings, as cross-subject variability is not explained by coordinates alone.

## Git hygiene

This repo should stay lightweight:
- Commit code, configs, notes, and small plots.
- Ignore dataset files, checkpoints, and generated heavy artifacts.
- Keep experiment results organized enough to trace which config produced which run.

## Notes

Neuroprobe covers multiple tasks across audio, language, and vision domains, and evaluates models with AUROC averaged across tasks. That makes reproducibility and clean experiment tracking especially important, because small gains on the leaderboard can come from real modeling improvements rather than noise.