# LingoLLM: Learning to Route Molecular Editing Experts for Multi-Objective Optimization

CMU 11-785 (Introduction to Deep Learning), Group 36.
Zihan Li (`lilyl2`), Yefeng Lin (`yefengl`), River Zhu (`riverz`).

The experts that propose molecular edits are held fixed. The question is whether a
learned router picks which expert to call next better than a heuristic one.

## Contents

```
src/        environment, editor pools, SMILES generator, routing policies
scripts/    one script per experiment or diagnostic
data/       scaffold-split seeds, generator corpus, within-scaffold analog pairs
oracle/     GSK3B and JNK3 predictors, SA fingerprint scores
```

`src/` holds the shared edit-and-score environment (`env.py`), the two editor
families (`editors.py` for the RDKit reaction pool, `gen_editors.py` +
`generator.py` for the conditional SMILES transformer), the six routing policies
(`policies.py`), and the paired-evaluation helpers (`analysis.py`).

## Setup

```bash
conda env create -f environment.yml
conda activate lingollm
```

`data/chembl_v29.csv` is not committed (150 MB, over GitHub's file size limit).
Download it again with `python scripts/make_seeds.py`; everything else in `data/`
is here.

## Running

```bash
# RDKit editor pool, 200 train / 100 test seeds, about 53 min on 8 CPU cores
python scripts/run_experiment.py --editor-kind rdkit \
    --n-train 200 --n-test 100 --repeats 3 --train-repeats 3 \
    --max-steps 20 --workers 8 --seed 0 --out results_base0

# generative editor pool (needs a trained checkpoint, see scripts/train_generator.py)
python scripts/run_experiment.py --editor-kind gen --out results_gen_router
```

Results land in the `--out` directory and are not committed. The analysis and
plotting scripts read from there; see the header comment of each script in
`scripts/` for what it answers.

## Scope

A methodology experiment on public benchmark oracles. Nothing here is a claim about
real biological activity. The files in `oracle/` are third-party predictors
redistributed from Therapeutics Data Commons and RDKit.
