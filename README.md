# Membership Inference Attack (RMIA)

## Overview

This repository implements a Membership Inference Attack (MIA) against a pretrained ResNet-18 model.
The goal is to predict whether a given sample was part of the model’s training dataset.

Our approach combines:

* Confidence-based inference
* Reference models (shadow models)
* Relative Membership Inference Attack (RMIA)

---

## Files

* `rmia.py` — main attack pipeline
* `pub.pt` — public dataset (with labels + membership)
* `priv.pt` — private dataset (membership unknown)
* `model.pt` — pretrained target model
* `run.sh` — execution script (cluster)
* `mia.sub` — HTCondor submission file

---

## How to Run

### 🖥️ Local (CPU or GPU)

```bash
python rmia.py
```

This will:

1. Train reference models
2. Compute RMIA scores
3. Save results to:

```
submission.csv
```

---

### ⚡ GPU Cluster (HTCondor)

Submit job:

```bash
condor_submit mia.sub
```

Logs:

```
runlogs/
```

---

## Output Format

The script produces:

```csv
id,score
123,0.82
124,0.11
```

Requirements:

* Scores ∈ [0,1]
* No duplicate IDs

---

## Reproducing Best Result

Best configuration:

* Reference models: **4**
* Epochs: **60**
* Augmentation: **disabled**
* Gamma: **2.0**
* a: **0.3**

Run:

```bash
python rmia.py
```

---

## Notes

* Using 16 reference models degraded performance due to increased variance
* Data augmentation reduced attack effectiveness by smoothing confidence signals

---

## References

* Shokri et al. (2017)
* Carlini et al. (2022)
* Zarifzadeh et al. (2023)
