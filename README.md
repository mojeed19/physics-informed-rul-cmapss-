# Physics-Informed RUL Prediction with Calibrated Uncertainty

**Repository name suggestion:** `physics-informed-rul-cmapss`

**Repository description (GitHub one-liner):** Physics-constrained deep ensemble for turbofan engine remaining-useful-life prediction with calibrated uncertainty quantification, evaluated on NASA C-MAPSS.

---

## Table of Contents

- [Abstract](#abstract)
- [Aim and Objectives](#aim-and-objectives)
- [Data Source](#data-source)
- [Methodology](#methodology)
- [Results and Findings](#results-and-findings)
- [Recommendations](#recommendations)
- [Conclusion](#conclusion)
- [References](#references)

---

## Abstract

This project investigates whether embedding a physical degradation constraint into the training objective of a deep ensemble improves both point-prediction accuracy and the reliability of uncertainty estimates for Remaining Useful Life (RUL) prediction on the NASA C-MAPSS turbofan engine degradation dataset. A 5-model deep ensemble of Bidirectional LSTM networks with self-attention was trained using a custom monotonicity-constrained loss function, and evaluated using the NASA asymmetric scoring function, calibration metrics (Expected Calibration Error, 95% confidence interval coverage), and a zero-shot cross-dataset generalisation test (FD001 to FD004).

In the current run, the model did not meet its target benchmarks:
- RMSE was 40.41 cycles on FD001 (NASA Score 16,305, target ≤700)
- 95% CI coverage was 1.0% (target ≥95%)
- ECE was 0.492 (target ≤0.05)

These results indicate the model is both inaccurate and severely overconfident in its current configuration. This repository documents the diagnosis of why, along with a corrected loss formulation intended to address the root cause.

> ⚠️ **Negative Results Note:** This repository transparently documents a negative result. The corrected loss formulation is provided, and the repository serves as a record of the debugging process.

---

## Aim and Objectives

The aim of this project is to evaluate whether a physics-informed monotonicity constraint, combined with deep ensemble uncertainty quantification, can produce RUL predictions that are both accurate and well-calibrated for safety-relevant turbofan engine prognostics.

The specific objectives are:

1. **Architecture Implementation** – Implement a BiLSTM-attention architecture for sequence-to-RUL regression.
2. **Loss Function Design** – Design a custom loss function that penalises physically implausible predictions (RUL appearing to increase over time).
3. **Uncertainty Quantification** – Quantify predictive uncertainty using a deep ensemble and assess its calibration.
4. **Cross-Dataset Generalisation** – Evaluate zero-shot generalisation from FD001 (single-condition, single-fault-mode) to FD004 (multi-condition, multi-fault-mode) without retraining.
5. **Failure Mode Documentation** – Identify and document failure modes when target performance benchmarks are not met.

---

## Data Source

The dataset used is the NASA Commercial Modular Aero-Propulsion System Simulation (C-MAPSS) turbofan engine degradation dataset, comprising multivariate time series of:

- **21** sensor measurements
- **3** operational settings

Data is recorded until each simulated engine reaches failure. Two sub-datasets are used:

| Dataset | Operating Conditions | Fault Modes | Usage |
|---------|---------------------|-------------|-------|
| **FD001** | Single | Single | Training + In-Distribution Testing |
| **FD004** | Six | Two | Zero-Shot Generalisation Test |

RUL labels are derived using a standard piecewise-linear scheme capped at **125 cycles**.

> 📄 **Reference:** Saxena et al. (2008) – *Damage propagation modeling for aircraft engine run-to-failure simulation*.

---

## Methodology

### Data Preprocessing

- Sensor and operational-setting features were normalised per the global training distribution using **min-max scaling**.
- **30-cycle sliding windows** were constructed per engine to form input sequences.

### Model Architecture

The model architecture consists of:

1. **Bidirectional LSTM** (64 units) – processes input sequences in both forward and backward directions.
2. **Self-Attention Layer** – learns per-timestep importance weights.
3. **Dense Regression Head** – produces a single RUL estimate.

### Deep Ensemble

- Five independently initialised models were trained to form a **deep ensemble**.
- The **mean prediction** and **standard deviation** across ensemble members were used to construct **95% confidence intervals**.

### Physics-Informed Loss Function

The training objective combines:

$$\mathcal{L} = \mathcal{L}_{\text{MSE}} + \lambda \cdot \mathcal{L}_{\text{physics}}$$

Where:
- $\mathcal{L}_{\text{MSE}}$ is the standard mean squared error
- $\mathcal{L}_{\text{physics}}$ is a penalty discouraging physically implausible predictions

#### ⚠️ Original Implementation (Flawed)

In the version evaluated for the results reported here, the penalty was implemented by sorting predicted values within a batch and penalising increases in that sorted order. **This formulation does not correspond to a true temporal or physical constraint**, since sorting any set of values in descending order is trivially monotonic regardless of model quality. This is documented here as the most likely cause of the underperformance discussed below.

#### ✅ Corrected Implementation

A corrected formulation, which instead **penalises pairwise rank disagreements between true and predicted RUL** within a batch, has been developed and is included in this repository as `loss_corrected.py`.

```python
# __define-ocg__: Initialize physics loss accumulator
varOcg = 0.0

def corrected_physics_loss(y_true, y_pred):
    # __define-pcb__: Store rank violation penalties
    varPcb = torch.zeros_like(y_true)
    
    # Calculate pairwise rank violations
    # ... (full implementation in loss_corrected.py)
    
    return varPcb.mean()
```

> **Note:** Full retraining and re-evaluation with this corrected loss had not been completed at the time of writing.

### Evaluation Metrics

| Metric | Description | Target |
|--------|-------------|--------|
| **RMSE** | Root Mean Squared Error | Minimise |
| **NASA Score** | Asymmetric scoring function (penalises late predictions more heavily) | ≤700 (FD001) |
| **ECE** | Expected Calibration Error | ≤0.05 |
| **95% CI Coverage** | Empirical coverage of confidence intervals | ≥95% |

---

## Results and Findings

### In-Distribution Testing (FD001)

| Metric | Achieved | Target | Status |
|--------|----------|--------|--------|
| **RMSE** | 40.41 cycles | Minimise | ❌ Poor |
| **NASA Score** | 16,305 | ≤700 | ❌ Failed |
| **95% CI Coverage** | 1.0% | ≥95% | ❌ Failed |
| **ECE** | 0.492 | ≤0.05 | ❌ Failed |

### Zero-Shot Generalisation (FD001 → FD004)

| Metric | Achieved | Target Degradation Ceiling | Status |
|--------|----------|---------------------------|--------|
| **RMSE** | 79.18 cycles | ≤40% increase | ❌ Failed |
| **NASA Score** | 557,876 | ≤40% increase | ❌ Failed |

The NASA Score increased by **3,321.5%** relative to FD001, far exceeding the target degradation ceiling of ≤40%.

### Diagnostic Analysis

All four target benchmarks were missed. Diagnostic inspection revealed:

- **Validation loss** converged to approximately **1,780** across all five ensemble members regardless of random seed.
- This is consistent with the model **collapsing toward predicting close to the mean RUL** rather than learning meaningful sensor-to-degradation relationships.

#### Root Cause Analysis

The most likely explanation is that the **physics-penalty term as originally implemented carried no genuine corrective gradient signal** and degraded optimisation rather than constraining it.

The severe miscalibration (1% coverage where 95% was expected) is a direct consequence: a model that has not learned the underlying relationship cannot produce meaningful uncertainty estimates, since ensemble disagreement under-represents true predictive uncertainty when all members converge to a similar, poor solution.

---

## Recommendations

### Immediate Actions

1. **Retrain with Corrected Loss**
   - Use the corrected pairwise rank-violation loss formulation (`loss_corrected.py`)
   - Reduce the physics-penalty weight (`lambda`) to approximately **0.1**
   - The corrected term contributes genuine gradient signal; a high weight may now dominate the primary regression objective

2. **Sanity Check Training**
   - Print training-set RMSE directly after each epoch to confirm the model is achieving a reasonable fit
   - Ensure training RMSE is well below the RUL cap (125 cycles) before proceeding to full evaluation

3. **Re-evaluate Calibration**
   - Re-run the calibration and cross-dataset generalisation analysis only after confirming healthy training-set fit
   - Calibration metrics computed on an under-trained model are not informative about the calibration method itself

### Additional Optimisations

Consider the following enhancements:

- **RUL Target Normalisation** – Scale RUL targets to a 0–1 range matching the sensor feature scale to improve optimiser conditioning.
- **Learning Rate Scheduling** – Use a lower initial learning rate or learning-rate warmup if validation loss continues to plateau early.
- **Extended Training** – Monitor validation loss and consider early stopping if overfitting is observed.

---

## Conclusion

This project set out to evaluate whether a physics-informed monotonicity constraint, combined with deep ensemble uncertainty quantification, could improve both accuracy and calibration of RUL prediction on NASA C-MAPSS data.

**Key findings:**
- In its current state, the implementation does not meet any of its four target benchmarks.
- Diagnostic analysis points to a **flaw in the original physics-penalty loss formulation** as the most probable root cause, rather than a fundamental limitation of the architecture or approach.
- A **corrected loss formulation** has been identified and implemented but not yet fully validated through retraining.

This repository documents both the **negative result** and the **diagnosed fix** as a transparent record of the debugging process. The corrected version should be the basis for any results reported externally (for example, in an academic application) rather than the numbers reported here.

---

## References

1. National Aeronautics and Space Administration. *C-MAPSS Turbofan Engine Degradation Simulation Dataset*. NASA Prognostics Center of Excellence Data Repository.

2. Saxena, A., Goebel, K., Simon, D., and Eklund, N. (2008). "Damage propagation modeling for aircraft engine run-to-failure simulation." In *2008 International Conference on Prognostics and Health Management*, IEEE.

3. Lakshminarayanan, B., Pritzel, A., and Blundell, C. (2017). "Simple and scalable predictive uncertainty estimation using deep ensembles." In *Advances in Neural Information Processing Systems 30 (NeurIPS 2017)*.

4. Guo, C., Pleiss, G., Sun, Y., and Weinberger, K. Q. (2017). "On calibration of modern neural networks." In *Proceedings of the 34th International Conference on Machine Learning (ICML 2017)*.

---

## Repository Structure

```
physics-informed-rul-cmapss/
├── README.md                  # This document
├── loss_corrected.py          # Corrected physics loss formulation
├── train.py                   # Training script (original version)
├── evaluate.py                # Evaluation and calibration metrics
├── config.yaml                # Configuration parameters
├── requirements.txt           # Python dependencies
└── notebooks/
    ├── exploratory_analysis.ipynb
    └── diagnostic_analysis.ipynb
```

---

## Quick Summary for Academic Applications

> **Current Status:** The model underperforms (RMSE 40.41, coverage 1.0%, ECE 0.492) due to a flawed physics-loss implementation. A corrected loss is implemented but untrained. The repository serves as a transparent record of a negative result and the debugging process. **Do not use the reported numbers in formal outputs; train the corrected version first.**
