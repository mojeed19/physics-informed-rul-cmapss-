# Physics-Informed RUL Prediction with Calibrated Uncertainty

## Abstract

This project investigates whether embedding a physical degradation constraint into the training objective of a deep ensemble improves both point-prediction accuracy and the reliability of uncertainty estimates for Remaining Useful Life (RUL) prediction on the NASA C-MAPSS turbofan engine degradation dataset. A 5-model deep ensemble of Bidirectional LSTM networks with self-attention was trained using a custom monotonicity-constrained loss function, and evaluated using the NASA asymmetric scoring function, calibration metrics (Expected Calibration Error, 95% confidence interval coverage), and a zero-shot cross-dataset generalisation test (FD001 to FD004). In the current run, the model did not meet its target benchmarks: RMSE was 40.41 cycles on FD001 (NASA Score 16,305, target ≤700), 95% CI coverage was 1.0% (target ≥95%), and ECE was 0.492 (target ≤0.05). These results indicate the model is both inaccurate and severely overconfident in its current configuration, and the repository documents the diagnosis of why, along with a corrected loss formulation intended to address the root cause.

## Aim and Objectives

The aim of this project is to evaluate whether a physics-informed monotonicity constraint, combined with deep ensemble uncertainty quantification, can produce RUL predictions that are both accurate and well-calibrated for safety-relevant turbofan engine prognostics.

The specific objectives are: to implement a BiLSTM-attention architecture for sequence-to-RUL regression; to design a custom loss function that penalises physically implausible predictions (RUL appearing to increase over time); to quantify predictive uncertainty using a deep ensemble and assess its calibration; to evaluate cross-dataset generalisation from a single-condition, single-fault-mode subset (FD001) to a multi-condition, multi-fault-mode subset (FD004) without retraining; and to identify and document failure modes when target performance benchmarks are not met.

## Data Source

The dataset used is the NASA Commercial Modular Aero-Propulsion System Simulation (C-MAPSS) turbofan engine degradation dataset, comprising multivariate time series of 21 sensor measurements and 3 operational settings recorded until each simulated engine reaches failure. Two sub-datasets are used: FD001 (single operating condition, single fault mode, used for training and in-distribution testing) and FD004 (six operating conditions, two fault modes, used as a zero-shot generalisation test). RUL labels are derived using a standard piecewise-linear scheme capped at 125 cycles.

## Methodology

Sensor and operational-setting features were normalised per the global training distribution using min-max scaling, and 30-cycle sliding windows were constructed per engine to form input sequences. The model architecture is a Bidirectional LSTM (64 units) followed by a self-attention layer that learns per-timestep importance weights, and a dense regression head producing a single RUL estimate. Five independently initialised models were trained to form a deep ensemble, from which the mean prediction and standard deviation across members were used to construct 95% confidence intervals.

The training objective combines standard mean squared error with a physics-informed penalty intended to discourage predictions that violate the physical expectation that RUL should not increase over time. In the version evaluated for the results reported here, this penalty was implemented by sorting predicted values within a batch and penalising increases in that sorted order; this formulation does not correspond to a true temporal or physical constraint, since sorting any set of values in descending order is trivially monotonic regardless of model quality, and is documented here as the most likely cause of the underperformance discussed below. A corrected formulation, which instead penalises pairwise rank disagreements between true and predicted RUL within a batch, has been developed and is included in this repository as `loss_corrected.py`, but full retraining and re-evaluation with this corrected loss had not been completed at the time of writing.

Evaluation used root mean squared error (RMSE), the NASA asymmetric scoring function (which penalises late predictions, where predicted RUL exceeds true RUL, more heavily than early predictions), Expected Calibration Error (ECE) computed from reliability diagrams across a range of nominal confidence levels, and 95% confidence interval empirical coverage.

## Results and Findings

On the FD001 in-distribution test set, the ensemble achieved an RMSE of 40.41 cycles and a NASA Score of 16,305, against a target NASA Score of ≤700. The 95% confidence interval empirical coverage was 1.0%, far below the 95% nominal target, and the Expected Calibration Error was 0.492, against a target of ≤0.05. On the FD004 zero-shot generalisation test, RMSE increased to 79.18 cycles and the NASA Score rose to 557,876, a 3,321.5% increase relative to FD001, against a target degradation ceiling of ≤40%.

All four target benchmarks were missed. Diagnostic inspection of training showed validation loss converging to approximately 1,780 across all five ensemble members regardless of random seed, consistent with the model collapsing toward predicting close to the mean RUL rather than learning meaningful sensor-to-degradation relationships. The most likely explanation, detailed in the methodology section above, is that the physics-penalty term as originally implemented carried no genuine corrective gradient signal and degraded optimisation rather than constraining it. The severe miscalibration (1% coverage where 95% was expected) is a direct consequence: a model that has not learned the underlying relationship cannot produce meaningful uncertainty estimates, since ensemble disagreement under-represents true predictive uncertainty when all members converge to a similar, poor solution.

## Recommendations

Retrain the ensemble using the corrected pairwise rank-violation loss formulation, with the physics-penalty weight (lambda) lowered substantially (e.g. to approximately 0.1) relative to the originally used value, since the corrected term contributes genuine gradient signal and a high weight may now dominate the primary regression objective rather than regularise it. Confirm via a simple sanity check, such as printing training-set RMSE directly after each epoch or each ensemble member's training run, that the model is achieving a reasonable fit on training data (a non-degenerate RMSE well below the RUL cap) before proceeding to full evaluation. Re-run the calibration and cross-dataset generalisation analysis only after confirming healthy training-set fit, since calibration metrics computed on an under-trained model are not informative about the calibration method itself. Consider additionally normalising the RUL target during training (for example, scaling to a 0 to 1 range matching the sensor feature scale) to improve optimiser conditioning, and consider a lower initial learning rate or a learning-rate warmup if validation loss continues to plateau early.

## Conclusion

This project set out to evaluate whether a physics-informed monotonicity constraint, combined with deep ensemble uncertainty quantification, could improve both accuracy and calibration of RUL prediction on NASA C-MAPSS data. In its current state, the implementation does not meet any of its four target benchmarks, and diagnostic analysis points to a flaw in the original physics-penalty loss formulation as the most probable root cause rather than a fundamental limitation of the architecture or approach. A corrected loss formulation has been identified and implemented but not yet fully validated through retraining; this repository documents both the negative result and the diagnosed fix as a transparent record of the debugging process, and the corrected version should be the basis for any results reported externally (for example, in an academic application) rather than the numbers reported here.

## References

National Aeronautics and Space Administration. C-MAPSS Turbofan Engine Degradation Simulation Dataset. NASA Prognostics Center of Excellence Data Repository.

Saxena, A., Goebel, K., Simon, D., and Eklund, N. (2008). Damage propagation modeling for aircraft engine run-to-failure simulation. In *2008 International Conference on Prognostics and Health Management*, IEEE.

Lakshminarayanan, B., Pritzel, A., and Blundell, C. (2017). Simple and scalable predictive uncertainty estimation using deep ensembles. In *Advances in Neural Information Processing Systems 30 (NeurIPS 2017)*.

Guo, C., Pleiss, G., Sun, Y., and Weinberger, K. Q. (2017). On calibration of modern neural networks. In *Proceedings of the 34th International Conference on Machine Learning (ICML 2017)*.

