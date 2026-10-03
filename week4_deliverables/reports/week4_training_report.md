# Week 4 Training Report: Full Multi-Seed Experiment

Project: Do CNN Attention Modules Survive Common Corruptions? A Severity-Conditioned Comparison of SE, BAM, and CBAM on CIFAR-10-C.

## 1. Scope

Week 4 completed the full three-seed training experiment for four model variants under two training conditions:

- Plain CNN (None)
- SE
- BAM
- CBAM

Each variant was trained from scratch with seeds 0, 1, and 2 under both clean CIFAR-10 training and corrupted-training conditions.

This resulted in 24 completed 100-epoch training runs:

4 variants × 3 seeds × 2 training conditions = 24 runs.

The frozen Week 3 architecture, attention insertion point, SE reduction ratio, BAM patch, CBAM settings, optimizer family, augmentation, full CIFAR-10 training split, and CIFAR-10-C evaluation design were preserved.

## 2. Frozen full-run configuration

- Variants: Plain CNN, SE, BAM, CBAM.
- Seeds: 0, 1, 2.
- Training data: full CIFAR-10 training split, 50,000 images.
- Schedule: 100 epochs, cosine annealing over the full schedule.
- Optimizer: SGD, learning rate 0.1, momentum 0.9, weight decay 5e-4.
- Batch size: 128.
- Augmentation: RandomCrop(32, padding=4), RandomHorizontalFlip, CIFAR-10 normalization.
- Clean evaluation: all 10,000 CIFAR-10 test images.
- Corruption evaluation: brightness, contrast, defocus_blur, and elastic_transform, severities 1-5, all 10,000 images per corruption and severity.
- Evaluation: model.eval() and torch.no_grad(), with model weights and batch-normalization statistics frozen.

The runnable configuration is `week4_config.json`.
The runnable full-run entry point is `scripts/run_week4_full.py`.

## 3. Training completion

All 24 planned runs completed the full 100-epoch schedule.

### Clean-trained models

| Variant | Seed 0 Final Loss | Seed 0 Final Acc. | Seed 1 Final Loss | Seed 1 Final Acc. | Seed 2 Final Loss | Seed 2 Final Acc. | Mean Acc. ± SD |
|---|---:|---:|---:|---:|---:|---:|---:|
| None | 0.015227 | 99.864% | 0.015603 | 99.826% | 0.016145 | 99.824% | 99.838% ± 0.018% |
| SE | 0.013588 | 99.864% | 0.013362 | 99.892% | 0.013815 | 99.880% | 99.879% ± 0.011% |
| BAM | 0.009491 | 99.934% | 0.009345 | 99.914% | 0.009216 | 99.914% | 99.921% ± 0.009% |
| CBAM | 0.013985 | 99.864% | 0.014081 | 99.856% | 0.014046 | 99.842% | 99.854% ± 0.009% |

### Corrupted-trained models

| Variant | Seed 0 Final Loss | Seed 0 Final Acc. | Seed 1 Final Loss | Seed 1 Final Acc. | Seed 2 Final Loss | Seed 2 Final Acc. | Mean Acc. ± SD |
|---|---:|---:|---:|---:|---:|---:|---:|
| None | 0.099444 | 96.898% | 0.100404 | 96.844% | 0.101217 | 96.842% | 96.861% ± 0.026% |
| SE | 0.082656 | 97.542% | 0.087336 | 97.362% | 0.085603 | 97.450% | 97.451% ± 0.073% |
| BAM | 0.070363 | 97.902% | 0.071301 | 97.836% | 0.070105 | 97.922% | 97.887% ± 0.037% |
| CBAM | 0.083465 | 97.522% | 0.083020 | 97.448% | 0.085122 | 97.470% | 97.480% ± 0.031% |

The values above are final training loss and training accuracy recorded at epoch 100.

## 4. Training observations

All 24 runs reached epoch 100, satisfying the planned training schedule.

For the clean-trained runs, the final training accuracies were above 99.8% for all four variants.

For the corrupted-trained runs, final training accuracies were approximately 96.8%-97.9% across the four variants.

These are training-set measurements and should not be interpreted as test-set accuracy or corruption robustness. Test and corruption evaluation results are reported separately.

## 5. Evaluation evidence

Clean and corrupted-trained checkpoints were evaluated separately using the Week 4 evaluation pipeline.

The clean evaluation results are stored in:

`evaluation_results/clean_results.json`

The corrupted evaluation results are stored in:

`evaluation_results/corrupted_results.json`

The clean-trained models were evaluated on the CIFAR-10 test set and on the specified CIFAR-10-C corruptions.

For checkpoint selection, clean-trained checkpoints were selected using maximum recorded validation accuracy. The corrupted-training logs did not contain validation accuracy, so corrupted-trained checkpoints were selected using maximum recorded training accuracy. This distinction is retained as an experimental limitation and should be considered when interpreting cross-condition comparisons.

## 6. Robustness analysis

The completed evaluation results were analyzed across the four attention variants, three random seeds, four corruption types, and five severity levels.

### 6.1 Clean test accuracy

Mean clean test accuracy across the three seeds was:

| Variant | Mean Clean Test Accuracy |
|---|---:|
| None | 92.973% |
| SE | 93.347% |
| BAM | 93.893% |
| CBAM | 92.983% |

Relative to the matched no-attention baseline, the mean clean-accuracy differences were:

| Variant | Gain vs. None |
|---|---:|
| SE | +0.373 percentage points |
| BAM | +0.920 percentage points |
| CBAM | +0.010 percentage points |

### 6.2 Corruption robustness

For each training condition, mean accuracy was calculated across all 20 corruption/severity cells (4 corruption types × 5 severity levels), averaging across the three seeds.

| Variant | Training condition | Mean corruption accuracy | SD | Degradation from clean |
|---|---|---:|---:|---:|
| None | Clean-trained | 78.722% | 17.267% | 14.252 pp |
| SE | Clean-trained | 79.791% | 16.821% | 13.555 pp |
| BAM | Clean-trained | 80.740% | 16.621% | 13.154 pp |
| CBAM | Clean-trained | 79.801% | 16.035% | 13.182 pp |
| None | Corrupted-trained | 91.360% | 2.353% | 1.477 pp |
| SE | Corrupted-trained | 91.844% | 2.266% | 1.363 pp |
| BAM | Corrupted-trained | 92.628% | 2.048% | 1.125 pp |
| CBAM | Corrupted-trained | 91.686% | 2.136% | 1.194 pp |

The degradation values are calculated relative to the clean-test accuracy of the corresponding training condition and variant.

For the corrupted-trained models, the mean accuracy gains over the no-attention baseline across the 20 corruption/severity cells were:

| Variant | Mean gain vs. None |
|---|---:|
| SE | +0.484 percentage points |
| BAM | +1.268 percentage points |
| CBAM | +0.326 percentage points |

Among the four tested corruption types, elastic transform produced the lowest accuracies at high severity. At severity 5, BAM achieved 92.717% on brightness, 92.630% on contrast, 91.033% on defocus blur, and 85.387% on elastic transform.

### 6.3 Interpretation

Under the tested CIFAR-10-C conditions, all three attention variants showed positive mean accuracy differences relative to the matched no-attention baseline in the analyzed corruption evaluations. BAM showed the largest mean gain in this experiment.

The comparison also shows a strong effect of training condition: models trained with the corrupted-training setup had substantially smaller degradation from their corresponding clean-test accuracy than clean-trained models across all four variants.

These results are descriptive of this experimental setup. They do not establish that any attention mechanism is universally more robust, and the effect of attention should be interpreted separately from the effect of corrupted training.

The detailed analysis outputs are stored in:

- `robustness_detailed_summary.csv`
- `attention_gain_vs_none.csv`
- `analyze_robustness_detailed.py`
- `analyze_attention_gain.py`
- `final_attention_comparison.py`
## 7. Training logs

The 12 clean-training CSV logs are stored in:

`week4_deliverables/logs/`

Files:

- `train_none_seed0.csv`
- `train_none_seed1.csv`
- `train_none_seed2.csv`
- `train_se_seed0.csv`
- `train_se_seed1.csv`
- `train_se_seed2.csv`
- `train_bam_seed0.csv`
- `train_bam_seed1.csv`
- `train_bam_seed2.csv`
- `train_cbam_seed0.csv`
- `train_cbam_seed1.csv`
- `train_cbam_seed2.csv`

The 12 corrupted-training CSV logs are stored in:

`week4_corrupted/logs/`

Files:

- `corrupted_none_seed0.csv`
- `corrupted_none_seed1.csv`
- `corrupted_none_seed2.csv`
- `corrupted_se_seed0.csv`
- `corrupted_se_seed1.csv`
- `corrupted_se_seed2.csv`
- `corrupted_bam_seed0.csv`
- `corrupted_bam_seed1.csv`
- `corrupted_bam_seed2.csv`
- `corrupted_cbam_seed0.csv`
- `corrupted_cbam_seed1.csv`
- `corrupted_cbam_seed2.csv`

## 8. Known limitations

- Final training accuracy is not a substitute for clean test accuracy or corruption robustness.
- Clean and corrupted checkpoint selection used different criteria because validation accuracy was not recorded for the corrupted-training runs.
- The primary CIFAR-10-C arrays are not copied into the deliverables archive because of their large storage size.
- Robustness conclusions should be based on the completed evaluation results rather than final training accuracy alone.

## 9. Status

WEEK 4 COMPLETED

24/24 planned training runs completed:
- Clean: 12/12
- Corrupted: 12/12
- Epochs per run: 100
- Seeds: 0, 1, 2
- Variants: None, SE, BAM, CBAM




