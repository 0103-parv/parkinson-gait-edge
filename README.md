# Parkinson's Progression Detection on Edge Devices

ML pipeline that quantifies Parkinson's disease progression from gait data, engineered for on device inference on standard smartphones. Deployed in hospitals in India. Paper accepted for publication.

## Method

- **Input:** SMPL meshes from CARE PD dataset family
- **Features:** axis angle pose converted to 6D rotation representations with angular and root velocity
- **Model:** motion transformer (GaitTx) with attention pooling
- **Validation:** 5 fold subject level cross validation (no data leakage)
- **Pooled cohorts:** BMCLab, 3DGait, PD GaM, T SDU PD (110 subjects, 2,953 trials)
- **Pretraining:** self supervised masked motion reconstruction

## Results

| Setup | Macro F1 |
|-------|----------|
| Baseline shallow CNN (BMCLab) | 0.29 |
| GaitTx + features + TTA (BMCLab) | 0.49 |
| Pooled cohorts, collapsed labels | 0.59 ± 0.06 |
| + SSL pretraining + ensemble (projected) | 0.62 to 0.70 |

The ~0.6 ceiling reflects inter rater label noise in clinical UPDRS scoring, not engineering. Anyone claiming 0.90 on this task is overfitting.

## Edge deployment

Built for on device inference: under 4GB RAM, exported to ONNX with int8 quantization. Bit width sweep from 8 to 2 bits documents the accuracy/compression tradeoff. Measured CPU latency as phone class proxy.

## Team

Adyuth Chirakala, Rohan Dravid, Parv Mehndiratta (Dougherty Valley HS).

## Stack

Python, PyTorch, ONNX Runtime, Google Colab (T4)
