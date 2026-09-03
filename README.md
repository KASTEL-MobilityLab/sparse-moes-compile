# Adapting Sparse Mixture-of-Experts Layers in CNNs for Compiler-Based Edge Deployment

Code for the paper: *Adapting Sparse Mixture-of-Experts Layers in CNNs for Compiler-Based Edge Deployment* (ECCV 2026).

We replace a single convolutional layer in standard CNN backbones (ResNet-50, DenseNet-121, VGG-11) with a sparse Mixture-of-Experts (MoE) layer and evaluate accuracy, latency, and energy on two edge deployment platforms: an NVIDIA Jetson Orin Nano (PyTorch FP32 and TensorRT INT8) and an FPGA (ONNX→TFLite→TVM→Gemmini). All models are trained on CIFAR-100.

## MoE Formulations

| `--moe_type`   | Description | Jetson FP32 | Jetson TRT INT8 | FPGA TFLite/TVM |
|----------------|-------------|:-----------:|:---------------:|:---------------:|
| `dynamic_top1` | Top-1 routing: per-sample if-else dispatch, true sparse execution | ✓ | ✗ | ✗ |
| `dynamic_top2` | Same, k=2 active experts | ✓ | ✗ | ✗ |
| `static_top1`  | Top-1 routing: all experts computed with tensor masking (torch.compile-friendly) | ✓ | ✓ | ✓ |
| `static_top2`  | Same, k=2 | ✓ | ✓ | ✓ |
| `all_experts`  | All N experts active, weighted sum (no routing) | ✓ | ✓ | ✓ |
| `merged_top1`  | Sparse inference: fuse selected expert weights at runtime into one Conv kernel | ✓ | ✓ | ✗ |
| `merged_top2`  | Same, k=2 | ✓ | ✓ | ✗ |
| `lowrank_top1` | Low-rank: shared static projection V + small per-expert U heads, merged at runtime | ✓ | ✓ | ✓ |
| `lowrank_top2` | Same, k=2 | ✓ | ✓ | ✓ |

**Dynamic** variants use data-dependent indexing (`.item()`, variable expert batch sizes) which breaks static graph export — they run only in eager PyTorch.
**Merged** variants export to ONNX/TFLite but are excluded from FPGA evaluation because full expert-weight merging at every inference step introduces excessive CPU-side overhead on the Gemmini SoC.
**Low-rank** is the only formulation that remains practical across both deployment pipelines.

## DenseNet-121 Variants

DenseNet-121 requires two architectural modifications for compiler-compatible deployment:

| Variant | `--post_bn` | `--max_pool` | Description |
|---------|:-----------:|:------------:|-------------|
| v0 (original) | — | — | Pre-activation BN (BN→ReLU→Conv); BN cannot fuse in ONNX export |
| v1 (post-BN) | ✓ | — | Post-activation BN (Conv→BN→ReLU); all BNs fuse into Conv weights |
| v2 (MaxPool+post-BN) | ✓ | ✓ | Post-BN + transition AvgPool replaced with MaxPool for FPGA compatibility |

`--post_bn` is required for TFLite/TVM deployment (otherwise standalone BatchNorm nodes remain in the graph). `--max_pool` further simplifies the execution graph for FPGA accelerators.

## Repository Structure

```
src/
  moe/             # MoE layer implementations (dynamic_top1, static_top1, merged, lowrank, ...)
  train/           # Training scripts (train_resnet.py, train_vgg.py, train_densenet.py)
  eval/            # Evaluation utilities (eval_checkpoint.py, compute_gflops.py)
  deploy/          # Export and benchmarking (export_tflite.py, jetson_eval.py, tensorrt_quantize.py, ...)
  models/          # LightningModule wrapper and backbone definitions
  datamodules/     # CIFAR-100 datamodule
  figure_*.py      # Paper figure generation scripts
dropped_ablations/ # Architectures and scripts not included in the final paper
```

## Requirements

```bash
pip install -r requirements.txt
```

`requirements.txt` covers training, TFLite export, and ONNX utilities. TensorRT (`tensorrt`, `pycuda`) must be installed separately from the TensorRT SDK and is only needed for `tensorrt_quantize.py`.

## Training

All ResNet experiments use `src/train/train_resnet.py`. VGG-11 and DenseNet-121 have separate scripts.

```bash
# ResNet-50, dynamic top-1, 4 experts (default placement: layer4[1].conv2)
python -m src.train.train_resnet --layers 50 --moe_type dynamic_top1 --num_experts 4 --k 1

# ResNet-50, static top-2, 8 experts
python -m src.train.train_resnet --layers 50 --moe_type static_top2 --num_experts 8 --k 2

# ResNet-50, low-rank top-1, rank 32
python -m src.train.train_resnet --layers 50 --moe_type lowrank_top1 --num_experts 4 --k 1 --rank 32

# VGG-11, merged top-1, 4 experts (features[18])
python -m src.train.train_vgg --use_moe --moe_type merged_top1 --num_experts 4 --k 1

# DenseNet-121 baseline (no MoE)
python -m src.train.train_densenet

# DenseNet-121 v2 (MaxPool + post-BN), low-rank top-1, 4 experts
python -m src.train.train_densenet --use_moe --moe_type lowrank_top1 --num_experts 4 --k 1 \
    --rank 32 --post_bn --max_pool
```

Checkpoints are saved to `checkpoints/`. Runs are logged to Weights & Biases project `resnet-moe`.

**Key training options:**

| Argument | Default | Description |
|----------|---------|-------------|
| `--moe_type` | `dynamic_top1` | MoE formulation (see table above) |
| `--num_experts` | 4 | Number of experts N |
| `--k` | 1 | Active experts per sample |
| `--rank` | 32 | Low-rank bottleneck dimension r (lowrank variants only) |
| `--moe_layer` | 4 | ResNet layer group to replace (3 or 4) |
| `--moe_scope` | `conv2` | Which conv(s) to replace: `conv1`, `conv2`, `both`, `stage` |
| `--aux_loss_coef` | 0.01 | Load-balancing auxiliary loss coefficient |
| `--post_bn` | — | DenseNet: reorder to post-activation BN (Conv→BN→ReLU) |
| `--max_pool` | — | DenseNet: replace transition AvgPool with MaxPool |
| `--max_epochs` | 200 | Training budget |

## Evaluation

```bash
# Test accuracy for any saved checkpoint
python -m src.eval.eval_checkpoint \
    --ckpt checkpoints/r50_4exp_k1_dynamic_top1_best.ckpt \
    --arch resnet50 --moe_type dynamic_top1 --num_experts 4 --k 1

# Compute parameter counts and GFLOPs for all architectures
python -m src.eval.compute_gflops
```

## Deployment

### Quantization

INT8 quantization converts model weights and activations from 32-bit float (FP32) to 8-bit integer (INT8), reducing memory footprint and speeding up inference on hardware with integer arithmetic units. Both deployment pipelines use **post-training quantization (PTQ)**: the model is trained in FP32, then quantized using a small calibration dataset to determine the scaling factors needed to map float values to the INT8 range, therefore no retraining is required.

- **TFLite INT8**: handled inside `export_tflite.py` using the TFLite converter with a representative calibration dataset.
- **TensorRT INT8**: handled inside `tensorrt_quantize.py`; TensorRT profiles the model on calibration batches and stores a calibration cache for fast subsequent exports.

### Jetson Orin Nano

```bash
# PyTorch FP32 latency + energy benchmark (run on device)
python -m src.deploy.jetson_eval

# Export to ONNX + TFLite FP32/INT8 (static, merged, low-rank only)
python -m src.deploy.export_tflite

# Export specific architectures only
python -m src.deploy.export_tflite --archs vgg11 resnet50

# TensorRT INT8 quantization + benchmark (run on Jetson or x86 with TensorRT)
python -m src.deploy.tensorrt_quantize
```

### FPGA (Gemmini via TVM)

The FPGA deployment pipeline starts from the TFLite INT8 graphs produced by `export_tflite.py` and is handled externally by the TVM-based workflow from [Peccia et al.](https://doi.org/10.1145/3626202.3637563) targeting a Gemmini accelerator on an AMD ZCU111 board. Only `static` and `lowrank` variants are supported; `merged` is excluded due to CPU-side overhead from runtime weight merging.

## MoE Target Layers

| Architecture | MoE target | Shape | Spatial |
|---|---|---|---|
| ResNet-50 | `layer4[1].conv2` | 512→512, 3×3 | 4×4 |
| VGG-11 | `features[18]` | 512→512, 3×3 | 2×2 |
| DenseNet-121 | `denseblock4.denselayer16.conv1` | 992→128, 1×1 | 4×4 |

## Citation

If you use this code, please cite our paper (details to be added upon publication).
