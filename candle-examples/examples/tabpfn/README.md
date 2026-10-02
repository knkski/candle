# TabPFN

Classify a small numeric table with TabPFN v1, v2, v2.5, v2.6, v3, v3.5,
or v3.5 Fast. The example downloads and caches the selected checkpoint and
loads its native PyTorch or safetensors weights. Running it requires no Python
installation or checkpoint conversion.

```sh
cargo run -p candle-examples --example tabpfn --release -- --cpu --model v3.5
```

For CUDA, build with the `cuda` feature and omit `--cpu`:

```sh
cargo run -p candle-examples --example tabpfn --release --features cuda -- --model v3.5
```

`--model` accepts `v1` (the default), `v2`, `v2.5`, `v2.6`, `v3`, `v3.5`,
and `v3.5-fast`. Every selection runs the same demonstration: 15 labeled rows,
two features, three classes, and three query rows. It prints class probabilities
and predicted labels; the default checkpoints predict classes 0, 1, and 2.

## Checkpoints and options

The default checkpoint sources are pinned to specific Hugging Face revisions:

| Selector | Checkpoint |
| --- | --- |
| `v1` | [Original epoch-42 model](https://huggingface.co/spaces/TabPFN/TabPFNPrediction/blob/b392eb0f0cfb5efe64af2be58b0a60521abed8f8/TabPFN/models_diff/prior_diff_real_checkpoint_n_0_epoch_42.cpkt) |
| `v2` | [tabpfn-v2-classifier-finetuned-zk73skhh.ckpt](https://huggingface.co/Prior-Labs/TabPFN-v2-clf/blob/f851f2a3c941544733b712d8c0f96dfae9b28862/tabpfn-v2-classifier-finetuned-zk73skhh.ckpt) |
| `v2.5` | [tabpfn-v2.5-classifier-v2.5_default.ckpt](https://huggingface.co/Prior-Labs/tabpfn_2_5/blob/6c45f3a6d0d07c6c5f62572e04a0c2929de91b8b/tabpfn-v2.5-classifier-v2.5_default.ckpt) |
| `v2.6` | [tabpfn-v2.6-classifier-v2.6_default.ckpt](https://huggingface.co/Prior-Labs/tabpfn_2_6/blob/24148873a9a5992b429833d07363690dcffd02a8/tabpfn-v2.6-classifier-v2.6_default.ckpt) |
| `v3` | [tabpfn-v3-classifier-v3_default.ckpt](https://huggingface.co/Prior-Labs/tabpfn_3/blob/24a16a89d245878b846555110985634aa2e656d7/tabpfn-v3-classifier-v3_default.ckpt) |
| `v3.5` | [tabpfn-v3.5-20260909.safetensors](https://huggingface.co/Prior-Labs/tabpfn_3_5/blob/06bf2ba35c80a92a3b9abb436b99cf49e7a0365e/tabpfn-v3.5-20260909.safetensors) |
| `v3.5-fast` | [tabpfn-v3.5-fast-20260909.safetensors](https://huggingface.co/Prior-Labs/tabpfn_3_5/blob/06bf2ba35c80a92a3b9abb436b99cf49e7a0365e/tabpfn-v3.5-fast-20260909.safetensors) |

Subsequent runs reuse the Hugging Face cache; `HF_HUB_CACHE` selects its directory.
To bypass downloading, supply a local checkpoint of the selected version:

```sh
cargo run -p candle-examples --example tabpfn --release -- \
  --cpu --model v2.6 --weight-file /path/to/tabpfn-v2.6-classifier-v2.6_default.ckpt
```

For a compatible checkpoint in another repository, use `--model-id OWNER/REPO`,
`--weight-filename FILE`, and optionally `--revision REV`. V1 uses a Hugging Face
Space; the other versions use model repositories. Custom repositories default
to revision `main`. The default repository retains its pinned revision unless
overridden. The checkpoint filename defaults to the selected version's filename.

Architecture configuration is read from the checkpoint for v2 and later; v1
uses the original epoch-42 configuration. `--temperature NUMBER` overrides the
checkpoint's softmax temperature. When the checkpoint provides no temperature,
defaults are 0.8 for v1, 0.9 for v2 through v3, and 1 for v3.5 and v3.5 Fast.

The v2 family also needs fixed column vectors. The example bundles two
`2000 × 48` tables: the original v2 CPU vectors with seed zero, and the upstream
fixed table shared by v2.5/v2.6. They cover 2,000 feature groups. A custom model
can supply `--column-embeddings-file FILE.safetensors`, containing an F32 tensor
named `column_embeddings` with the required number of groups and embedding
width. Custom v2 dimensions or seeds require this option. The bundled vectors
can be reproduced with [prepare_column_embeddings.py](prepare_column_embeddings.py)
using its documented pinned upstream source and PyTorch version.

The bundled `v2_5` tensor is converted from upstream
[`tabpfn_col_embedding.pt`](https://github.com/PriorLabs/TabPFN/blob/de50fac665bdfacba41b049aa3a942c70e7c5001/src/tabpfn/architectures/shared/tabpfn_col_embedding.pt)
to safetensors without changing its values. It is distributed under the
[Apache License 2.0](../../../LICENSE-APACHE), with the upstream attribution:

> TabPFN
>
> Copyright (c) 2026 Prior Labs GmbH

## CUDA temporary-memory retention

The CUDA example sets the current device pool's soft release threshold to
8 GiB (`--cuda-pool-retention-mib 8192`). Reusing freed temporary buffers across
synchronizations avoids repeatedly allocating and releasing GPU memory. This
neither reserves 8 GiB nor caps allocation or total VRAM use; the driver can
release unused memory, and live allocations can exceed the threshold. It does
not reuse table preprocessing, learned activations, predictions, or KV state.
Checkpoint loading and model-weight lifetime are separate.

Use `--cuda-pool-retention-mib 0` to permit releasing unused buffers at
synchronization, or choose a smaller retention threshold on a shared GPU. The
setting affects the current device pool in this process, including other CUDA
users of that pool. CPU, Metal, and CUDA devices without stream-ordered pools
are unchanged.
Direct library users opt in through `CudaDevice::set_memory_pool_release_threshold`;
constructing a Candle device or a TabPFN model does not change the CUDA default.

## CPU execution settings

Use a release build for inference. On supported x86 CPUs, the optional `avx512`
Cargo feature enables AVX-512 kernels in the Rust GEMM dependency. Native CPU
targeting alone does not enable that dependency feature. For example:

```sh
RUSTFLAGS="-C target-cpu=native" cargo build \
  -p candle-examples --example tabpfn --release --features avx512
```

A native build targets the build machine and may not run on an older CPU.
AVX-512 benefits depend on the matrix shapes, CPU and thread count; compare it
with the ordinary release build on the target machine. The MKL and Accelerate
features select different CPU backends.

`RAYON_NUM_THREADS` selects the Rayon/GEMM worker count.
`CANDLE_NUM_THREADS` separately selects the elementwise worker count, defaulting
to the Rayon count; the two pools do not share a total thread budget.
`CANDLE_BARRIER_POOL_SPIN_LIMIT=0` makes idle elementwise workers park immediately.
For example, a twelve-thread GEMM pool and six-thread elementwise pool can be
selected before starting the process:

```sh
RAYON_NUM_THREADS=12 CANDLE_NUM_THREADS=6 CANDLE_BARRIER_POOL_SPIN_LIMIT=0 \
  ./target/release/examples/tabpfn --cpu --model v1
```

These are optional tuning settings, not universal performance defaults. Earlier
v1 CPU measurements also used explicit worker affinity to physical cores. That
requires a machine-specific launcher; merely setting the thread counts above
does not reproduce the same placement or establish the same performance.

## Library interface and scope

Version selection and dispatch follow the Qwen example's structure. The library
modules are `candle_transformers::models::tabpfn::{v1, v2, v3, v3_5}`.
The `v2` module covers v2, v2.5, and v2.6 through one `Model` and a `Variant`
selection. `v2::Config::from_checkpoint(variant, config)` applies the selected
variant's defaults and validates the checkpoint configuration before model
construction; the resulting configuration is immutable.
V3.5 reuses the v3 attention components; v3.5 Fast uses the same
v3.5 architecture with its own weights and fewer ICL layers.

Each model exposes `predict_proba(&train_x, &train_y, &test_x)` and
`predict_proba_with_temperature(..., temperature)`. Features are F32 tensors
shaped `[rows, features]`; training labels are U32 `[rows]` with contiguous
class IDs starting at zero. Training and query tables must use the same feature
order. V1 requires finite features and supports up to 100 features, 10 classes,
and 1,024 training rows. V2 and later handle NaN and infinity with their
architecture's imputation and missing-value indicators; class limits come from
the selected configuration.

Inference includes each architecture's internal preprocessing and computes one
deterministic classification member. It does not implement the Python estimator's
external feature transformations, feature/class permutation ensembles,
regression interface, or KV caching. Probabilities therefore describe a single
member, rather than the full `TabPFNClassifier` ensemble.

## Verification

During development, all seven default checkpoints were compared with upstream
PyTorch in CPU F32. V1 and the six later variants were also checked on CUDA,
with the modern variants evaluated in FP32 with TF32 disabled on an RTX 3090 Ti.
Metal execution has not been verified.

The Rust unit tests cover model configuration, preprocessing, projection packing,
final-layer pruning, and the example's argument and metadata handling:

```sh
cargo test -p candle-transformers --lib models::tabpfn::
cargo test -p candle-examples --example tabpfn
```

The shared CPU/CUDA attention and fused operators have independent numerical tests
in `candle-nn/tests`.
