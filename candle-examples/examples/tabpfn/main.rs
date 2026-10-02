//! TabPFN classification with version selection and automatic weight downloads.

#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

#[cfg(feature = "accelerate")]
extern crate accelerate_src;

use anyhow::Context;
use candle::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::tabpfn::{v1, v2, v3, v3_5};
use clap::Parser;
use std::path::PathBuf;

mod checkpoint;

const DEFAULT_CUDA_POOL_RETENTION_MIB: u64 = 8192;

enum Model {
    V1(v1::Model),
    V2(v2::Model),
    V3(v3::Model),
    V3_5(v3_5::Model),
}

impl Model {
    fn predict_proba(
        &self,
        train_x: &Tensor,
        train_y: &Tensor,
        test_x: &Tensor,
        temperature: f64,
    ) -> candle::Result<Tensor> {
        match self {
            Self::V1(model) => {
                model.predict_proba_with_temperature(train_x, train_y, test_x, temperature)
            }
            Self::V2(model) => {
                model.predict_proba_with_temperature(train_x, train_y, test_x, temperature)
            }
            Self::V3(model) => {
                model.predict_proba_with_temperature(train_x, train_y, test_x, temperature)
            }
            Self::V3_5(model) => {
                model.predict_proba_with_temperature(train_x, train_y, test_x, temperature)
            }
        }
    }
}

#[derive(Clone, Copy, Debug, clap::ValueEnum, PartialEq, Eq)]
enum WhichModel {
    V1,
    V2,
    #[value(name = "v2.5")]
    V2_5,
    #[value(name = "v2.6")]
    V2_6,
    V3,
    #[value(name = "v3.5")]
    V3_5,
    #[value(name = "v3.5-fast")]
    V3_5Fast,
}

struct ModelSource {
    repo: &'static str,
    revision: &'static str,
    filename: &'static str,
}

impl WhichModel {
    fn source(self) -> ModelSource {
        let (repo, revision, filename) = match self {
            Self::V1 => (
                "TabPFN/TabPFNPrediction",
                "b392eb0f0cfb5efe64af2be58b0a60521abed8f8",
                "TabPFN/models_diff/prior_diff_real_checkpoint_n_0_epoch_42.cpkt",
            ),
            Self::V2 => (
                "Prior-Labs/TabPFN-v2-clf",
                "f851f2a3c941544733b712d8c0f96dfae9b28862",
                "tabpfn-v2-classifier-finetuned-zk73skhh.ckpt",
            ),
            Self::V2_5 => (
                "Prior-Labs/tabpfn_2_5",
                "6c45f3a6d0d07c6c5f62572e04a0c2929de91b8b",
                "tabpfn-v2.5-classifier-v2.5_default.ckpt",
            ),
            Self::V2_6 => (
                "Prior-Labs/tabpfn_2_6",
                "24148873a9a5992b429833d07363690dcffd02a8",
                "tabpfn-v2.6-classifier-v2.6_default.ckpt",
            ),
            Self::V3 => (
                "Prior-Labs/tabpfn_3",
                "24a16a89d245878b846555110985634aa2e656d7",
                "tabpfn-v3-classifier-v3_default.ckpt",
            ),
            Self::V3_5 => (
                "Prior-Labs/tabpfn_3_5",
                "06bf2ba35c80a92a3b9abb436b99cf49e7a0365e",
                "tabpfn-v3.5-20260909.safetensors",
            ),
            Self::V3_5Fast => (
                "Prior-Labs/tabpfn_3_5",
                "06bf2ba35c80a92a3b9abb436b99cf49e7a0365e",
                "tabpfn-v3.5-fast-20260909.safetensors",
            ),
        };
        ModelSource {
            repo,
            revision,
            filename,
        }
    }

    fn default_temperature(self) -> f64 {
        match self {
            Self::V1 => 0.8,
            Self::V2 | Self::V2_5 | Self::V2_6 | Self::V3 => 0.9,
            Self::V3_5 | Self::V3_5Fast => 1.,
        }
    }

    fn check_architecture(self, architecture: Option<&str>) -> anyhow::Result<()> {
        let matches = match self {
            Self::V1 => true,
            Self::V2 => matches!(architecture, None | Some("tabpfn_v2")),
            Self::V2_5 => matches!(architecture, Some("base" | "tabpfn_v2_5")),
            Self::V2_6 => architecture == Some("tabpfn_v2_6"),
            Self::V3 => architecture == Some("tabpfn_v3"),
            Self::V3_5 | Self::V3_5Fast => architecture == Some("tabpfn_v3_5"),
        };
        anyhow::ensure!(
            matches,
            "checkpoint architecture {architecture:?} does not match --model {self}"
        );
        Ok(())
    }
}

impl std::fmt::Display for WhichModel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Self::V1 => "v1",
            Self::V2 => "v2",
            Self::V2_5 => "v2.5",
            Self::V2_6 => "v2.6",
            Self::V3 => "v3",
            Self::V3_5 => "v3.5",
            Self::V3_5Fast => "v3.5-fast",
        })
    }
}

#[derive(Parser, Debug)]
#[command(author, version, about = "Classify a small numeric table with TabPFN")]
struct Args {
    /// Model version.
    #[arg(long, default_value = "v1")]
    model: WhichModel,

    /// Load a local checkpoint instead of downloading weights.
    #[arg(long)]
    weight_file: Option<PathBuf>,

    /// Override the Hugging Face model repository (a Space for v1).
    #[arg(long)]
    model_id: Option<String>,

    /// Override the revision (pinned by default; main for a custom repository).
    #[arg(long)]
    revision: Option<String>,

    /// Override the checkpoint filename within the repository.
    #[arg(long)]
    weight_filename: Option<String>,

    /// Override the softmax temperature stored in the checkpoint.
    #[arg(long)]
    temperature: Option<f64>,

    /// Safetensors with a `column_embeddings` tensor for a custom v2-family model.
    #[arg(long)]
    column_embeddings_file: Option<PathBuf>,

    /// CUDA temporary-memory retention in MiB; 0 releases unused memory at sync.
    /// This is a cache policy, not a reservation or memory limit.
    #[arg(long, default_value_t = DEFAULT_CUDA_POOL_RETENTION_MIB)]
    cuda_pool_retention_mib: u64,

    /// Run on CPU rather than on GPU.
    #[arg(long)]
    cpu: bool,
}

fn cuda_pool_retention_bytes(mib: u64) -> anyhow::Result<u64> {
    mib.checked_mul(1024 * 1024)
        .context("--cuda-pool-retention-mib is too large")
}

// This changes the current process's CUDA pool; CPU/Metal and devices without
// pools are untouched.
fn configure_cuda_pool(device: &Device, retention_mib: u64) -> anyhow::Result<Option<u64>> {
    let bytes = cuda_pool_retention_bytes(retention_mib)?;
    #[cfg(feature = "cuda")]
    if let Device::Cuda(device) = device {
        if device.memory_pool_release_threshold()?.is_some() {
            device.set_memory_pool_release_threshold(bytes)?;
            return Ok(Some(bytes));
        }
    }
    let _ = (device, bytes);
    Ok(None)
}

fn column_embeddings(args: &Args, device: &Device) -> anyhow::Result<Tensor> {
    if let Some(path) = &args.column_embeddings_file {
        return candle::safetensors::load(path, device)?
            .remove("column_embeddings")
            .context("column embeddings file has no column_embeddings tensor");
    }
    let name = if args.model == WhichModel::V2 {
        "v2"
    } else {
        "v2_5"
    };
    candle::safetensors::load_buffer(include_bytes!("column_embeddings.safetensors"), device)?
        .remove(name)
        .context("missing bundled column embeddings")
}

fn load_model(args: &Args, device: &Device) -> anyhow::Result<(Model, f64)> {
    let source = args.model.source();
    let model_file = match &args.weight_file {
        Some(path) => path.clone(),
        None => {
            let api = candle_examples::hub::Api::new()?;
            let model_id = args.model_id.as_deref().unwrap_or(source.repo);
            let revision = args
                .revision
                .as_deref()
                .unwrap_or(if model_id == source.repo {
                    source.revision
                } else {
                    "main"
                });
            let filename = args.weight_filename.as_deref().unwrap_or(source.filename);
            if args.model == WhichModel::V1 {
                let (owner, name) = hf_hub::split_id(model_id);
                candle_examples::hub::Repo::new(api.as_ref().space(owner, name))
                    .with_revision(revision)
                    .get(filename)
            } else {
                api.model(model_id).with_revision(revision).get(filename)
            }
            .with_context(|| {
                format!(
                    "downloading TabPFN {}; use --weight-file for a local checkpoint",
                    args.model
                )
            })?
        }
    };
    if args.model == WhichModel::V1 {
        let vb = VarBuilder::from_pth_with_state(&model_file, DType::F32, "0", device)?;
        let model = Model::V1(v1::Model::new(&v1::Config::v1(), vb.pp("module"))?);
        return Ok((model, args.temperature.unwrap_or(0.8)));
    }

    let metadata = checkpoint::metadata(&model_file)
        .with_context(|| format!("reading metadata from {}", model_file.display()))?;
    args.model
        .check_architecture(metadata.architecture.as_deref())?;
    let temperature = args
        .temperature
        .or(metadata.temperature)
        .unwrap_or(args.model.default_temperature());
    anyhow::ensure!(
        temperature.is_finite() && temperature > 0.,
        "temperature must be finite and positive"
    );
    let vb = if checkpoint::is_safetensors(&model_file) {
        unsafe { VarBuilder::from_mmaped_safetensors(&[&model_file], DType::F32, device)? }
    } else {
        VarBuilder::from_pth_with_state(&model_file, DType::F32, "state_dict", device)?
    };
    let model = match args.model {
        WhichModel::V1 => unreachable!(),
        WhichModel::V2 | WhichModel::V2_5 | WhichModel::V2_6 => {
            let variant = match args.model {
                WhichModel::V2 => v2::Variant::V2,
                WhichModel::V2_5 => v2::Variant::V2_5,
                WhichModel::V2_6 => v2::Variant::V2_6,
                _ => unreachable!(),
            };
            let config = v2::Config::from_checkpoint(variant, metadata.config)?;
            if variant == v2::Variant::V2 {
                anyhow::ensure!(
                    args.column_embeddings_file.is_some()
                        || (config.embedding_dim() == 192 && config.seed() == 0),
                    "custom v2 dimensions or seed require --column-embeddings-file"
                );
            }
            Model::V2(v2::Model::new(
                &config,
                vb,
                &column_embeddings(args, device)?,
            )?)
        }
        WhichModel::V3 => Model::V3(v3::Model::new(
            &serde_json::from_value(metadata.config)?,
            vb,
        )?),
        WhichModel::V3_5 | WhichModel::V3_5Fast => Model::V3_5(v3_5::Model::new(
            &serde_json::from_value(metadata.config)?,
            vb,
        )?),
    };
    Ok((model, temperature))
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    if let Some(temperature) = args.temperature {
        anyhow::ensure!(
            temperature.is_finite() && temperature > 0.,
            "temperature must be finite and positive"
        );
    }
    anyhow::ensure!(
        args.column_embeddings_file.is_none()
            || matches!(
                args.model,
                WhichModel::V2 | WhichModel::V2_5 | WhichModel::V2_6
            ),
        "--column-embeddings-file is only supported for v2, v2.5, and v2.6"
    );
    let device = candle_examples::device(args.cpu)?;
    configure_cuda_pool(&device, args.cuda_pool_retention_mib)?;
    let (model, temperature) = load_model(&args, &device)?;

    // Three clusters with two continuous features and contiguous class labels.
    let train_rows = [
        [-2.0f32, -1.5],
        [-1.8, -1.7],
        [-2.2, -1.3],
        [-1.6, -1.4],
        [-2.1, -1.9],
        [2.0, -1.5],
        [1.8, -1.7],
        [2.2, -1.3],
        [1.6, -1.4],
        [2.1, -1.9],
        [0.0, 2.0],
        [-0.2, 1.8],
        [0.2, 2.2],
        [-0.4, 2.1],
        [0.3, 1.7],
    ];
    let labels = [0u32, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2];
    let test_rows = [[-1.9f32, -1.6], [1.9, -1.6], [0.1, 1.9]];
    // Preprocessing runs on the host and transfers the prepared inputs once.
    let train_x = Tensor::new(&train_rows, &Device::Cpu)?;
    let train_y = Tensor::new(&labels, &Device::Cpu)?;
    let test_x = Tensor::new(&test_rows, &Device::Cpu)?;
    let probabilities = model
        .predict_proba(&train_x, &train_y, &test_x, temperature)?
        .to_vec2::<f32>()?;

    println!(
        "TabPFN {}: 15 training rows, 2 features, 3 classes",
        args.model
    );
    println!(" feature_1  feature_2     P(0)     P(1)     P(2)  prediction");
    for (row, probabilities) in test_rows.iter().zip(probabilities.iter()) {
        let prediction = probabilities
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.total_cmp(b))
            .map(|(class, _)| class)
            .context("model returned no class probabilities")?;
        println!(
            "{:>10.2} {:>10.2} {:>8.4} {:>8.4} {:>8.4} {:>11}",
            row[0], row[1], probabilities[0], probabilities[1], probabilities[2], prediction,
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cuda_pool_retention_cli_and_units() -> anyhow::Result<()> {
        for model in ["v1", "v2", "v3.5-fast"] {
            let args = Args::try_parse_from(["tabpfn", "--model", model])?;
            assert_eq!(
                args.cuda_pool_retention_mib,
                DEFAULT_CUDA_POOL_RETENTION_MIB
            );
        }
        let args = Args::try_parse_from(["tabpfn", "--cuda-pool-retention-mib", "0"])?;
        assert_eq!(args.cuda_pool_retention_mib, 0);
        assert_eq!(cuda_pool_retention_bytes(0)?, 0);
        assert_eq!(cuda_pool_retention_bytes(8192)?, 8 * 1024 * 1024 * 1024);
        let largest = u64::MAX / (1024 * 1024);
        assert!(cuda_pool_retention_bytes(largest).is_ok());
        assert!(cuda_pool_retention_bytes(largest + 1).is_err());
        assert!(Args::try_parse_from(["tabpfn", "--cuda-pool-retention-mib", "-1"]).is_err());
        Ok(())
    }

    #[test]
    fn cpu_does_not_configure_cuda_pool() -> anyhow::Result<()> {
        assert_eq!(configure_cuda_pool(&Device::Cpu, 8192)?, None);
        assert_eq!(configure_cuda_pool(&Device::Cpu, 0)?, None);
        Ok(())
    }
}
