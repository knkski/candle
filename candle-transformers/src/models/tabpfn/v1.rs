//! Original TabPFN transformer for small tabular classification problems.
//!
//! The pretrained transformer treats labeled training rows as context. No gradient
//! updates are needed. [`Model::predict_proba`] implements one deterministic pass
//! with the original normalization, outlier compression and feature padding. It
//! supports finite numeric features and contiguous class IDs. Power transforms and
//! the Python classifier's feature/class permutation ensemble are not implemented.
//!
//! Reference: <https://github.com/PriorLabs/TabPFN/tree/v1.0.0>.

use candle::{DType, Result, Tensor, D};
use candle_nn::{layer_norm, LayerNorm, Linear, Module, VarBuilder};

fn prepare_linear(layer: Linear) -> Result<Linear> {
    if layer.weight().device().is_cpu()
        && !layer.weight().track_op()
        && !candle::utils::has_mkl()
        && !candle::utils::has_accelerate()
    {
        // Preserve the logical [output, input] shape, but store columns
        // contiguously. GEMM can then pack its transposed weight operand with
        // contiguous reads instead of repeatedly gathering strided values.
        let weight = layer.weight().t()?.contiguous()?.t()?;
        Ok(Linear::new(weight, layer.bias().cloned()))
    } else {
        Ok(layer)
    }
}

fn linear(input: usize, output: usize, vb: VarBuilder) -> Result<Linear> {
    prepare_linear(candle_nn::linear(input, output, vb)?)
}

/// Fixed architecture dimensions for the original TabPFN transformer.
///
/// Construct with [`Config::v1`] for the original classification checkpoint.
#[derive(Debug, Clone)]
pub struct Config {
    num_features: usize,
    max_num_classes: usize,
    emsize: usize,
    nhead: usize,
    nhid_factor: usize,
    nlayers: usize,
}

impl Config {
    /// Dimensions of the original epoch-42 classification checkpoint.
    pub fn v1() -> Self {
        Self {
            num_features: 100,
            max_num_classes: 10,
            emsize: 512,
            nhead: 4,
            nhid_factor: 2,
            nlayers: 12,
        }
    }
}

#[derive(Debug, Clone)]
struct Attention {
    qkv: Linear,
    output: Linear,
    heads: usize,
}

impl Attention {
    fn new(config: &Config, vb: VarBuilder) -> Result<Self> {
        let hidden = config.emsize;
        Ok(Self {
            qkv: prepare_linear(Linear::new(
                vb.get((3 * hidden, hidden), "in_proj_weight")?,
                Some(vb.get(3 * hidden, "in_proj_bias")?),
            ))?,
            output: linear(hidden, hidden, vb.pp("out_proj"))?,
            heads: config.nhead,
        })
    }

    fn forward(&self, xs: &Tensor, train_len: usize, only_queries: bool) -> Result<Tensor> {
        let (batch, rows, hidden) = xs.dims3()?;
        let query_start = if only_queries { train_len } else { 0 };
        let query_rows = rows - query_start;
        let head_dim = hidden / self.heads;
        let qkv = self
            .qkv
            .forward(xs)?
            .reshape((batch, rows, 3, self.heads, head_dim))?;
        let blocked_cpu = qkv.device().is_cpu()
            && !candle::utils::has_mkl()
            && !candle::utils::has_accelerate()
            && rayon::current_num_threads() > 1
            && !qkv.track_op();
        let project = |index, start, len| -> Result<Tensor> {
            qkv.narrow(2, index, 1)?
                .squeeze(2)?
                .narrow(1, start, len)?
                .transpose(1, 2)
        };
        let (q, k, v, scale) = {
            let q = project(0, query_start, query_rows)?;
            let (q, scale) = if blocked_cpu {
                // Match Tensor division: compute the reciprocal in f64, then
                // round to F32 before multiplying Q in the kernel's tile buffer.
                (q, (1.0 / (head_dim as f64).sqrt()) as f32)
            } else {
                ((q / (head_dim as f64).sqrt())?, 1.0)
            };
            // All rows query the training context; query rows never become keys.
            let k = project(1, 0, train_len)?.contiguous()?;
            let v = project(2, 0, train_len)?.contiguous()?;
            (q, k, v, scale)
        };
        let xs = if blocked_cpu {
            // Scale directly into tile scratch; do not allocate a full scaled Q.
            candle_nn::attention::cpu_flash::blocked::flash_attn_f32(
                &q.transpose(1, 2)?,
                &k.transpose(1, 2)?,
                &v.transpose(1, 2)?,
                scale,
            )?
        } else {
            let scores = q.matmul(&k.t()?)?;
            let attention = candle_nn::ops::softmax_last_dim(&scores)?;

            attention.matmul(&v)?
        };
        let xs = xs.transpose(1, 2)?.reshape((batch, query_rows, hidden))?;

        self.output.forward(&xs)
    }
}

#[derive(Debug, Clone)]
struct Layer {
    attention: Attention,
    norm1: LayerNorm,
    norm2: LayerNorm,
    linear1: Linear,
    linear2: Linear,
}

impl Layer {
    fn new(config: &Config, vb: VarBuilder) -> Result<Self> {
        let hidden = config.emsize;
        let intermediate = hidden * config.nhid_factor;
        Ok(Self {
            attention: Attention::new(config, vb.pp("self_attn"))?,
            norm1: layer_norm(hidden, 1e-5, vb.pp("norm1"))?,
            norm2: layer_norm(hidden, 1e-5, vb.pp("norm2"))?,
            linear1: linear(hidden, intermediate, vb.pp("linear1"))?,
            linear2: linear(intermediate, hidden, vb.pp("linear2"))?,
        })
    }

    fn forward(&self, xs: &Tensor, train_len: usize, only_queries: bool) -> Result<Tensor> {
        let xs = {
            let residual = if only_queries {
                xs.narrow(1, train_len, xs.dim(1)? - train_len)?
            } else {
                xs.clone()
            };
            let attention = self.attention.forward(xs, train_len, only_queries)?;

            self.norm1.forward_residual(&residual, &attention)?
        };
        let hidden = {
            let hidden = self.linear1.forward_gelu_erf(&xs)?;

            self.linear2.forward(&hidden)?
        };

        self.norm2.forward_residual(&xs, &hidden)
    }
}

/// Original v1 network. Each row is one token, with no positional embedding.
#[derive(Debug, Clone)]
pub struct Model {
    config: Config,
    encoder: Linear,
    y_encoder: Linear,
    layers: Vec<Layer>,
    decoder1: Linear,
    decoder2: Linear,
}

impl Model {
    pub fn new(config: &Config, vb: VarBuilder) -> Result<Self> {
        if vb.dtype() != DType::F32 {
            candle::bail!("TabPFN v1 currently requires F32 weights")
        }
        let hidden = config.emsize;
        let intermediate = hidden * config.nhid_factor;
        let layers = (0..config.nlayers)
            .map(|index| Layer::new(config, vb.pp(format!("transformer_encoder.layers.{index}"))))
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            config: config.clone(),
            encoder: linear(config.num_features, hidden, vb.pp("encoder"))?,
            y_encoder: linear(1, hidden, vb.pp("y_encoder"))?,
            layers,
            decoder1: linear(hidden, intermediate, vb.pp("decoder.0"))?,
            decoder2: linear(intermediate, config.max_num_classes, vb.pp("decoder.2"))?,
        })
    }

    /// Run on already-preprocessed F32 features `[batch, rows, num_features]`
    /// and scalar training labels `[batch, train_len]`. Returns query logits
    /// `[batch, rows - train_len, max_num_classes]`.
    pub fn forward(&self, xs: &Tensor, ys: &Tensor, train_len: usize) -> Result<Tensor> {
        let (batch, rows, features) = xs.dims3()?;
        if batch == 0 || train_len == 0 || train_len >= rows || features != self.config.num_features
        {
            candle::bail!(
                "invalid TabPFN input shape {:?} or train_len {train_len}",
                xs.dims()
            )
        }
        if ys.dims() != [batch, train_len] || xs.dtype() != DType::F32 || ys.dtype() != DType::F32 {
            candle::bail!("TabPFN expects F32 features and F32 labels [batch, train_len]")
        }
        let mut hidden = {
            let encoded = self.encoder.forward(xs)?;
            let train = (encoded.narrow(1, 0, train_len)?
                + self.y_encoder.forward(&ys.unsqueeze(D::Minus1)?)?)?;
            Tensor::cat(&[train, encoded.narrow(1, train_len, rows - train_len)?], 1)?
        };
        for (index, layer) in self.layers.iter().enumerate() {
            // Final training outputs are unused, but their input states
            // still supply the keys and values for the query rows.
            let only_queries = index + 1 == self.layers.len();
            hidden = layer.forward(&hidden, train_len, only_queries)?;
        }

        self.decoder2
            .forward(&self.decoder1.forward_gelu_erf(&hidden)?)
    }

    /// Classify finite numeric rows with a single deterministic prediction pass.
    ///
    /// Features are F32 matrices `[rows, features]`; training labels are U32
    /// `[rows]`, covering every ID in `0..num_classes` (at least two classes).
    /// Training-derived preprocessing runs on CPU, followed by inference on the
    /// model's device. Returns `[test_rows, num_classes]` probabilities using the
    /// original temperature of 0.8, without rotations or power transforms.
    pub fn predict_proba(
        &self,
        train_x: &Tensor,
        train_y: &Tensor,
        test_x: &Tensor,
    ) -> Result<Tensor> {
        self.predict_proba_with_temperature(train_x, train_y, test_x, 0.8)
    }

    /// Classify rows using an explicit positive softmax temperature.
    pub fn predict_proba_with_temperature(
        &self,
        train_x: &Tensor,
        train_y: &Tensor,
        test_x: &Tensor,
        temperature: f64,
    ) -> Result<Tensor> {
        if !temperature.is_finite() || temperature <= 0. {
            candle::bail!("TabPFN temperature must be finite and positive")
        }
        let (train_len, features) = train_x.dims2()?;
        let (test_len, test_features) = test_x.dims2()?;
        if !(2..=1024).contains(&train_len)
            || test_len == 0
            || features == 0
            || features > self.config.num_features
            || features != test_features
        {
            candle::bail!("TabPFN expects 2..=1024 training rows, nonempty queries, and matching 1..={} features", self.config.num_features)
        }
        if train_y.dims() != [train_len] || train_y.dtype() != DType::U32 {
            candle::bail!(
                "TabPFN training labels must be a U32 vector with one label per training row"
            )
        }
        if train_x.dtype() != DType::F32 || test_x.dtype() != DType::F32 {
            candle::bail!("TabPFN feature tables must be F32")
        }
        let labels = train_y.to_vec1::<u32>()?;
        let max_label = labels.iter().copied().max().unwrap() as usize;
        if max_label == 0 || max_label >= self.config.max_num_classes {
            candle::bail!(
                "TabPFN requires 2..={} classes",
                self.config.max_num_classes
            )
        }
        let num_classes = max_label + 1;
        let mut seen = vec![false; num_classes];
        for label in &labels {
            seen[*label as usize] = true;
        }
        if seen.iter().any(|present| !present) {
            candle::bail!("TabPFN labels must cover contiguous class IDs starting at zero")
        }
        let train = train_x.to_vec2::<f32>()?;
        let test = test_x.to_vec2::<f32>()?;
        let processed = preprocess(&train, &test, &self.config)?;
        let device = self.encoder.weight().device();
        let xs = Tensor::from_vec(
            processed,
            (1, train_len + test_len, self.config.num_features),
            device,
        )?;
        let ys = Tensor::from_vec(
            labels.into_iter().map(|label| label as f32).collect(),
            (1, train_len),
            device,
        )?;
        let logits = self.forward(&xs, &ys, train_len)?;

        let logits = (logits.squeeze(0)?.narrow(1, 0, num_classes)? / temperature)?;
        candle_nn::ops::softmax_last_dim(&logits.contiguous()?)
    }
}

/// Training statistics use sample variance (n - 1), as in original torch_nanstd.
fn mean_std(values: &[f32]) -> Result<(f32, f32)> {
    if values.len() < 2 {
        candle::bail!("TabPFN preprocessing needs at least two values per retained feature")
    }
    let mean = values.iter().sum::<f32>() / values.len() as f32;
    let variance =
        values.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / (values.len() - 1) as f32;
    let std = variance.sqrt();
    if !mean.is_finite() || !std.is_finite() {
        candle::bail!("TabPFN feature statistics overflowed F32")
    }
    Ok((mean, std))
}

fn preprocess(train: &[Vec<f32>], test: &[Vec<f32>], config: &Config) -> Result<Vec<f32>> {
    if train
        .iter()
        .chain(test)
        .flatten()
        .any(|value| !value.is_finite())
    {
        candle::bail!("TabPFN v1 currently requires finite numeric features (no NaN or infinity)")
    }
    let mut columns = Vec::new();
    for feature in 0..train[0].len() {
        let train_column = train.iter().map(|row| row[feature]).collect::<Vec<_>>();
        let (mean, std) = mean_std(&train_column)?;
        let mut column = train
            .iter()
            .chain(test)
            .map(|row| ((row[feature] - mean) / (std + 1e-6)).clamp(-100., 100.))
            .collect::<Vec<_>>();
        let context = &column[..train.len()];
        if context.iter().all(|value| *value == context[0]) {
            continue;
        }
        let (mean, std) = mean_std(context)?;
        let retained = context
            .iter()
            .copied()
            .filter(|value| *value >= mean - 4. * std && *value <= mean + 4. * std)
            .collect::<Vec<_>>();
        let (mean, std) = mean_std(&retained)?;
        let (lower, upper) = (mean - 4. * std, mean + 4. * std);
        for value in &mut column {
            *value = value.max(-(1. + value.abs()).ln() + lower);
            *value = value.min((1. + value.abs()).ln() + upper);
        }
        columns.push(column);
    }
    if columns.is_empty() {
        candle::bail!("TabPFN requires at least one nonconstant training feature")
    }
    let scale = config.num_features as f32 / columns.len() as f32;
    let mut output = vec![0f32; (train.len() + test.len()) * config.num_features];
    for (feature, column) in columns.iter().enumerate() {
        for (row, value) in column.iter().enumerate() {
            output[row * config.num_features + feature] = value * scale;
        }
    }
    Ok(output)
}

#[cfg(test)]
mod tests;
