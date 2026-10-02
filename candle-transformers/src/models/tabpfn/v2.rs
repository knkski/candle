//! TabPFN v2, v2.5, and v2.6: attention across both feature groups and rows.
//!
//! This implements one classification pass through the architecture, including its
//! internal feature preprocessing. The sklearn estimator's preprocessing ensemble
//! and probability calibration are separate from this model.
//!
//! Select a [`Variant`] when reading checkpoint configuration with
//! [`Config::from_checkpoint`]. V2.5 adds thinking rows and optional MLP feature
//! encoding; v2.6 additionally uses learned RMSNorm.
//!
//! Reference: <https://github.com/PriorLabs/TabPFN>.

use candle::{DType, Device, IndexOp, Result, Tensor, D};
use candle_nn::{linear, linear_no_bias, LayerNorm, Linear, Module, VarBuilder};

/// Architecture variant, selected independently of the checkpoint's weight layout.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Variant {
    V2,
    V2_5,
    V2_6,
}

impl Variant {
    fn hidden_factor(self) -> usize {
        if self == Self::V2 {
            4
        } else {
            2
        }
    }
}

// These private schemas retain each release's defaults and ignore unused metadata.
#[derive(Debug, serde::Deserialize)]
#[serde(default)]
struct V2CheckpointConfig {
    emsize: usize,
    nlayers: usize,
    nhead: usize,
    features_per_group: usize,
    max_num_classes: usize,
    seed: u64,
}

impl Default for V2CheckpointConfig {
    fn default() -> Self {
        Self {
            emsize: 192,
            nlayers: 12,
            nhead: 6,
            features_per_group: 2,
            max_num_classes: 10,
            seed: 0,
        }
    }
}

#[derive(Debug, serde::Deserialize)]
#[serde(rename_all = "lowercase")]
enum EncoderType {
    Linear,
    Mlp,
}

#[derive(Debug, serde::Deserialize)]
#[serde(default)]
struct V2_5CheckpointConfig {
    emsize: usize,
    nlayers: usize,
    nhead: usize,
    features_per_group: usize,
    max_num_classes: usize,
    num_thinking_rows: usize,
    encoder_type: EncoderType,
    encoder_mlp_hidden_dim: usize,
}

impl Default for V2_5CheckpointConfig {
    fn default() -> Self {
        Self {
            emsize: 192,
            nlayers: 24,
            nhead: 3,
            features_per_group: 3,
            max_num_classes: 10,
            num_thinking_rows: 64,
            encoder_type: EncoderType::Linear,
            encoder_mlp_hidden_dim: 1024,
        }
    }
}

/// Immutable, validated configuration for a v2-family model.
#[derive(Debug, Clone)]
pub struct Config {
    variant: Variant,
    emsize: usize,
    nlayers: usize,
    nhead: usize,
    features_per_group: usize,
    max_num_classes: usize,
    thinking_rows: usize,
    encoder_hidden: Option<usize>,
    seed: u64,
}

impl Config {
    /// Resolve checkpoint fields and validate all configuration-derived dimensions.
    ///
    /// The variant must be selected explicitly: v2.5 and v2.6 use the same schema.
    /// Missing fields use that variant's defaults; unused upstream fields are ignored.
    pub fn from_checkpoint(variant: Variant, value: serde_json::Value) -> Result<Self> {
        if !value.is_object() {
            candle::bail!("TabPFN checkpoint config must be a dictionary")
        }
        let (raw, seed) = match variant {
            Variant::V2 => {
                let raw: V2CheckpointConfig =
                    serde_json::from_value(value).map_err(candle::Error::wrap)?;
                (
                    V2_5CheckpointConfig {
                        emsize: raw.emsize,
                        nlayers: raw.nlayers,
                        nhead: raw.nhead,
                        features_per_group: raw.features_per_group,
                        max_num_classes: raw.max_num_classes,
                        num_thinking_rows: 0,
                        ..Default::default()
                    },
                    raw.seed,
                )
            }
            Variant::V2_5 | Variant::V2_6 => (
                serde_json::from_value(value).map_err(candle::Error::wrap)?,
                0,
            ),
        };
        if raw.emsize < 4
            || raw.nlayers == 0
            || raw.features_per_group == 0
            || raw.max_num_classes < 2
        {
            candle::bail!("invalid TabPFN {variant:?} dimensions: {raw:?}")
        }
        if raw.nhead == 0 || !raw.emsize.is_multiple_of(raw.nhead) {
            candle::bail!(
                "embedding dimension {} must be divisible by nonzero head count {}",
                raw.emsize,
                raw.nhead
            )
        }
        if variant == Variant::V2 && !matches!(raw.features_per_group, 1 | 2) {
            candle::bail!("TabPFN v2 supports one or two features per group")
        }
        let encoder_hidden = match raw.encoder_type {
            EncoderType::Linear => None,
            EncoderType::Mlp if raw.encoder_mlp_hidden_dim == 0 => {
                candle::bail!("MLP encoder hidden dimension must be positive")
            }
            EncoderType::Mlp => Some(raw.encoder_mlp_hidden_dim),
        };
        let e = raw.emsize;
        let intermediate = e.checked_mul(variant.hidden_factor());
        let encoder = encoder_hidden.unwrap_or(e);
        for (name, size) in [
            ("attention", e.checked_mul(e).and_then(|n| n.checked_mul(3))),
            ("feed-forward", intermediate.and_then(|n| n.checked_mul(e))),
            (
                "decoder",
                intermediate.and_then(|n| n.checked_mul(raw.max_num_classes)),
            ),
            (
                "feature encoder",
                raw.features_per_group
                    .checked_mul(2)
                    .and_then(|n| n.checked_mul(encoder)),
            ),
            ("encoder projection", encoder.checked_mul(e)),
            ("thinking rows", raw.num_thinking_rows.checked_mul(e)),
        ] {
            if size.is_none() {
                candle::bail!("TabPFN {name} dimensions overflow usize")
            }
        }
        Ok(Self {
            variant,
            emsize: raw.emsize,
            nlayers: raw.nlayers,
            nhead: raw.nhead,
            features_per_group: raw.features_per_group,
            max_num_classes: raw.max_num_classes,
            thinking_rows: raw.num_thinking_rows,
            encoder_hidden,
            seed,
        })
    }

    pub fn embedding_dim(&self) -> usize {
        self.emsize
    }

    /// Original v2 column-embedding seed; later variants use a fixed table.
    pub fn seed(&self) -> u64 {
        self.seed
    }
}

#[derive(Debug, Clone)]
struct Attention {
    projections: super::attention_projection::AttentionProjections,
    out: Linear,
}

impl Attention {
    fn new(cfg: &Config, vb: VarBuilder, legacy: bool) -> Result<Self> {
        let e = cfg.emsize;
        let h = cfg.nhead;
        let (q, k, v, out) = if legacy {
            // Original base checkpoints store (qkv, heads, head_dim, input).
            let qkv = vb.get((3, h, e / h, e), "_w_qkv")?;
            let projection =
                |i| -> Result<Linear> { Ok(Linear::new(qkv.i(i)?.reshape((e, e))?, None)) };
            (
                projection(0)?,
                projection(1)?,
                projection(2)?,
                Linear::new(
                    vb.get((h, e / h, e), "_w_out")?
                        .reshape((e, e))?
                        .t()?
                        .contiguous()?,
                    None,
                ),
            )
        } else {
            (
                linear_no_bias(e, e, vb.pp("q_projection"))?,
                linear_no_bias(e, e, vb.pp("k_projection"))?,
                linear_no_bias(e, e, vb.pp("v_projection"))?,
                linear_no_bias(e, e, vb.pp("out_projection"))?,
            )
        };
        Ok(Self {
            projections: super::attention_projection::AttentionProjections::new(
                q,
                k,
                v,
                h,
                h,
                e / h,
            )?,
            out,
        })
    }

    fn attend(q: &Tensor, k: &Tensor, v: &Tensor) -> Result<Tensor> {
        if candle_nn::attention::cuda_f32::is_supported(q, k, v) {
            return candle_nn::attention::cuda_f32::sdpa(
                q,
                k,
                v,
                1. / (q.dim(D::Minus1)? as f32).sqrt(),
            );
        }
        let scores = (q.contiguous()?.broadcast_matmul(&k.contiguous()?.t()?)?
            / (q.dim(D::Minus1)? as f64).sqrt())?;
        candle_nn::ops::softmax_last_dim(&scores)?.broadcast_matmul(&v.contiguous()?)
    }

    fn forward(&self, xs: &Tensor, train_len: Option<usize>) -> Result<Tensor> {
        let (b, n, e) = xs.dims3()?;
        let context = match train_len {
            Some(t) => xs.narrow(1, 0, t)?,
            None => xs.clone(),
        };
        let (q, k, v) = self
            .projections
            .forward(xs, &context, train_len.is_none())?;
        let heads_first = |x: Tensor| -> Result<Tensor> {
            let x = x.transpose(1, 2)?;
            if xs.device().is_cuda() {
                Ok(x)
            } else {
                x.contiguous()
            }
        };
        let (q, k, v) = (heads_first(q)?, heads_first(k)?, heads_first(v)?);
        let xs = match train_len {
            // Training rows use all heads; test rows share the first K/V head.
            Some(t) if t < n => {
                let train = Self::attend(&q.narrow(2, 0, t)?, &k, &v)?;
                let test = Self::attend(
                    &q.narrow(2, t, n - t)?,
                    &k.narrow(1, 0, 1)?,
                    &v.narrow(1, 0, 1)?,
                )?;
                Tensor::cat(&[train.transpose(1, 2)?, test.transpose(1, 2)?], 1)?
            }
            _ => Self::attend(&q, &k, &v)?.transpose(1, 2)?,
        };
        self.out.forward(&xs.reshape((b, n, e))?)
    }

    fn forward_cross(
        &self,
        query: &Tensor,
        context: &Tensor,
        first_kv_head: bool,
    ) -> Result<Tensor> {
        let (batch, rows, hidden) = query.dims3()?;
        let (q, k, v) = self.projections.forward(query, context, false)?;
        let (k, v) = if first_kv_head {
            (k.narrow(2, 0, 1)?, v.narrow(2, 0, 1)?)
        } else {
            (k, v)
        };
        let output = Self::attend(
            &q.transpose(1, 2)?,
            &k.transpose(1, 2)?,
            &v.transpose(1, 2)?,
        )?;
        self.out
            .forward(&output.transpose(1, 2)?.reshape((batch, rows, hidden))?)
    }
}

#[derive(Debug, Clone)]
struct BlockNorm {
    inner: LayerNorm,
    // The nonaffine layer norm still uses the fused kernel's affine interface.
    // Its unit scale and zero bias are fixed model inputs, allocated once.
    zero_bias: Option<Tensor>,
}

impl BlockNorm {
    fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let inner = if cfg.variant == Variant::V2_6 {
            LayerNorm::rms_norm(vb.get(cfg.emsize, "weight")?, f32::EPSILON as f64)
        } else {
            LayerNorm::new_no_bias(Tensor::ones(cfg.emsize, vb.dtype(), vb.device())?, 1e-5)
        };
        let zero_bias = if inner.remove_mean() && vb.device().is_cuda() {
            Some(Tensor::zeros(cfg.emsize, vb.dtype(), vb.device())?)
        } else {
            None
        };
        Ok(Self { inner, zero_bias })
    }

    fn forward_residual(&self, x: &Tensor, residual: &Tensor) -> Result<Tensor> {
        if x.device().is_cuda()
            && x.dtype() == DType::F32
            && !x.track_op()
            && !residual.track_op()
            && !self.inner.weight().track_op()
        {
            if let Some(bias) = &self.zero_bias {
                return candle_nn::ops::residual_layer_norm(
                    x,
                    residual,
                    self.inner.weight(),
                    bias,
                    self.inner.eps() as f32,
                );
            }
            return candle_nn::ops::residual_rms_norm(
                x,
                residual,
                self.inner.weight(),
                self.inner.eps() as f32,
            );
        }
        self.inner.forward(&(x + residual)?)
    }

    fn forward_residual_transpose(&self, x: &Tensor, residual: &Tensor) -> Result<Tensor> {
        if x.device().is_cuda()
            && x.dtype() == DType::F32
            && !x.track_op()
            && !residual.track_op()
            && !self.inner.weight().track_op()
        {
            if let Some(bias) = &self.zero_bias {
                return candle_nn::ops::residual_layer_norm_transpose(
                    x,
                    residual,
                    self.inner.weight(),
                    bias,
                    self.inner.eps() as f32,
                );
            }
            return candle_nn::ops::residual_rms_norm_transpose(
                x,
                residual,
                self.inner.weight(),
                self.inner.eps() as f32,
            );
        }
        self.forward_residual(x, residual)?
            .transpose(0, 1)?
            .contiguous()
    }
}

#[derive(Debug, Clone)]
struct Block {
    across_features: Attention,
    across_rows: Attention,
    norms: [BlockNorm; 3],
    linear1: Linear,
    linear2: Linear,
}

impl Block {
    fn new(cfg: &Config, vb: VarBuilder, legacy: bool) -> Result<Self> {
        let e = cfg.emsize;
        let norm = |name| BlockNorm::new(cfg, vb.pp(name));
        Ok(Self {
            across_features: Attention::new(
                cfg,
                vb.pp(if legacy {
                    "self_attn_between_features"
                } else {
                    "per_sample_attention_between_features"
                }),
                legacy,
            )?,
            across_rows: Attention::new(
                cfg,
                vb.pp(if legacy {
                    "self_attn_between_items"
                } else {
                    "per_column_attention_between_cells"
                }),
                legacy,
            )?,
            norms: [
                norm("layernorm_mha1")?,
                norm("layernorm_mha2")?,
                norm("layernorm_mlp")?,
            ],
            linear1: linear_no_bias(
                e,
                e * cfg.variant.hidden_factor(),
                vb.pp(if legacy { "mlp.linear1" } else { "mlp.0" }),
            )?,
            linear2: linear_no_bias(
                e * cfg.variant.hidden_factor(),
                e,
                vb.pp(if legacy { "mlp.linear2" } else { "mlp.2" }),
            )?,
        })
    }

    fn forward(&self, xs: &Tensor, train_len: usize) -> Result<Tensor> {
        let (rows, columns, hidden) = xs.dims3()?;
        let xs = self.norms[0]
            .forward_residual_transpose(xs, &self.across_features.forward(xs, None)?)?;
        let xs = self.norms[1]
            .forward_residual_transpose(&xs, &self.across_rows.forward(&xs, Some(train_len))?)?;

        let xs = xs.reshape((rows * columns, hidden))?;
        let mlp = self
            .linear2
            .forward(&self.linear1.forward(&xs)?.gelu_erf()?)?;
        self.norms[2]
            .forward_residual(&xs, &mlp)?
            .reshape((rows, columns, hidden))
    }

    /// Only the query target-column cells are consumed after the final block.
    /// Feature attention still reads every column for every row. Row attention
    /// then needs only the target column's training context, including thinking
    /// rows; its updated training outputs are never used by the query outputs.
    fn forward_final(&self, xs: &Tensor, train_len: usize) -> Result<Tensor> {
        let (rows, columns, hidden) = xs.dims3()?;
        let query_rows = rows - train_len;
        let xs = {
            let target = xs.narrow(1, columns - 1, 1)?.contiguous()?;
            let attention = self.across_features.forward_cross(&target, xs, false)?;
            self.norms[0].forward_residual(&target, &attention)?
        };
        let xs = {
            let xs = xs.reshape((1, rows, hidden))?;
            let query = xs.narrow(1, train_len, query_rows)?;
            let context = xs.narrow(1, 0, train_len)?;
            let attention = self.across_rows.forward_cross(&query, &context, true)?;
            self.norms[1].forward_residual(&query, &attention)?
        };

        let xs = xs.reshape((query_rows, hidden))?;
        let mlp = self
            .linear2
            .forward(&self.linear1.forward(&xs)?.gelu_erf()?)?;
        self.norms[2].forward_residual(&xs, &mlp)
    }
}

#[derive(Debug, Clone)]
enum FeatureEmbedder {
    Linear(Linear),
    Mlp(Linear, Linear),
}

impl FeatureEmbedder {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        match self {
            Self::Linear(layer) => layer.forward(xs),
            Self::Mlp(first, second) => second.forward(&first.forward(xs)?.gelu_erf()?),
        }
    }
}

/// Shared model for v2, v2.5, and v2.6 classification checkpoints.
#[derive(Debug, Clone)]
pub struct Model {
    config: Config,
    device: Device,
    features: FeatureEmbedder,
    targets: Linear,
    column_projection: Linear,
    column_embeddings: Tensor,
    thinking_rows: Option<Tensor>,
    blocks: Vec<Block>,
    output1: Linear,
    output2: Linear,
}

impl Model {
    /// Load weights with the selected variant and fixed column vectors `[groups, emsize / 4]`.
    pub fn new(config: &Config, vb: VarBuilder, column_embeddings: &Tensor) -> Result<Self> {
        if vb.dtype() != DType::F32 {
            candle::bail!("TabPFN currently requires F32 weights")
        }
        let e = config.emsize;
        let (_, width) = column_embeddings.dims2()?;
        if width != e / 4 {
            candle::bail!("column embedding width must be {}, got {width}", e / 4)
        }
        let legacy =
            vb.contains_tensor("transformer_encoder.layers.0.self_attn_between_features._w_qkv");
        let feature_prefix = if legacy {
            (0..12)
                .map(|i| format!("encoder.{i}"))
                .find(|prefix| {
                    vb.contains_tensor(&format!("{prefix}.layer.weight"))
                        || vb.contains_tensor(&format!("{prefix}.mlp.0.weight"))
                })
                .ok_or_else(|| {
                    candle::Error::Msg("missing TabPFN feature encoder weights".into())
                })?
        } else {
            "feature_group_embedder".into()
        };
        let features = match config.encoder_hidden {
            None => FeatureEmbedder::Linear(linear_no_bias(
                config.features_per_group * 2,
                e,
                vb.pp(if legacy {
                    format!("{feature_prefix}.layer")
                } else {
                    feature_prefix.clone()
                }),
            )?),
            Some(hidden) => {
                let prefix = if legacy {
                    format!("{feature_prefix}.mlp")
                } else {
                    feature_prefix
                };
                FeatureEmbedder::Mlp(
                    linear_no_bias(
                        config.features_per_group * 2,
                        hidden,
                        vb.pp(format!("{prefix}.0")),
                    )?,
                    linear_no_bias(hidden, e, vb.pp(format!("{prefix}.2")))?,
                )
            }
        };
        let target_prefix = if legacy {
            "y_encoder.2.layer"
        } else {
            "target_embedder"
        };
        let decoder_prefix = if legacy {
            "decoder_dict.standard"
        } else {
            "output_projection"
        };
        let thinking_rows = if config.thinking_rows > 0 {
            Some(vb.get(
                (config.thinking_rows, e),
                if legacy {
                    "add_thinking_tokens.row_token_values"
                } else {
                    "add_thinking_rows.row_token_values_TE"
                },
            )?)
        } else {
            None
        };
        let blocks = (0..config.nlayers)
            .map(|i| {
                Block::new(
                    config,
                    vb.pp(if legacy {
                        format!("transformer_encoder.layers.{i}")
                    } else {
                        format!("blocks.{i}")
                    }),
                    legacy,
                )
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            features,
            targets: linear(2, e, vb.pp(target_prefix))?,
            column_projection: linear(e / 4, e, vb.pp("feature_positional_embedding_embeddings"))?,
            column_embeddings: column_embeddings
                .to_device(vb.device())?
                .to_dtype(DType::F32)?,
            thinking_rows,
            blocks,
            output1: linear(
                e,
                e * config.variant.hidden_factor(),
                vb.pp(format!("{decoder_prefix}.0")),
            )?,
            output2: linear(
                e * config.variant.hidden_factor(),
                config.max_num_classes,
                vb.pp(format!("{decoder_prefix}.2")),
            )?,
            device: vb.device().clone(),
            config: config.clone(),
        })
    }

    fn validate_inputs(
        &self,
        xs: &Tensor,
        ys: &Tensor,
        test_x: &Tensor,
    ) -> Result<(usize, usize, usize)> {
        let (train, features) = xs.dims2()?;
        let (test, test_features) = test_x.dims2()?;
        if train == 0
            || test == 0
            || features == 0
            || features != test_features
            || ys.dims() != [train]
        {
            candle::bail!("expected nonempty train/test tables with matching features and one label per training row")
        }
        if xs.dtype() != DType::F32 || test_x.dtype() != DType::F32 || ys.dtype() != DType::U32 {
            candle::bail!("TabPFN expects F32 features and U32 labels")
        }
        let labels = ys.to_vec1::<u32>()?;
        let highest_label = labels.iter().copied().max().unwrap() as usize;
        if highest_label == 0 || highest_label >= self.config.max_num_classes {
            candle::bail!(
                "TabPFN requires between 2 and {} classes",
                self.config.max_num_classes
            )
        }
        let classes = highest_label + 1;
        let mut seen = vec![false; classes];
        for label in labels {
            seen[label as usize] = true;
        }
        if seen.contains(&false) {
            candle::bail!("class IDs must be contiguous starting at zero")
        }
        Ok((train, test, classes))
    }

    /// Nonlearned preprocessing on the host. Returns grouped features
    /// `[rows, groups, 2 * features_per_group]` and targets `[rows, 1, 2]`.
    /// No learned feature, label, or positional projection is performed here.
    fn prepare_inputs(
        &self,
        train_x: &Tensor,
        train_y: &Tensor,
        test_x: &Tensor,
    ) -> Result<(Tensor, Tensor)> {
        let (train, test, _) = self.validate_inputs(train_x, train_y, test_x)?;
        let rows = train + test;
        let mut data = train_x.to_vec2::<f32>()?;
        data.extend(test_x.to_vec2::<f32>()?);
        let (features, groups) = preprocess_features(
            &data,
            train,
            self.config.features_per_group,
            self.config.variant != Variant::V2,
        );
        if groups == 0 || groups > self.column_embeddings.dim(0)? {
            candle::bail!("invalid feature group count {groups} for column embedding table")
        }
        let features = Tensor::from_vec(
            features,
            (rows, groups, self.config.features_per_group * 2),
            &Device::Cpu,
        )?;
        let labels = train_y.to_vec1::<u32>()?;
        // With contiguous classes, v2's rank(mean) is exactly ceil(mean).
        let missing_label = (labels.iter().map(|&v| v as f32).sum::<f32>() / train as f32).ceil();
        let mut targets = Vec::with_capacity(rows * 2);
        for label in labels {
            targets.extend([label as f32, 0.]);
        }
        for _ in 0..test {
            targets.extend([missing_label, -2.]);
        }
        Ok((
            features,
            Tensor::from_vec(targets, (rows, 1, 2), &Device::Cpu)?,
        ))
    }

    /// Learned network on prepared tensors resident on the model's device.
    /// Includes all encoders, learned positional projection, thinking rows,
    /// attention blocks and the decoder, pruning final-block outputs unused by
    /// the query decoder. Returns `[query, max_num_classes]`.
    fn forward_prepared(
        &self,
        features: &Tensor,
        targets: &Tensor,
        train: usize,
    ) -> Result<Tensor> {
        let (rows, groups, width) = features.dims3()?;
        if train == 0
            || train >= rows
            || groups == 0
            || groups > self.column_embeddings.dim(0)?
            || width != self.config.features_per_group * 2
            || targets.dims() != [rows, 1, 2]
            || features.dtype() != DType::F32
            || targets.dtype() != DType::F32
            || !features.device().same_device(&self.device)
            || !targets.device().same_device(&self.device)
        {
            candle::bail!("invalid prepared TabPFN v2 tensors or device")
        }
        let mut xs = {
            let positions = self
                .column_projection
                .forward(&self.column_embeddings.narrow(0, 0, groups)?)?;
            let features = self.features.forward(features)?.broadcast_add(&positions)?;
            let targets = self.targets.forward(targets)?;
            let mut xs = Tensor::cat(&[features, targets], 1)?;
            if let Some(thinking) = &self.thinking_rows {
                let thinking = thinking.unsqueeze(1)?.broadcast_as((
                    self.config.thinking_rows,
                    groups + 1,
                    self.config.emsize,
                ))?;
                xs = Tensor::cat(&[thinking, xs], 0)?;
            }
            xs
        };
        let context = train + self.config.thinking_rows;
        for (index, block) in self.blocks.iter().enumerate() {
            xs = if index + 1 == self.blocks.len() {
                block.forward_final(&xs, context)?
            } else {
                block.forward(&xs, context)?
            };
        }

        self.output2
            .forward(&self.output1.forward(&xs)?.gelu_erf()?)
    }

    /// Uncalibrated logits for a single table, including internal preprocessing.
    pub fn forward(&self, train_x: &Tensor, train_y: &Tensor, test_x: &Tensor) -> Result<Tensor> {
        let (features, targets) = self.prepare_inputs(train_x, train_y, test_x)?;
        let (features, targets) = (
            features.to_device(&self.device)?,
            targets.to_device(&self.device)?,
        );

        self.forward_prepared(&features, &targets, train_x.dim(0)?)
    }

    /// Single-member classification probabilities at the default temperature of 0.9.
    pub fn predict_proba(
        &self,
        train_x: &Tensor,
        train_y: &Tensor,
        test_x: &Tensor,
    ) -> Result<Tensor> {
        self.predict_proba_with_temperature(train_x, train_y, test_x, 0.9)
    }

    pub fn predict_proba_with_temperature(
        &self,
        train_x: &Tensor,
        train_y: &Tensor,
        test_x: &Tensor,
        temperature: f64,
    ) -> Result<Tensor> {
        if !temperature.is_finite() || temperature <= 0. {
            candle::bail!("softmax temperature must be finite and positive")
        }
        let classes = self.validate_inputs(train_x, train_y, test_x)?.2;
        let logits = self.forward(train_x, train_y, test_x)?;

        candle_nn::ops::softmax_last_dim(&(logits.narrow(1, 0, classes)? / temperature)?)
    }
}

/// Match the upstream nonlearned encoder: constant removal, imputation, sample
/// standardization, clipping, group scaling, and NaN/+Inf/-Inf indicators.
fn preprocess_features(
    data: &[Vec<f32>],
    train: usize,
    group_size: usize,
    global_constants: bool,
) -> (Vec<f32>, usize) {
    let rows = data.len();
    let columns = data[0].len();
    let nonconstant = |column: usize| data[1..].iter().any(|row| row[column] != data[0][column]);
    let mut selected: Vec<Vec<usize>> = if global_constants {
        let cols = (0..columns).filter(|&i| nonconstant(i)).collect::<Vec<_>>();
        cols.chunks(group_size).map(<[usize]>::to_vec).collect()
    } else {
        (0..columns.div_ceil(group_size))
            .map(|g| {
                (g * group_size..((g + 1) * group_size).min(columns))
                    .filter(|&i| nonconstant(i))
                    .collect()
            })
            .collect()
    };
    // A single empty group is retained in v2 only when there are other groups.
    if !global_constants && selected.len() == 1 && selected[0].is_empty() {
        selected.clear();
    }
    let groups = selected.len();
    let mut output = vec![0f32; rows * groups * group_size * 2];
    for (g, columns) in selected.iter().enumerate() {
        let mut normalized = vec![vec![0f32; rows]; columns.len()];
        let mut indicators = vec![vec![0f32; rows]; columns.len()];
        for (f, &column) in columns.iter().enumerate() {
            let finite = data[..train]
                .iter()
                .map(|r| r[column])
                .filter(|v| v.is_finite())
                .collect::<Vec<_>>();
            let imputation = finite.iter().sum::<f32>() / finite.len().max(1) as f32;
            let values = data
                .iter()
                .map(|row| {
                    if row[column].is_finite() {
                        row[column]
                    } else {
                        imputation
                    }
                })
                .collect::<Vec<_>>();
            let mean = values[..train].iter().sum::<f32>() / train as f32;
            let variance = values[..train]
                .iter()
                .map(|v| (v - mean).powi(2))
                .sum::<f32>()
                / train.saturating_sub(1).max(1) as f32;
            let std = if variance == 0. || train == 1 {
                1.
            } else {
                variance.sqrt()
            };
            for (r, row) in data.iter().enumerate() {
                normalized[f][r] = ((values[r] - mean) / (std + f32::EPSILON)).clamp(-100., 100.);
                indicators[f][r] = if row[column].is_nan() {
                    -2.
                } else if row[column] == f32::INFINITY {
                    2.
                } else if row[column] == f32::NEG_INFINITY {
                    4.
                } else {
                    0.
                };
            }
        }
        let used = normalized
            .iter()
            .filter(|values| values[1..].iter().any(|v| *v != values[0]))
            .count();
        let scale = (group_size as f32 / used.max(1) as f32).sqrt();
        for f in 0..columns.len() {
            let varies = normalized[f][1..].iter().any(|v| *v != normalized[f][0]);
            for r in 0..rows {
                let base = (r * groups + g) * group_size * 2;
                output[base + f] = if varies { normalized[f][r] * scale } else { 0. };
                output[base + group_size + f] = indicators[f][r];
            }
        }
    }
    (output, groups)
}

#[cfg(test)]
mod tests;
