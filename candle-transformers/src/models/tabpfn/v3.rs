//! TabPFN v3 classification transformer.
//!
//! Distribution embeddings, rotary feature aggregation, and row attention with
//! learned query scaling are followed by an attention-based classification head.
//! Inference uses one deterministic member, without the Python estimator's
//! feature transformations or permutation ensemble.
//!
//! Reference: <https://github.com/PriorLabs/TabPFN>.

use candle::{DType, Result, Tensor, D};
use candle_nn::{embedding, linear, linear_no_bias, Embedding, Linear, Module, VarBuilder};

/// Architecture configuration stored in TabPFN v3 checkpoints.
#[derive(Debug, Clone, serde::Deserialize, serde::Serialize)]
#[serde(default)]
pub struct Config {
    pub max_num_classes: usize,
    pub embed_dim: usize,
    pub dist_embed_num_blocks: usize,
    pub dist_embed_num_heads: usize,
    pub dist_embed_num_inducing_points: usize,
    pub feature_group_size: usize,
    pub feat_agg_num_blocks: usize,
    pub feat_agg_num_heads: usize,
    pub feat_agg_num_cls_tokens: usize,
    pub feat_agg_rope_base: f64,
    pub use_rope: bool,
    pub nlayers: usize,
    pub icl_num_heads: usize,
    pub icl_num_kv_heads: Option<usize>,
    pub icl_num_kv_heads_test: Option<usize>,
    pub decoder_head_dim: usize,
    pub decoder_num_heads: usize,
    pub decoder_use_softmax_scaling: bool,
    pub ff_factor: usize,
    pub softmax_scaling_mlp_hidden_dim: usize,
    pub layernorm_elementwise_affine: bool,
    pub use_nan_indicators: bool,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            max_num_classes: 10,
            embed_dim: 128,
            dist_embed_num_blocks: 3,
            dist_embed_num_heads: 8,
            dist_embed_num_inducing_points: 128,
            feature_group_size: 3,
            feat_agg_num_blocks: 3,
            feat_agg_num_heads: 8,
            feat_agg_num_cls_tokens: 4,
            feat_agg_rope_base: 100_000.,
            use_rope: true,
            nlayers: 24,
            icl_num_heads: 8,
            icl_num_kv_heads: None,
            icl_num_kv_heads_test: None,
            decoder_head_dim: 64,
            decoder_num_heads: 6,
            decoder_use_softmax_scaling: false,
            ff_factor: 2,
            softmax_scaling_mlp_hidden_dim: 64,
            layernorm_elementwise_affine: true,
            use_nan_indicators: true,
        }
    }
}

impl Config {
    pub(super) fn validate(&self) -> Result<()> {
        let dimensions = [
            self.embed_dim,
            self.dist_embed_num_heads,
            self.dist_embed_num_inducing_points,
            self.feature_group_size,
            self.feat_agg_num_blocks,
            self.feat_agg_num_heads,
            self.feat_agg_num_cls_tokens,
            self.icl_num_heads,
            self.decoder_head_dim,
            self.decoder_num_heads,
            self.ff_factor,
            self.softmax_scaling_mlp_hidden_dim,
        ];
        let icl_dim = self.embed_dim.checked_mul(self.feat_agg_num_cls_tokens);
        if dimensions.contains(&0) || self.max_num_classes < 2 || icl_dim.is_none() {
            candle::bail!("invalid TabPFN v3 dimensions: {self:?}")
        }
        let icl_dim = icl_dim.unwrap();
        let kv_heads = self.icl_num_kv_heads.unwrap_or(self.icl_num_heads);
        let test_heads = self.icl_num_kv_heads_test.unwrap_or(kv_heads);
        if !self.embed_dim.is_multiple_of(self.dist_embed_num_heads)
            || !self.embed_dim.is_multiple_of(self.feat_agg_num_heads)
            || !icl_dim.is_multiple_of(self.icl_num_heads)
            || kv_heads == 0
            || test_heads == 0
            || !self.icl_num_heads.is_multiple_of(kv_heads)
            || !self.icl_num_heads.is_multiple_of(test_heads)
            || test_heads > kv_heads
            || (self.use_rope && !(self.embed_dim / self.feat_agg_num_heads).is_multiple_of(2))
            || !self.feat_agg_rope_base.is_finite()
            || self.feat_agg_rope_base <= 0.
            || self.feature_group_size >= usize::BITS as usize
        {
            candle::bail!("invalid TabPFN v3 attention dimensions: {self:?}")
        }
        Ok(())
    }
}

// PyTorch RMSNorm uses the input dtype's machine epsilon by default.
#[derive(Debug, Clone)]
pub(super) struct Norm {
    weight: Tensor,
    affine: bool,
}

impl Norm {
    pub(super) fn new(size: usize, affine: bool, vb: VarBuilder) -> Result<Self> {
        let weight = if affine {
            vb.get(size, "weight")?
        } else {
            Tensor::ones(size, vb.dtype(), vb.device())?
        };
        Ok(Self { weight, affine })
    }
}

impl Module for Norm {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        if x.device().is_cuda()
            && x.dtype() == DType::F32
            && !x.track_op()
            && !self.weight.track_op()
        {
            return candle_nn::ops::rms_norm(&x.contiguous()?, &self.weight, f32::EPSILON);
        }
        let x =
            x.broadcast_div(&(x.sqr()?.mean_keepdim(D::Minus1)? + f32::EPSILON as f64)?.sqrt()?)?;
        if self.affine {
            x.broadcast_mul(&self.weight)
        } else {
            Ok(x)
        }
    }
}

#[derive(Debug, Clone)]
pub(super) struct Mlp {
    first: Linear,
    second: Linear,
}

impl Mlp {
    pub(super) fn new(size: usize, hidden: usize, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            first: linear_no_bias(size, hidden, vb.pp("0"))?,
            second: linear_no_bias(hidden, size, vb.pp("2"))?,
        })
    }
}

impl Module for Mlp {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        self.second.forward(&self.first.forward(x)?.gelu_erf()?)
    }
}

#[derive(Debug, Clone)]
struct SoftmaxScaling {
    base_first: Linear,
    base_second: Linear,
    query_first: Linear,
    query_second: Linear,
    heads: usize,
    head_dim: usize,
}

impl SoftmaxScaling {
    fn new(heads: usize, head_dim: usize, hidden: usize, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            base_first: linear(1, hidden, vb.pp("base_mlp.0"))?,
            base_second: linear(hidden, heads * head_dim, vb.pp("base_mlp.2"))?,
            query_first: linear(head_dim, hidden, vb.pp("query_mlp.0"))?,
            query_second: linear(hidden, head_dim, vb.pp("query_mlp.2"))?,
            heads,
            head_dim,
        })
    }

    fn forward(&self, q: &Tensor, count: usize) -> Result<Tensor> {
        // Keep this input-dependent scalar on the device instead of uploading
        // a tiny host tensor at every attention layer.
        let log_count = Tensor::full((count.max(1) as f32).ln(), (1, 1), q.device())?;
        let base = self
            .base_second
            .forward(&self.base_first.forward_gelu_erf(&log_count)?)?
            .reshape((1, 1, self.heads, self.head_dim))?;
        let modulation = (self
            .query_second
            .forward(&self.query_first.forward_gelu_erf(&q.contiguous()?)?)?
            .tanh()?
            + 1.)?;
        q.mul(&modulation)?.broadcast_mul(&base)
    }
}

fn attend(q: &Tensor, k: &Tensor, v: &Tensor) -> Result<Tensor> {
    let (b, seq, heads, d) = q.dims4()?;
    let (_, count, kv_heads, _) = k.dims4()?;
    let q_heads = q.transpose(1, 2)?;
    let k_heads = k.transpose(1, 2)?;
    let v_heads = v.transpose(1, 2)?;
    if candle_nn::attention::cuda_f32::is_supported(&q_heads, &k_heads, &v_heads) {
        return candle_nn::attention::cuda_f32::sdpa(
            &q_heads,
            &k_heads,
            &v_heads,
            1. / (d as f32).sqrt(),
        )?
        .transpose(1, 2);
    }
    let expand_kv = |x: &Tensor| -> Result<Tensor> {
        x.transpose(1, 2)?
            .unsqueeze(2)?
            .broadcast_as((b, kv_heads, heads / kv_heads, count, d))?
            .reshape((b, heads, count, d))?
            .contiguous()
    };
    let q = q.transpose(1, 2)?.contiguous()?;
    let scores = (q.matmul(&expand_kv(k)?.t()?)? / (d as f64).sqrt())?;
    candle_nn::ops::softmax_last_dim(&scores)?
        .matmul(&expand_kv(v)?)?
        .transpose(1, 2)?
        .reshape((b, seq, heads, d))
}

fn rotate(x: &Tensor, freqs: &Tensor) -> Result<Tensor> {
    let (_, seq, _, d) = x.dims4()?;
    let positions = Tensor::arange(0u32, seq as u32, x.device())?.to_dtype(DType::F32)?;
    let angles = positions
        .unsqueeze(1)?
        .broadcast_mul(&freqs.unsqueeze(0)?)?;
    if x.device().is_cuda() && !x.track_op() && !freqs.track_op() {
        return candle_nn::rotary_emb::rope_thd(&x.contiguous()?, &angles.cos()?, &angles.sin()?);
    }
    let cos = Tensor::cat(&[angles.cos()?, angles.cos()?], 1)?.reshape((1, seq, 1, d))?;
    let sin = Tensor::cat(&[angles.sin()?, angles.sin()?], 1)?.reshape((1, seq, 1, d))?;
    let rotated = Tensor::cat(
        &[x.narrow(3, d / 2, d / 2)?.neg()?, x.narrow(3, 0, d / 2)?],
        3,
    )?;
    x.broadcast_mul(&cos)? + rotated.broadcast_mul(&sin)?
}

#[derive(Debug, Clone)]
struct Attention {
    projections: super::attention_projection::AttentionProjections,
    out: Linear,
    q_norm: Option<Norm>,
    k_norm: Option<Norm>,
    scaling: Option<SoftmaxScaling>,
}

impl Attention {
    fn new(
        size: usize,
        heads: usize,
        kv_heads: usize,
        config: &Config,
        scaled: bool,
        qk_norm: bool,
        vb: VarBuilder,
    ) -> Result<Self> {
        let head_dim = size / heads;
        Ok(Self {
            projections: super::attention_projection::AttentionProjections::new(
                linear_no_bias(size, size, vb.pp("q_projection"))?,
                linear_no_bias(size, kv_heads * head_dim, vb.pp("k_projection"))?,
                linear_no_bias(size, kv_heads * head_dim, vb.pp("v_projection"))?,
                heads,
                kv_heads,
                head_dim,
            )?,
            out: linear_no_bias(size, size, vb.pp("out_projection"))?,
            q_norm: qk_norm
                .then(|| {
                    Norm::new(
                        head_dim,
                        config.layernorm_elementwise_affine,
                        vb.pp("q_norm"),
                    )
                })
                .transpose()?,
            k_norm: qk_norm
                .then(|| {
                    Norm::new(
                        head_dim,
                        config.layernorm_elementwise_affine,
                        vb.pp("k_norm"),
                    )
                })
                .transpose()?,
            scaling: scaled
                .then(|| {
                    SoftmaxScaling::new(
                        heads,
                        head_dim,
                        config.softmax_scaling_mlp_hidden_dim,
                        vb.pp("softmax_scaling_layer"),
                    )
                })
                .transpose()?,
        })
    }

    fn forward(
        &self,
        x: &Tensor,
        context: &Tensor,
        rope: Option<&Tensor>,
        train_len: Option<usize>,
        test_heads: Option<usize>,
    ) -> Result<Tensor> {
        let (b, seq, size) = x.dims3()?;
        let count = context.dim(1)?;
        let (mut q, mut k, v) = self
            .projections
            .forward(x, context, x.id() == context.id())?;
        if let Some(freqs) = rope {
            q = rotate(&q, freqs)?;
            k = rotate(&k, freqs)?;
        }
        if let Some(norm) = &self.q_norm {
            q = norm.forward(&q)?;
        }
        if let Some(norm) = &self.k_norm {
            k = norm.forward(&k)?;
        }
        // Scaling is row-independent and both subsets share the context size.
        // Evaluate its learned base and query projections once per attention.
        if let Some(scaling) = &self.scaling {
            q = scaling.forward(&q, count)?;
        }
        let out = if let (Some(n), Some(h)) = (train_len, test_heads) {
            let train = attend(&q.narrow(1, 0, n)?, &k, &v)?;
            let test = attend(
                &q.narrow(1, n, seq - n)?,
                &k.narrow(2, 0, h)?,
                &v.narrow(2, 0, h)?,
            )?;
            Tensor::cat(&[train, test], 1)?
        } else {
            attend(&q, &k, &v)?
        };
        self.out.forward(&out.reshape((b, seq, size))?)
    }
}

#[derive(Debug, Clone)]
struct CrossBlock {
    attention: Attention,
    mlp: Mlp,
    norm_q: Norm,
    norm_kv: Norm,
    norm_mlp: Norm,
}

impl CrossBlock {
    fn new(config: &Config, scaled: bool, qk_norm: bool, vb: VarBuilder) -> Result<Self> {
        let size = config.embed_dim;
        Ok(Self {
            attention: Attention::new(
                size,
                config.dist_embed_num_heads,
                config.dist_embed_num_heads,
                config,
                scaled,
                qk_norm,
                vb.pp("attn"),
            )?,
            mlp: Mlp::new(size, size * config.ff_factor, vb.pp("mlp"))?,
            norm_q: Norm::new(
                size,
                config.layernorm_elementwise_affine,
                vb.pp("layernorm_q"),
            )?,
            norm_kv: Norm::new(
                size,
                config.layernorm_elementwise_affine,
                vb.pp("layernorm_kv"),
            )?,
            norm_mlp: Norm::new(
                size,
                config.layernorm_elementwise_affine,
                vb.pp("layernorm2"),
            )?,
        })
    }
    fn forward(&self, x: &Tensor, context: &Tensor) -> Result<Tensor> {
        let x = (x + self.attention.forward(
            &self.norm_q.forward(x)?,
            &self.norm_kv.forward(context)?,
            None,
            None,
            None,
        )?)?;
        &x + self.mlp.forward(&self.norm_mlp.forward(&x)?)?
    }
}

#[derive(Debug, Clone)]
struct DistributionLayer {
    inducing: Tensor,
    first: CrossBlock,
    second: CrossBlock,
}

#[derive(Debug, Clone)]
pub(super) struct DistributionEmbedder(Vec<DistributionLayer>);

impl DistributionEmbedder {
    pub(super) fn new(config: &Config, qk_norm: bool, vb: VarBuilder) -> Result<Self> {
        let layers = (0..config.dist_embed_num_blocks)
            .map(|i| {
                let vb = vb.pp("layers").pp(i);
                Ok(DistributionLayer {
                    inducing: vb.get(
                        (config.dist_embed_num_inducing_points, config.embed_dim),
                        "inducing_vectors",
                    )?,
                    first: CrossBlock::new(config, true, qk_norm, vb.pp("cross_attn_block1"))?,
                    second: CrossBlock::new(config, false, qk_norm, vb.pp("cross_attn_block2"))?,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Self(layers))
    }
    pub(super) fn forward(&self, x: &Tensor, train_len: usize) -> Result<Tensor> {
        let (b, rows, cols, size) = x.dims4()?;
        let mut x = x.transpose(1, 2)?.reshape((b * cols, rows, size))?;
        for layer in &self.0 {
            let inducing = layer.inducing.unsqueeze(0)?.broadcast_as((
                b * cols,
                layer.inducing.dim(0)?,
                size,
            ))?;
            let hidden = layer
                .first
                .forward(&inducing, &x.narrow(1, 0, train_len)?)?;
            x = layer.second.forward(&x, &hidden)?;
        }
        x.reshape((b, cols, rows, size))?.transpose(1, 2)
    }
}

#[derive(Debug, Clone)]
struct FeatureBlock {
    attention: Attention,
    norm: Norm,
    norm_mlp: Norm,
    mlp: Mlp,
}

impl FeatureBlock {
    fn new(config: &Config, qk_norm: bool, vb: VarBuilder) -> Result<Self> {
        let size = config.embed_dim;
        Ok(Self {
            attention: Attention::new(
                size,
                config.feat_agg_num_heads,
                config.feat_agg_num_heads,
                config,
                false,
                qk_norm,
                vb.pp("attention"),
            )?,
            norm: Norm::new(
                size,
                config.layernorm_elementwise_affine,
                vb.pp("layernorm"),
            )?,
            norm_mlp: Norm::new(
                size,
                config.layernorm_elementwise_affine,
                vb.pp("layernorm_mlp"),
            )?,
            mlp: Mlp::new(size, size * config.ff_factor, vb.pp("mlp"))?,
        })
    }
    fn forward(
        &self,
        x: &Tensor,
        cls_count: Option<usize>,
        rope: Option<&Tensor>,
    ) -> Result<Tensor> {
        let query = match cls_count {
            Some(n) => x.narrow(1, 0, n)?,
            None => x.clone(),
        };
        // RMSNorm is independent for each row, so the query subset can reuse
        // the context's normalization without recomputing it.
        let normalized = self.norm.forward(x)?;
        let normalized_query = match cls_count {
            Some(n) => normalized.narrow(1, 0, n)?.contiguous()?,
            None => normalized.clone(),
        };
        let out = self
            .attention
            .forward(&normalized_query, &normalized, rope, None, None)?;
        let x = (query + out)?;
        &x + self.mlp.forward(&self.norm_mlp.forward(&x)?)?
    }
}

#[derive(Debug, Clone)]
pub(super) struct ColumnAggregator {
    cls: Tensor,
    blocks: Vec<FeatureBlock>,
    norm: Norm,
    rope: Option<Tensor>,
}

impl ColumnAggregator {
    pub(super) fn new(config: &Config, qk_norm: bool, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            cls: vb.get(
                (config.feat_agg_num_cls_tokens, config.embed_dim),
                "cls_tokens",
            )?,
            blocks: (0..config.feat_agg_num_blocks)
                .map(|i| FeatureBlock::new(config, qk_norm, vb.pp("blocks").pp(i)))
                .collect::<Result<_>>()?,
            norm: Norm::new(
                config.embed_dim,
                config.layernorm_elementwise_affine,
                vb.pp("out_ln"),
            )?,
            rope: config
                .use_rope
                .then(|| {
                    vb.pp("rope")
                        .get(config.embed_dim / config.feat_agg_num_heads / 2, "freqs")
                })
                .transpose()?,
        })
    }
    pub(super) fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (b, rows, cols, size) = x.dims4()?;
        let ncls = self.cls.dim(0)?;
        let cls = self
            .cls
            .unsqueeze(0)?
            .broadcast_as((b * rows, ncls, size))?;
        let mut x = Tensor::cat(&[cls, x.reshape((b * rows, cols, size))?], 1)?;
        for (i, block) in self.blocks.iter().enumerate() {
            x = block.forward(
                &x,
                (i + 1 == self.blocks.len()).then_some(ncls),
                self.rope.as_ref(),
            )?;
        }
        self.norm.forward(&x)?.reshape((b, rows, ncls, size))
    }
}

#[derive(Debug, Clone)]
pub(super) struct IclBlock {
    attention: Attention,
    norm: Norm,
    norm_mlp: Norm,
    mlp: Mlp,
    test_heads: Option<usize>,
}

impl IclBlock {
    pub(super) fn new(config: &Config, qk_norm: bool, vb: VarBuilder) -> Result<Self> {
        let size = config.embed_dim * config.feat_agg_num_cls_tokens;
        Ok(Self {
            attention: Attention::new(
                size,
                config.icl_num_heads,
                config.icl_num_kv_heads.unwrap_or(config.icl_num_heads),
                config,
                true,
                qk_norm,
                vb.pp("icl_attention"),
            )?,
            norm: Norm::new(
                size,
                config.layernorm_elementwise_affine,
                vb.pp("layernorm"),
            )?,
            norm_mlp: Norm::new(
                size,
                config.layernorm_elementwise_affine,
                vb.pp("layernorm_mlp"),
            )?,
            mlp: Mlp::new(size, size * config.ff_factor, vb.pp("mlp"))?,
            test_heads: config.icl_num_kv_heads_test,
        })
    }
    pub(super) fn forward(&self, x: &Tensor, train_len: usize) -> Result<Tensor> {
        let x = {
            let norm = self.norm.forward(x)?;
            (x + self.attention.forward(
                &norm,
                &norm.narrow(1, 0, train_len)?,
                None,
                Some(train_len),
                self.test_heads,
            )?)?
        };

        &x + self.mlp.forward(&self.norm_mlp.forward(&x)?)?
    }
}

#[derive(Debug, Clone)]
pub(super) struct ManyClassDecoder {
    q: Linear,
    k: Linear,
    scaling: Option<SoftmaxScaling>,
    heads: usize,
    head_dim: usize,
    classes: usize,
}

impl ManyClassDecoder {
    pub(super) fn new(config: &Config, vb: VarBuilder) -> Result<Self> {
        let size = config.embed_dim * config.feat_agg_num_cls_tokens;
        let heads = config.decoder_num_heads;
        let head_dim = config.decoder_head_dim;
        Ok(Self {
            q: linear(size, heads * head_dim, vb.pp("q_projection"))?,
            k: linear(size, heads * head_dim, vb.pp("k_projection"))?,
            scaling: config
                .decoder_use_softmax_scaling
                .then(|| {
                    SoftmaxScaling::new(
                        heads,
                        head_dim,
                        config.softmax_scaling_mlp_hidden_dim,
                        vb.pp("softmax_scaling_layer"),
                    )
                })
                .transpose()?,
            heads,
            head_dim,
            classes: config.max_num_classes,
        })
    }
    pub(super) fn forward(&self, train: &Tensor, test: &Tensor, labels: &Tensor) -> Result<Tensor> {
        let (b, n, _) = train.dims3()?;
        let m = test.dim(1)?;
        let mut q = self
            .q
            .forward(test)?
            .reshape((b, m, self.heads, self.head_dim))?;
        let k = self
            .k
            .forward(train)?
            .reshape((b, n, self.heads, self.head_dim))?;
        if let Some(scaling) = &self.scaling {
            q = scaling.forward(&q, n)?;
        }
        // Like upstream, attend only over present class columns and pad afterward.
        // Besides saving memory, this leaves absent classes at zero when an
        // overflow in the input statistics makes an attention row non-finite.
        let classes = labels.max_all()?.to_scalar::<u32>()? as usize + 1;
        let one_hot = Tensor::arange(0u32, classes as u32, labels.device())?
            .reshape((1, 1, classes))?
            .broadcast_eq(&labels.unsqueeze(2)?)?
            .to_dtype(DType::F32)?;
        let mut probabilities = retrieval_probabilities(&q, &k, &one_hot)?;
        if classes < self.classes {
            probabilities = Tensor::cat(
                &[
                    probabilities,
                    Tensor::zeros((b, m, self.classes - classes), DType::F32, labels.device())?,
                ],
                2,
            )?;
        }
        (probabilities.clamp(1e-5f64, f64::INFINITY)? + 3e-5)?.log()
    }
}

// One-hot class values need not have the same width as Q/K. Small class sets
// fit one native attention call after padding, without a query-by-training-row
// score matrix. Keep the materialized path for wider class sets and autograd.
fn retrieval_probabilities(q: &Tensor, k: &Tensor, one_hot: &Tensor) -> Result<Tensor> {
    let (batch, _, heads, head_dim) = q.dims4()?;
    let (_, keys, classes) = one_hot.dims3()?;
    let q = q.transpose(1, 2)?;
    let k = k.transpose(1, 2)?;
    if classes <= head_dim
        && one_hot.dtype() == DType::F32
        && !one_hot.track_op()
        && one_hot.device().same_device(q.device())
        && candle_nn::attention::cuda_f32::is_supported(&q, &k, &k)
    {
        let values = one_hot
            .pad_with_zeros(2, 0, head_dim - classes)?
            .unsqueeze(1)?
            .broadcast_as((batch, heads, keys, head_dim))?;
        return candle_nn::attention::cuda_f32::sdpa(
            &q,
            &k,
            &values,
            1. / (head_dim as f32).sqrt(),
        )?
        .narrow(3, 0, classes)?
        .mean(1);
    }
    let scores = (q.contiguous()?.matmul(&k.contiguous()?.t()?)? / (head_dim as f64).sqrt())?;
    candle_nn::ops::softmax_last_dim(&scores)?
        .mean(1)?
        .matmul(one_hot)
}

/// TabPFN v3 model, including its trained classification retrieval decoder.
#[derive(Debug, Clone)]
pub struct Model {
    config: Config,
    x_embed: Linear,
    col_y: Embedding,
    icl_y: Embedding,
    distribution: DistributionEmbedder,
    aggregator: ColumnAggregator,
    blocks: Vec<IclBlock>,
    norm: Norm,
    decoder: ManyClassDecoder,
}

impl Model {
    pub fn new(config: &Config, vb: VarBuilder) -> Result<Self> {
        config.validate()?;
        if vb.dtype() != DType::F32 {
            candle::bail!("TabPFN v3 currently requires F32 weights")
        }
        let icl_size = config.embed_dim * config.feat_agg_num_cls_tokens;
        Ok(Self {
            config: config.clone(),
            x_embed: linear(
                config.feature_group_size * if config.use_nan_indicators { 2 } else { 1 },
                config.embed_dim,
                vb.pp("x_embed"),
            )?,
            col_y: embedding(
                config.max_num_classes,
                config.embed_dim,
                vb.pp("col_y_encoder.embedding"),
            )?,
            icl_y: embedding(
                config.max_num_classes,
                icl_size,
                vb.pp("icl_y_encoder.embedding"),
            )?,
            distribution: DistributionEmbedder::new(
                config,
                false,
                vb.pp("feature_distribution_embedder"),
            )?,
            aggregator: ColumnAggregator::new(config, false, vb.pp("column_aggregator"))?,
            blocks: (0..config.nlayers)
                .map(|i| IclBlock::new(config, false, vb.pp("icl_blocks").pp(i)))
                .collect::<Result<_>>()?,
            norm: Norm::new(
                icl_size,
                config.layernorm_elementwise_affine,
                vb.pp("output_norm"),
            )?,
            decoder: ManyClassDecoder::new(config, vb.pp("many_class_decoder"))?,
        })
    }

    /// Batched raw-feature inference. `x` is `[batch, train + query, features]`
    /// and `y` is `[batch, train]` of U32 class IDs. Returns classification
    /// logits `[batch, query, max_num_classes]`, including internal preprocessing.
    pub fn forward(&self, x: &Tensor, y: &Tensor) -> Result<Tensor> {
        let grouped = self.prepare_inputs(x, y)?;
        let device = self.x_embed.weight().device();
        let (grouped, labels) = (grouped.to_device(device)?, y.to_device(device)?);

        self.forward_prepared(&grouped, &labels)
    }

    /// Nonlearned feature preparation. Returns grouped F32 cells on the CPU.
    fn prepare_inputs(&self, x: &Tensor, y: &Tensor) -> Result<Tensor> {
        let (b, rows, cols) = x.dims3()?;
        let (yb, n) = y.dims2()?;
        if x.dtype() != DType::F32
            || y.dtype() != DType::U32
            || b == 0
            || b != yb
            || n == 0
            || n >= rows
            || cols == 0
        {
            candle::bail!("TabPFN v3 expects F32 features [batch, train+query, features] and U32 labels [batch, train]")
        }
        if y.flatten_all()?
            .to_vec1::<u32>()?
            .iter()
            .any(|&v| v as usize >= self.config.max_num_classes)
        {
            candle::bail!("TabPFN v3 label exceeds max_num_classes")
        }
        preprocess_and_group(
            x,
            n,
            self.config.feature_group_size,
            self.config.use_nan_indicators,
        )
    }

    /// Run every learned stage on grouped features and labels already on the model device.
    fn forward_prepared(&self, grouped: &Tensor, y: &Tensor) -> Result<Tensor> {
        let (b, rows, cols, width) = grouped.dims4()?;
        let (yb, n) = y.dims2()?;
        if b == 0
            || b != yb
            || n == 0
            || n >= rows
            || cols == 0
            || width
                != self.config.feature_group_size
                    * if self.config.use_nan_indicators { 2 } else { 1 }
            || grouped.dtype() != DType::F32
            || y.dtype() != DType::U32
            || !grouped.device().same_device(self.x_embed.weight().device())
            || !y.device().same_device(self.x_embed.weight().device())
        {
            candle::bail!("invalid prepared TabPFN v3 tensors or device")
        }
        let mut x = {
            let embedded = self.x_embed.forward(grouped)?;
            let col_labels = self.col_y.forward(y)?.unsqueeze(2)?;
            let train = embedded.narrow(1, 0, n)?.broadcast_add(&col_labels)?;
            Tensor::cat(&[train, embedded.narrow(1, n, rows - n)?], 1)?
        };
        x = self.distribution.forward(&x, n)?;
        x = self.aggregator.forward(&x)?.reshape((
            b,
            rows,
            self.config.embed_dim * self.config.feat_agg_num_cls_tokens,
        ))?;
        x = add_training_labels(&x, &self.icl_y.forward(y)?)?;
        for block in &self.blocks {
            x = block.forward(&x, n)?;
        }

        x = self.norm.forward(&x)?;
        finite_logits(
            &self
                .decoder
                .forward(&x.narrow(1, 0, n)?, &x.narrow(1, n, rows - n)?, y)?,
        )
    }

    /// Predict one deterministic member, with the default temperature 0.9 and contiguous class IDs.
    pub fn predict_proba(
        &self,
        train_x: &Tensor,
        train_y: &Tensor,
        test_x: &Tensor,
    ) -> Result<Tensor> {
        self.predict_proba_with_temperature(train_x, train_y, test_x, 0.9)
    }

    /// Predict with an explicit positive softmax temperature.
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
        let classes = validate_table(train_x, train_y, test_x, self.config.max_num_classes)?;
        let xs = Tensor::cat(&[train_x, test_x], 0)?.unsqueeze(0)?;
        let ys = train_y.unsqueeze(0)?;
        let logits = self.forward(&xs, &ys)?;

        candle_nn::ops::softmax_last_dim(
            &(logits.narrow(2, 0, classes)?.squeeze(0)? / temperature)?,
        )
    }
}

pub(super) fn add_training_labels(x: &Tensor, labels: &Tensor) -> Result<Tensor> {
    let n = labels.dim(1)?;
    Tensor::cat(
        &[
            x.narrow(1, 0, n)?.broadcast_add(labels)?,
            x.narrow(1, n, x.dim(1)? - n)?,
        ],
        1,
    )
}

pub(super) fn validate_table(
    train_x: &Tensor,
    train_y: &Tensor,
    test_x: &Tensor,
    max_classes: usize,
) -> Result<usize> {
    let (n, c) = train_x.dims2()?;
    let (m, tc) = test_x.dims2()?;
    if n == 0
        || m == 0
        || c == 0
        || c != tc
        || train_y.dims() != [n]
        || train_x.dtype() != DType::F32
        || test_x.dtype() != DType::F32
        || train_y.dtype() != DType::U32
    {
        candle::bail!("TabPFN expects nonempty F32 train/query tables with matching features and U32 training labels")
    }
    let labels = train_y.to_vec1::<u32>()?;
    let highest_label = labels.iter().copied().max().unwrap_or(0) as usize;
    if highest_label == 0 || highest_label >= max_classes {
        candle::bail!("TabPFN requires between 2 and {max_classes} classes")
    }
    let classes = highest_label + 1;
    let mut present = vec![false; classes];
    for label in labels {
        present[label as usize] = true;
    }
    if present.contains(&false) {
        candle::bail!("TabPFN class IDs must be contiguous, starting at zero")
    }
    Ok(classes)
}

// Upstream sanitizes both the imputation mean and the final model output with
// torch.nan_to_num. Even finite inputs can overflow a float32 reduction.
pub(super) fn finite_imputation_mean(values: &[f32]) -> f32 {
    let mean = values.iter().sum::<f32>() / values.len().max(1) as f32;
    if mean.is_nan() {
        0.
    } else {
        mean.clamp(f32::MIN, f32::MAX)
    }
}

pub(super) fn finite_logits(logits: &Tensor) -> Result<Tensor> {
    logits
        .eq(logits)?
        .where_cond(logits, &logits.zeros_like()?)?
        .clamp(f32::MIN as f64, f32::MAX as f64)
}

// The same feature neighbourhood is used for every row and every input channel.
// Compute its modulo operations once per table shape, rather than per cell.
pub(super) fn feature_indices(columns: usize, group: usize) -> Vec<usize> {
    (0..columns)
        .flat_map(|column| (0..group).map(move |i| (column + (1usize << i) % columns) % columns))
        .collect()
}

// Preprocessing is intentionally performed in F32 to match the checkpoint's
// scaler, including unbiased variance and NaN/+Inf/-Inf indicators.
fn preprocess_and_group(x: &Tensor, n: usize, group: usize, indicators: bool) -> Result<Tensor> {
    let (b, rows, cols) = x.dims3()?;
    let input = x.to_vec3::<f32>()?;
    let width = group * if indicators { 2 } else { 1 };
    let feature_indices = feature_indices(cols, group);
    let mut output = vec![0f32; b * rows * cols * width];
    for (batch, table) in input.iter().enumerate() {
        let mut scaled = vec![vec![0f32; cols]; rows];
        for col in 0..cols {
            let finite: Vec<f32> = table[..n]
                .iter()
                .map(|r| r[col])
                .filter(|v| v.is_finite())
                .collect();
            let impute = finite_imputation_mean(&finite);
            let train: Vec<f32> = table[..n]
                .iter()
                .map(|r| if r[col].is_finite() { r[col] } else { impute })
                .collect();
            let mean = train.iter().sum::<f32>() / n as f32;
            let std = (train.iter().map(|v| (v - mean).powi(2)).sum::<f32>()
                / (n - 1).max(1) as f32)
                .sqrt();
            let std = if std == 0. || n == 1 { 1. } else { std };
            for row in 0..rows {
                let value = if table[row][col].is_finite() {
                    table[row][col]
                } else {
                    impute
                };
                scaled[row][col] = ((value - mean) / (std + f32::EPSILON)).clamp(-100., 100.);
            }
        }
        for row in 0..rows {
            for col in 0..cols {
                let start = ((batch * rows + row) * cols + col) * width;
                for (pos, &src) in feature_indices[col * group..(col + 1) * group]
                    .iter()
                    .enumerate()
                {
                    output[start + pos] = scaled[row][src];
                    if indicators {
                        let value = table[row][src];
                        output[start + group + pos] = if value.is_nan() {
                            -2.
                        } else if value == f32::INFINITY {
                            2.
                        } else if value == f32::NEG_INFINITY {
                            4.
                        } else {
                            0.
                        };
                    }
                }
            }
        }
    }
    Tensor::from_vec(output, (b, rows, cols, width), &candle::Device::Cpu)
}

#[cfg(test)]
mod input_tests {
    use super::*;
    use candle::Device;

    #[test]
    fn prepared_input_validation() -> Result<()> {
        let config = Config {
            max_num_classes: 4,
            embed_dim: 16,
            dist_embed_num_blocks: 1,
            dist_embed_num_heads: 2,
            dist_embed_num_inducing_points: 3,
            feat_agg_num_blocks: 1,
            feat_agg_num_heads: 2,
            feat_agg_num_cls_tokens: 2,
            nlayers: 1,
            icl_num_heads: 4,
            icl_num_kv_heads: Some(2),
            icl_num_kv_heads_test: Some(1),
            decoder_head_dim: 8,
            decoder_num_heads: 2,
            softmax_scaling_mlp_hidden_dim: 5,
            ..Config::default()
        };
        let device = &Device::Cpu;
        let model = Model::new(&config, VarBuilder::zeros(DType::F32, device))?;
        let x = Tensor::new(&[[[0f32, 1.], [1., 2.], [2., 3.]]], device)?;
        let y = Tensor::new(&[[0u32, 1]], device)?;
        let grouped = model.prepare_inputs(&x, &y)?;
        assert!(grouped.device().is_cpu());
        assert!(model
            .forward_prepared(&grouped, &y.to_dtype(DType::F32)?)
            .is_err());
        assert!(model
            .forward_prepared(&grouped.narrow(3, 0, 1)?, &y)
            .is_err());
        Ok(())
    }
}

#[cfg(test)]
mod retrieval_tests {
    use super::*;
    use candle::{test_device, Device, Var};

    fn data(size: usize, offset: usize) -> Vec<f32> {
        (0..size)
            .map(|i| (((i + offset) * 19 % 47) as f32 - 23.) / 37.)
            .collect()
    }

    fn one_hot(labels: &Tensor, classes: usize) -> Result<Tensor> {
        Tensor::arange(0u32, classes as u32, labels.device())?
            .reshape((1, 1, classes))?
            .broadcast_eq(&labels.unsqueeze(2)?)?
            .to_dtype(DType::F32)
    }

    fn reference(q: &Tensor, k: &Tensor, values: &Tensor) -> Result<Tensor> {
        let scores = (q
            .transpose(1, 2)?
            .contiguous()?
            .matmul(&k.transpose(1, 2)?.contiguous()?.t()?)?
            / (q.dim(3)? as f64).sqrt())?;
        candle_nn::ops::softmax_last_dim(&scores)?
            .mean(1)?
            .matmul(values)
    }

    fn close(actual: &Tensor, expected: &Tensor) -> Result<()> {
        assert_eq!(actual.dims(), expected.dims());
        let actual = actual.flatten_all()?.to_vec1::<f32>()?;
        let expected = expected.flatten_all()?.to_vec1::<f32>()?;
        for (actual, expected) in actual.into_iter().zip(expected) {
            if expected.is_nan() {
                assert!(actual.is_nan());
            } else {
                assert!(
                    (actual - expected).abs() <= 3e-5 * (1. + expected.abs()),
                    "{actual} != {expected}"
                );
            }
        }
        Ok(())
    }

    fn class_attention(device: &Device) -> Result<()> {
        // D=5 exercises unsupported-head fallback. D+1 exercises the wider
        // class fallback; other cases use the CUDA kernel when available.
        for width in [5, 8, 16, 24, 32, 64] {
            for classes in [1, 3, width, width + 1] {
                let (batch, queries, heads, keys) = (2, 5, 3, 2 * classes + 3);
                let q = Tensor::from_vec(
                    data(batch * (queries + 2) * heads * width, 1),
                    (batch, queries + 2, heads, width),
                    device,
                )?
                .narrow(1, 1, queries)?;
                let k = Tensor::from_vec(
                    data(batch * (keys + 2) * heads * width, 7),
                    (batch, keys + 2, heads, width),
                    device,
                )?
                .narrow(1, 1, keys)?;
                let labels = Tensor::from_vec(
                    (0..batch * keys)
                        .map(|i| (i % classes) as u32)
                        .collect::<Vec<_>>(),
                    (batch, keys),
                    device,
                )?;
                let values = one_hot(&labels, classes)?;
                let actual = retrieval_probabilities(&q, &k, &values)?;
                let expected = reference(&q, &k, &values)?;
                close(&actual, &expected)?;
                close(
                    &actual.sum(2)?,
                    &Tensor::ones((batch, queries), DType::F32, device)?,
                )?;
                if classes > 1 {
                    let output = actual.flatten_all()?.to_vec1::<f32>()?;
                    assert!(output.iter().all(|p| *p > 0. && *p < 1.));
                    assert!(output
                        .windows(2)
                        .any(|pair| (pair[0] - pair[1]).abs() > 1e-4));
                }
            }
        }
        Ok(())
    }

    fn tracked_class_attention(device: &Device) -> Result<()> {
        let q = Var::from_tensor(&Tensor::from_vec(data(2 * 3 * 8, 2), (1, 2, 3, 8), device)?)?;
        let k = Var::from_tensor(&Tensor::from_vec(data(7 * 3 * 8, 9), (1, 7, 3, 8), device)?)?;
        let values = one_hot(&Tensor::new(&[[0u32, 1, 0, 2, 1, 0, 0]], device)?, 3)?;
        // Derived variables must retain the materialized differentiable path.
        let q_derived = (q.as_tensor() * 1.3)?;
        let k_derived = (k.as_tensor() * 0.7)?;
        for (queries, keys) in [
            (q_derived.clone(), k_derived.clone()),
            (q_derived, k.as_tensor().detach()),
            (q.as_tensor().detach(), k_derived),
        ] {
            let actual = retrieval_probabilities(&queries, &keys, &values)?;
            let expected = reference(&queries, &keys, &values)?;
            close(&actual, &expected)?;
            let actual_grad = actual.sqr()?.sum_all()?.backward()?;
            let expected_grad = expected.sqr()?.sum_all()?.backward()?;
            for variable in [&q, &k] {
                if let Some(expected) = expected_grad.get(variable) {
                    close(
                        actual_grad.get(variable).expect("missing gradient"),
                        expected,
                    )?;
                }
            }
        }
        Ok(())
    }

    fn decoder_padding_and_nonfinite(device: &Device) -> Result<()> {
        let decoder = ManyClassDecoder {
            q: Linear::new(Tensor::eye(8, DType::F32, device)?, None),
            k: Linear::new(Tensor::eye(8, DType::F32, device)?, None),
            scaling: None,
            heads: 1,
            head_dim: 8,
            classes: 11,
        };
        let train = Tensor::from_vec(data(7 * 8, 4), (1, 7, 8), device)?;
        let labels = Tensor::new(&[[0u32, 1, 0, 2, 1, 0, 0]], device)?;
        for exceptional in [None, Some(f32::NAN), Some(f32::INFINITY)] {
            let mut input = data(3 * 8, 11);
            if let Some(value) = exceptional {
                input[..8].fill(value);
            }
            let query = Tensor::from_vec(input, (1, 3, 8), device)?;
            let output = decoder.forward(&train, &query, &labels)?;
            let probabilities = reference(
                &query.reshape((1, 3, 1, 8))?,
                &train.reshape((1, 7, 1, 8))?,
                &one_hot(&labels, 3)?,
            )?;
            let expected = (probabilities
                .pad_with_zeros(2, 0, 8)?
                .clamp(1e-5f64, f64::INFINITY)?
                + 3e-5)?
                .log()?;
            close(&output, &expected)?;
            close(&finite_logits(&output)?, &finite_logits(&expected)?)?;
            close(
                &output.narrow(2, 3, 8)?,
                &Tensor::full((4e-5f32).ln(), (1, 3, 8), device)?,
            )?;
        }
        Ok(())
    }

    test_device!(
        class_attention,
        class_attention_cpu,
        class_attention_cuda,
        class_attention_metal
    );
    test_device!(
        tracked_class_attention,
        tracked_class_attention_cpu,
        tracked_class_attention_cuda,
        tracked_class_attention_metal
    );
    test_device!(
        decoder_padding_and_nonfinite,
        decoder_padding_and_nonfinite_cpu,
        decoder_padding_and_nonfinite_cuda,
        decoder_padding_and_nonfinite_metal
    );
}
