//! TabPFN v3.5 and v3.5 Fast, for deterministic single-member classification.
//!
//! The two releases share an architecture and differ in their checkpoint configuration.
//! Cell values are embedded with learned Fourier features and empirical distribution
//! ranks before the distribution, column, and in-context transformers.

use super::v3::{
    add_training_labels, finite_imputation_mean, finite_logits, validate_table, ColumnAggregator,
    Config as BackboneConfig, DistributionEmbedder, IclBlock, ManyClassDecoder, Mlp, Norm,
};
use candle::{DType, Module, Result, Tensor, D};
use candle_nn::{embedding, layer_norm, linear_no_bias, Embedding, LayerNorm, Linear, VarBuilder};
use serde::Deserialize;

#[derive(Debug, Clone, Deserialize, serde::Serialize)]
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
    pub nlayers: usize,
    pub icl_num_heads: usize,
    pub icl_num_kv_heads: Option<usize>,
    pub icl_num_kv_heads_test: Option<usize>,
    pub decoder_head_dim: usize,
    pub decoder_num_heads: usize,
    pub decoder_use_softmax_scaling: bool,
    pub ff_factor: usize,
    pub softmax_scaling_mlp_hidden_dim: usize,
    pub fourier_encoding_num_frequencies: usize,
    pub cell_ecdf_num_frequencies: usize,
    pub cell_ecdf_num_buckets: usize,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            max_num_classes: 0,
            embed_dim: 128,
            dist_embed_num_blocks: 3,
            dist_embed_num_heads: 8,
            dist_embed_num_inducing_points: 128,
            feature_group_size: 3,
            feat_agg_num_blocks: 3,
            feat_agg_num_heads: 8,
            feat_agg_num_cls_tokens: 8,
            feat_agg_rope_base: 100_000.,
            nlayers: 24,
            icl_num_heads: 16,
            icl_num_kv_heads: None,
            icl_num_kv_heads_test: Some(1),
            decoder_head_dim: 64,
            decoder_num_heads: 6,
            decoder_use_softmax_scaling: true,
            ff_factor: 2,
            softmax_scaling_mlp_hidden_dim: 64,
            fourier_encoding_num_frequencies: 32,
            cell_ecdf_num_frequencies: 4,
            cell_ecdf_num_buckets: 8192,
        }
    }
}

impl Config {
    fn backbone(&self) -> BackboneConfig {
        BackboneConfig {
            max_num_classes: self.max_num_classes,
            embed_dim: self.embed_dim,
            dist_embed_num_blocks: self.dist_embed_num_blocks,
            dist_embed_num_heads: self.dist_embed_num_heads,
            dist_embed_num_inducing_points: self.dist_embed_num_inducing_points,
            feature_group_size: self.feature_group_size,
            feat_agg_num_blocks: self.feat_agg_num_blocks,
            feat_agg_num_heads: self.feat_agg_num_heads,
            feat_agg_num_cls_tokens: self.feat_agg_num_cls_tokens,
            feat_agg_rope_base: self.feat_agg_rope_base,
            nlayers: self.nlayers,
            icl_num_heads: self.icl_num_heads,
            icl_num_kv_heads: self.icl_num_kv_heads,
            icl_num_kv_heads_test: self.icl_num_kv_heads_test,
            decoder_head_dim: self.decoder_head_dim,
            decoder_num_heads: self.decoder_num_heads,
            decoder_use_softmax_scaling: self.decoder_use_softmax_scaling,
            ff_factor: self.ff_factor,
            softmax_scaling_mlp_hidden_dim: self.softmax_scaling_mlp_hidden_dim,
            layernorm_elementwise_affine: true,
            use_rope: true,
            use_nan_indicators: true,
        }
    }
}

#[derive(Debug, Clone)]
struct CellEmbedding {
    frequencies: Tensor,
    ecdf_frequencies: Tensor,
    fourier: Linear,
    metadata: Linear,
    norm: LayerNorm,
    group_size: usize,
}

impl CellEmbedding {
    fn new(config: &Config, vb: VarBuilder) -> Result<Self> {
        let group_size = config.feature_group_size;
        let num_freq = config.fourier_encoding_num_frequencies;
        let ecdf_freq = config.cell_ecdf_num_frequencies;
        Ok(Self {
            frequencies: vb.get((group_size, num_freq), "fourier.frequencies")?,
            ecdf_frequencies: Tensor::from_vec(
                (1..=ecdf_freq)
                    .map(|v| std::f32::consts::PI * v as f32)
                    .collect::<Vec<_>>(),
                ecdf_freq,
                vb.device(),
            )?,
            fourier: linear_no_bias(num_freq * 2, config.embed_dim, vb.pp("fourier.in_linear"))?,
            metadata: linear_no_bias(
                group_size * (2 + 2 * ecdf_freq),
                config.embed_dim,
                vb.pp("metadata_linear"),
            )?,
            norm: layer_norm(config.embed_dim, 1e-5, vb.pp("layernorm"))?,
            group_size,
        })
    }

    fn forward(&self, grouped: &Tensor) -> Result<Tensor> {
        let values = grouped.narrow(D::Minus1, 0, self.group_size)?;
        let phases = values
            .unsqueeze(D::Minus1)?
            .broadcast_mul(&self.frequencies)?;
        let fourier = Tensor::cat(&[phases.sin()?, phases.cos()?], D::Minus1)?;
        let fourier = self.fourier.forward(&fourier.sum(D::Minus2)?)?;
        let ranks = grouped.narrow(D::Minus1, self.group_size * 2, self.group_size)?;
        let phases = ranks
            .unsqueeze(D::Minus1)?
            .broadcast_mul(&self.ecdf_frequencies)?;
        let ecdf =
            Tensor::cat(&[phases.sin()?, phases.cos()?], D::Minus1)?.flatten_from(D::Minus2)?;
        let metadata = Tensor::cat(
            &[grouped.narrow(D::Minus1, 0, self.group_size * 2)?, ecdf],
            D::Minus1,
        )?;
        self.norm
            .forward(&(fourier + self.metadata.forward(&metadata)?)?)
    }
}

#[derive(Debug, Clone)]
struct PreHead {
    norm: Norm,
    mlp: Mlp,
}

impl PreHead {
    fn new(dim: usize, factor: usize, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            norm: Norm::new(dim, true, vb.pp("norm"))?,
            mlp: Mlp::new(dim, dim * factor, vb.pp("mlp"))?,
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        xs + self.mlp.forward(&self.norm.forward(xs)?)?
    }
}

#[derive(Debug, Clone)]
pub struct Model {
    config: Config,
    x_embed: CellEmbedding,
    col_y_encoder: Embedding,
    col_y_layernorm: LayerNorm,
    distribution: DistributionEmbedder,
    aggregator: ColumnAggregator,
    icl_y_encoder: Embedding,
    icl_y_layernorm: LayerNorm,
    icl_blocks: Vec<IclBlock>,
    output_norm: Norm,
    pre_head: PreHead,
    decoder: ManyClassDecoder,
}

impl Model {
    pub fn new(config: &Config, vb: VarBuilder) -> Result<Self> {
        if vb.dtype() != DType::F32 {
            candle::bail!("TabPFN v3.5 currently requires F32 weights")
        }
        if config.feature_group_size == 0
            || config.feature_group_size >= usize::BITS as usize
            || config.fourier_encoding_num_frequencies == 0
            || config.cell_ecdf_num_frequencies == 0
            || config.cell_ecdf_num_buckets < 2
        {
            candle::bail!("invalid TabPFN v3.5 cell embedding configuration")
        }
        let backbone = config.backbone();
        backbone.validate()?;
        let dim = config.embed_dim;
        let icl_dim = dim * config.feat_agg_num_cls_tokens;
        Ok(Self {
            config: config.clone(),
            x_embed: CellEmbedding::new(config, vb.pp("x_embed"))?,
            col_y_encoder: embedding(
                config.max_num_classes,
                dim,
                vb.pp("col_y_encoder.multiclass.embedding"),
            )?,
            col_y_layernorm: layer_norm(dim, 1e-5, vb.pp("col_y_layernorm"))?,
            distribution: DistributionEmbedder::new(
                &backbone,
                true,
                vb.pp("feature_distribution_embedder"),
            )?,
            aggregator: ColumnAggregator::new(&backbone, true, vb.pp("column_aggregator"))?,
            icl_y_encoder: embedding(
                config.max_num_classes,
                icl_dim,
                vb.pp("icl_y_encoder.multiclass.embedding"),
            )?,
            icl_y_layernorm: layer_norm(icl_dim, 1e-5, vb.pp("icl_y_layernorm"))?,
            icl_blocks: (0..config.nlayers)
                .map(|i| IclBlock::new(&backbone, true, vb.pp(format!("icl_blocks.{i}"))))
                .collect::<Result<_>>()?,
            output_norm: Norm::new(icl_dim, true, vb.pp("output_norm"))?,
            pre_head: PreHead::new(icl_dim, config.ff_factor, vb.pp("heads.mlp_classification"))?,
            decoder: ManyClassDecoder::new(&backbone, vb.pp("heads.many_class_decoder"))?,
        })
    }

    /// Return classification logits from F32 cells `[batch, rows, features]` and
    /// U32 training labels `[batch, train_len]`. Only training rows supply context.
    pub fn forward(&self, xs: &Tensor, ys: &Tensor) -> Result<Tensor> {
        let grouped = self.prepare_inputs(xs, ys)?;
        let device = self.col_y_encoder.embeddings().device();
        let (grouped, labels) = (grouped.to_device(device)?, ys.to_device(device)?);

        self.forward_prepared(&grouped, &labels)
    }

    /// Nonlearned feature preparation. Returns grouped F32 cells on the CPU.
    fn prepare_inputs(&self, xs: &Tensor, ys: &Tensor) -> Result<Tensor> {
        let (batch, rows, features) = xs.dims3()?;
        let (label_batch, train_len) = ys.dims2()?;
        if xs.dtype() != DType::F32 || ys.dtype() != DType::U32 {
            candle::bail!("TabPFN v3.5 expects F32 features and U32 labels")
        }
        if batch == 0
            || train_len == 0
            || train_len >= rows
            || features == 0
            || label_batch != batch
        {
            candle::bail!("invalid TabPFN v3.5 input shapes")
        }
        let labels = ys.to_vec2::<u32>()?;
        if labels
            .iter()
            .flatten()
            .any(|&v| v as usize >= self.config.max_num_classes)
        {
            candle::bail!("TabPFN v3.5 label exceeds configured class count")
        }
        preprocess(xs, train_len, &self.config)
    }

    /// Run every learned stage on grouped features and labels already on the model device.
    fn forward_prepared(&self, grouped: &Tensor, ys: &Tensor) -> Result<Tensor> {
        let (batch, rows, columns, width) = grouped.dims4()?;
        let (label_batch, train_len) = ys.dims2()?;
        if batch == 0
            || batch != label_batch
            || train_len == 0
            || train_len >= rows
            || columns == 0
            || width != self.config.feature_group_size * 3
            || grouped.dtype() != DType::F32
            || ys.dtype() != DType::U32
            || !grouped
                .device()
                .same_device(self.col_y_encoder.embeddings().device())
            || !ys
                .device()
                .same_device(self.col_y_encoder.embeddings().device())
        {
            candle::bail!("invalid prepared TabPFN v3.5 tensors or device")
        }
        let mut hidden = {
            let hidden = self.x_embed.forward(grouped)?;
            let labels = self
                .col_y_layernorm
                .forward(&self.col_y_encoder.forward(ys)?)?
                .unsqueeze(2)?;
            add_training_labels(&hidden, &labels)?
        };
        hidden = self.distribution.forward(&hidden, train_len)?;
        hidden = self.aggregator.forward(&hidden)?.flatten_from(2)?;
        hidden = {
            let labels = self
                .icl_y_layernorm
                .forward(&self.icl_y_encoder.forward(ys)?)?;
            add_training_labels(&hidden, &labels)?
        };
        for block in &self.icl_blocks {
            hidden = block.forward(&hidden, train_len)?;
        }

        hidden = self.pre_head.forward(&self.output_norm.forward(&hidden)?)?;
        finite_logits(&self.decoder.forward(
            &hidden.narrow(1, 0, train_len)?,
            &hidden.narrow(1, train_len, rows - train_len)?,
            ys,
        )?)
    }

    /// Predict a single deterministic ensemble member, using temperature 1.
    /// Labels must be contiguous integers starting at zero; NaN and infinity in
    /// features are imputed and accompanied by the learned missing-value indicators.
    pub fn predict_proba(
        &self,
        train_x: &Tensor,
        train_y: &Tensor,
        test_x: &Tensor,
    ) -> Result<Tensor> {
        self.predict_proba_with_temperature(train_x, train_y, test_x, 1.)
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

// Build ECDF edges by distinct values if they fit, otherwise by evenly spaced
// row positions. Counts are retained exactly, including ties at bucket edges.
fn ecdf_edges(sorted: &[f32], num_buckets: usize) -> Vec<(f32, usize, usize)> {
    let mut distinct = Vec::new();
    let mut start = 0;
    while start < sorted.len() {
        let mut end = start + 1;
        while end < sorted.len() && sorted[end] == sorted[start] {
            end += 1;
        }
        distinct.push((sorted[start], start, end));
        start = end;
    }
    let k = num_buckets.min(sorted.len());
    (0..k)
        .map(|i| {
            if k == 1 {
                return distinct[0];
            }
            if distinct.len() <= k {
                let index = (i as f32 * ((distinct.len() - 1) as f32 / (k - 1) as f32))
                    .round_ties_even() as usize;
                distinct[index]
            } else {
                let row = (i as f32 * ((sorted.len() - 1) as f32 / (k - 1) as f32))
                    .round_ties_even() as usize;
                let index = distinct.partition_point(|&(v, _, _)| v < sorted[row]);
                distinct[index]
            }
        })
        .collect()
}

fn ecdf_rank(value: f32, edges: &[(f32, usize, usize)], train_len: usize) -> f32 {
    let left = edges.partition_point(|&(v, _, _)| v < value);
    ecdf_rank_at(value, edges, train_len, left)
}

fn ecdf_rank_at(value: f32, edges: &[(f32, usize, usize)], train_len: usize, left: usize) -> f32 {
    let count = if left < edges.len() && edges[left].0 == value {
        (edges[left].1 + edges[left].2) as f32 * 0.5
    } else if left == 0 {
        0.
    } else if left == edges.len() {
        train_len as f32
    } else {
        let (low, _, at_most) = edges[left - 1];
        let (high, below, _) = edges[left];
        let weight = (value - low) / (high - low);
        at_most as f32 + weight * (below as f32 - at_most as f32)
    };
    count / train_len as f32
}

fn preprocess(xs: &Tensor, train_len: usize, config: &Config) -> Result<Tensor> {
    let (batch, rows, columns) = xs.dims3()?;
    let tables = xs.flatten_all()?.to_vec1::<f32>()?;
    let group = config.feature_group_size;
    let feature_indices = super::v3::feature_indices(columns, group);
    let mut grouped = Vec::with_capacity(batch * rows * columns * group * 3);
    for table in tables.chunks_exact(rows * columns) {
        let mut scaled = vec![0.; rows * columns];
        let mut indicators = vec![0.; rows * columns];
        let mut ranks = vec![0.; rows * columns];
        // Statistics and sorting repeatedly scan one column. Keep that working
        // column contiguous, while retaining row-major buffers for grouping.
        let mut values = vec![0.; rows];
        let mut finite = Vec::with_capacity(train_len);
        let mut sorted_rows = Vec::with_capacity(train_len);
        for column in 0..columns {
            for (row, value) in values.iter_mut().enumerate() {
                *value = table[row * columns + column];
            }
            finite.clear();
            finite.extend(
                values[..train_len]
                    .iter()
                    .copied()
                    .filter(|v| v.is_finite()),
            );
            let impute_mean = finite_imputation_mean(&finite);
            for (row, value) in values.iter_mut().enumerate() {
                if !value.is_finite() {
                    indicators[row * columns + column] = if value.is_nan() {
                        -2.
                    } else if value.is_sign_positive() {
                        2.
                    } else {
                        4.
                    };
                    *value = impute_mean;
                }
            }
            // Preserve training-row order in both F32 reductions.
            let mean = values[..train_len].iter().copied().sum::<f32>() / train_len as f32;
            let variance = values[..train_len]
                .iter()
                .map(|v| (v - mean).powi(2))
                .sum::<f32>()
                / train_len.saturating_sub(1).max(1) as f32;
            let std = if variance == 0. || train_len == 1 {
                1.
            } else {
                variance.sqrt()
            };
            // Upstream aligns imputed values to the fitted scaler mean before ranking.
            for (row, value) in values.iter_mut().enumerate() {
                if indicators[row * columns + column] != 0. {
                    *value = mean;
                }
            }
            sorted_rows.clear();
            sorted_rows.extend(
                values[..train_len]
                    .iter()
                    .copied()
                    .enumerate()
                    .map(|(row, value)| (value, row)),
            );
            sorted_rows.sort_unstable_by(|a, b| a.0.total_cmp(&b.0));
            let sorted = sorted_rows.iter().map(|&(v, _)| v).collect::<Vec<_>>();
            let edges = ecdf_edges(&sorted, config.cell_ecdf_num_buckets);
            // Training values already have a sorted order. Walk the edge lower
            // bounds once, including repeated edges and tied values, instead
            // of performing a binary search for every training cell.
            let mut left = 0;
            for &(value, row) in &sorted_rows {
                while left < edges.len() && edges[left].0 < value {
                    left += 1;
                }
                ranks[row * columns + column] = ecdf_rank_at(value, &edges, train_len, left);
            }
            for (row, &value) in values.iter().enumerate() {
                if row >= train_len {
                    ranks[row * columns + column] = ecdf_rank(value, &edges, train_len);
                }
                scaled[row * columns + column] =
                    ((value - mean) / (std + f32::EPSILON)).clamp(-100., 100.);
            }
        }
        for row in 0..rows {
            for column in 0..columns {
                for block in [&scaled, &indicators, &ranks] {
                    for &source in &feature_indices[column * group..(column + 1) * group] {
                        grouped.push(block[row * columns + source]);
                    }
                }
            }
        }
    }
    Tensor::from_vec(
        grouped,
        (batch, rows, columns, group * 3),
        &candle::Device::Cpu,
    )
}

#[cfg(test)]
mod tests;
