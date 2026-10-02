use super::{ecdf_edges, ecdf_rank, finite_imputation_mean, preprocess, Config, Model};
use crate::models::tabpfn::v3::feature_indices;
use candle::{DType, Device, Result, Tensor};
use candle_nn::VarBuilder;

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
        fourier_encoding_num_frequencies: 3,
        cell_ecdf_num_frequencies: 2,
        cell_ecdf_num_buckets: 4,
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

// The original row-major implementation searches every training and query
// value independently, providing a reference for the sorted training sweep.
fn reference_preprocess(
    input: &[f32],
    train_len: usize,
    rows: usize,
    columns: usize,
    config: &Config,
) -> Vec<f32> {
    let batch = 1;
    let mut values = vec![input
        .chunks_exact(columns)
        .map(|r| r.to_vec())
        .collect::<Vec<_>>()];
    let group = config.feature_group_size;
    let feature_indices = feature_indices(columns, group);
    let mut grouped = Vec::with_capacity(batch * rows * columns * group * 3);
    for table in &mut values {
        let mut indicators = vec![vec![0.; columns]; rows];
        let mut ranks = vec![vec![0.; columns]; rows];
        for column in 0..columns {
            let finite = table[..train_len]
                .iter()
                .map(|r| r[column])
                .filter(|v| v.is_finite())
                .collect::<Vec<_>>();
            let impute_mean = finite_imputation_mean(&finite);
            for (row, indicator) in table.iter_mut().zip(&mut indicators) {
                let value = row[column];
                if !value.is_finite() {
                    indicator[column] = if value.is_nan() {
                        -2.
                    } else if value.is_sign_positive() {
                        2.
                    } else {
                        4.
                    };
                    row[column] = impute_mean;
                }
            }
            let mean = table[..train_len].iter().map(|r| r[column]).sum::<f32>() / train_len as f32;
            let variance = table[..train_len]
                .iter()
                .map(|r| (r[column] - mean).powi(2))
                .sum::<f32>()
                / train_len.saturating_sub(1).max(1) as f32;
            let std = if variance == 0. || train_len == 1 {
                1.
            } else {
                variance.sqrt()
            };
            // Upstream aligns imputed values to the fitted scaler mean before ranking.
            for (row, indicator) in table.iter_mut().zip(&indicators) {
                if indicator[column] != 0. {
                    row[column] = mean;
                }
            }
            let mut sorted = table[..train_len]
                .iter()
                .map(|r| r[column])
                .collect::<Vec<_>>();
            sorted.sort_unstable_by(f32::total_cmp);
            let edges = ecdf_edges(&sorted, config.cell_ecdf_num_buckets);
            for (row, rank) in table.iter_mut().zip(&mut ranks) {
                rank[column] = ecdf_rank(row[column], &edges, train_len);
                row[column] = ((row[column] - mean) / (std + f32::EPSILON)).clamp(-100., 100.);
            }
        }
        for row in 0..rows {
            for column in 0..columns {
                for block in [&table[row], &indicators[row], &ranks[row]] {
                    for &source in &feature_indices[column * group..(column + 1) * group] {
                        grouped.push(block[source]);
                    }
                }
            }
        }
    }
    grouped
}

fn assert_reference(
    input: &[f32],
    batch: usize,
    rows: usize,
    columns: usize,
    train: usize,
    config: &Config,
) -> Result<()> {
    let expected = input
        .chunks_exact(rows * columns)
        .flat_map(|table| reference_preprocess(table, train, rows, columns, config))
        .collect::<Vec<_>>();
    let xs = Tensor::from_vec(input.to_vec(), (batch, rows, columns), &Device::Cpu)?;
    let strided = xs.transpose(1, 2)?.contiguous()?.transpose(1, 2)?;
    for xs in [&xs, &strided] {
        let actual = preprocess(xs, train, config)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        assert_eq!(actual.len(), expected.len());
        for (index, (&actual, &expected)) in actual.iter().zip(&expected).enumerate() {
            assert_eq!(
                actual.to_bits(),
                expected.to_bits(),
                "element {index}, train={train}, buckets={}: {actual:?} != {expected:?}",
                config.cell_ecdf_num_buckets
            );
        }
    }
    Ok(())
}

#[test]
fn training_rank_sweep_matches_independent_search() -> Result<()> {
    let choices = [
        0.,
        -0.,
        1.,
        -1.,
        3.,
        f32::NAN,
        f32::INFINITY,
        f32::NEG_INFINITY,
        f32::MAX,
        f32::MIN,
    ];
    let mut state = 917_u64;
    for train in [1, 2, 3, 7, 31, 128] {
        for columns in [1, 3, 8] {
            for buckets in [2, 3, 7, 8192] {
                let rows = train + 11;
                let config = Config {
                    cell_ecdf_num_buckets: buckets,
                    ..Config::default()
                };
                let input = (0..2 * rows * columns)
                    .map(|_| {
                        state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
                        choices[((state >> 32) as usize) % choices.len()]
                    })
                    .collect::<Vec<_>>();
                assert_reference(&input, 2, rows, columns, train, &config)?;
            }
        }
    }
    Ok(())
}

#[test]
fn column_scratch_preserves_constant_and_missing_columns() -> Result<()> {
    let input = [
        [f32::NAN, 4., -0., f32::INFINITY],
        [f32::NAN, 4., 0., f32::NEG_INFINITY],
        [f32::NAN, 4., 1., f32::NAN],
        [f32::INFINITY, 4., -1., f32::NAN],
        [f32::NEG_INFINITY, 5., 0., f32::INFINITY],
    ];
    let input = input.into_iter().flatten().collect::<Vec<_>>();
    for buckets in [2, 8192] {
        for group in [1, 3, 5] {
            let config = Config {
                cell_ecdf_num_buckets: buckets,
                feature_group_size: group,
                ..Config::default()
            };
            assert_reference(&input, 1, 5, 4, 3, &config)?;
        }
    }
    Ok(())
}
