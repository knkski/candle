use super::{Block, Config, Model, Variant};
use candle::{test_device, DType, Device, IndexOp, Result, Tensor};
use candle_nn::VarBuilder;
use serde_json::json;

#[test]
fn checkpoint_defaults_follow_the_selected_variant() -> Result<()> {
    for (variant, layers, heads, group, thinking, expansion) in [
        (Variant::V2, 12, 6, 2, 0, 4),
        (Variant::V2_5, 24, 3, 3, 64, 2),
        (Variant::V2_6, 24, 3, 3, 64, 2),
    ] {
        let config = Config::from_checkpoint(variant, json!({}))?;
        assert_eq!(config.variant, variant);
        assert_eq!(config.emsize, 192);
        assert_eq!(config.max_num_classes, 10);
        assert_eq!(config.nlayers, layers);
        assert_eq!(config.nhead, heads);
        assert_eq!(config.features_per_group, group);
        assert_eq!(config.thinking_rows, thinking);
        assert_eq!(config.variant.hidden_factor(), expansion);
        assert_eq!(config.encoder_hidden, None);
        assert_eq!(config.seed, 0);
    }
    Ok(())
}

#[test]
fn optional_components_resolve_only_for_supported_variants() -> Result<()> {
    for variant in [Variant::V2, Variant::V2_5, Variant::V2_6] {
        let config = Config::from_checkpoint(
            variant,
            json!({"encoder_type": "mlp", "encoder_mlp_hidden_dim": 12, "num_thinking_rows": 2}),
        )?;
        assert_eq!(
            config.encoder_hidden,
            if variant == Variant::V2 {
                None
            } else {
                Some(12)
            }
        );
        assert_eq!(
            config.thinking_rows,
            if variant == Variant::V2 { 0 } else { 2 }
        );
        let config = Config::from_checkpoint(
            variant,
            json!({"encoder_type": "linear", "encoder_mlp_hidden_dim": 0, "num_thinking_rows": 0}),
        )?;
        assert_eq!(config.encoder_hidden, None);
        assert_eq!(config.thinking_rows, 0);
    }
    Ok(())
}

#[test]
fn prepared_input_validation() -> Result<()> {
    let device = &Device::Cpu;
    let train = Tensor::new(&[[0f32, 1.], [1., 2.]], device)?;
    let labels = Tensor::new(&[0u32, 1], device)?;
    let query = Tensor::new(&[[2f32, 3.]], device)?;
    for variant in [Variant::V2, Variant::V2_5, Variant::V2_6] {
        let config =
            Config::from_checkpoint(variant, json!({"emsize": 8, "nhead": 2, "nlayers": 1}))?;
        let columns = Tensor::zeros((1, 2), DType::F32, device)?;
        let model = Model::new(&config, VarBuilder::zeros(DType::F32, device), &columns)?;
        let (features, targets) = model.prepare_inputs(&train, &labels, &query)?;
        assert!(features.device().is_cpu() && targets.device().is_cpu());
        assert!(model.forward_prepared(&features, &targets, 0).is_err());
        assert!(model
            .forward_prepared(&features, &targets.to_dtype(DType::F64)?, train.dim(0)?)
            .is_err());
    }
    Ok(())
}

fn final_block_matches_full_block(device: &Device) -> Result<()> {
    for (name, variant, legacy) in [
        ("v2-modern", Variant::V2, false),
        ("v2-legacy", Variant::V2, true),
        ("v2.5-modern", Variant::V2_5, false),
        ("v2.5-legacy", Variant::V2_5, true),
        ("v2.6-modern", Variant::V2_6, false),
    ] {
        // Head width 8 also exercises the CUDA attention kernel when available.
        let config = Config::from_checkpoint(
            variant,
            json!({
                "emsize": 16, "nhead": 2, "nlayers": 1, "num_thinking_rows": 2,
            }),
        )?;
        let block = Block::new(&config, super::super::test_utils::weights(device), legacy)?;
        for (train, queries, columns) in [(1, 1, 2), (5, 3, 7)] {
            // Thinking rows are part of the attention context, not predictions.
            let context = train + config.thinking_rows;
            let rows = context + queries;
            for (offset, scale) in [(0., 0.), (0., 1.), (10_000., 0.1)] {
                let input = Tensor::from_vec(
                    (0..rows * columns * config.emsize)
                        .map(|i| offset + (((i * 17) % 37) as f32 - 18.) * scale)
                        .collect::<Vec<_>>(),
                    (rows, columns, config.emsize),
                    device,
                )?;
                // Also exercise a noncontiguous hidden state, independent of the
                // particular layout produced by the preceding transformer layer.
                for input in [
                    input.clone(),
                    input.transpose(0, 1)?.contiguous()?.transpose(0, 1)?,
                ] {
                    let expected = block
                        .forward(&input, context)?
                        .narrow(0, context, queries)?
                        .i((.., columns - 1, ..))?;
                    let actual = block.forward_final(&input, context)?;
                    assert_eq!(actual.dims(), &[queries, config.emsize]);
                    let actual = actual.flatten_all()?.to_vec1::<f32>()?;
                    let expected = expected.flatten_all()?.to_vec1::<f32>()?;
                    for (actual, expected) in actual.into_iter().zip(expected) {
                        assert!(
                            actual.is_finite()
                                && expected.is_finite()
                                && (actual - expected).abs() <= 2e-5 * (1. + expected.abs()),
                            "{name}, train={train}, queries={queries}, columns={columns}, \
                             offset={offset}, scale={scale}: {actual} != {expected}"
                        );
                    }
                }
            }
        }
    }
    Ok(())
}

test_device!(
    final_block_matches_full_block,
    final_block_matches_full_block_cpu,
    final_block_matches_full_block_cuda,
    final_block_matches_full_block_metal
);

#[test]
fn checkpoint_configuration_rejects_invalid_dimensions() {
    for variant in [Variant::V2, Variant::V2_5, Variant::V2_6] {
        for value in [
            json!({"emsize": 3}),
            json!({"emsize": 16, "nhead": 3}),
            json!({"nlayers": 0}),
            json!({"nhead": 0}),
            json!({"features_per_group": 0}),
            json!({"max_num_classes": 1}),
            json!({"max_num_classes": usize::MAX}),
            json!({"emsize": usize::MAX, "nhead": 1}),
            json!({"emsize": 1usize << (usize::BITS / 2), "nhead": 1}),
            json!({"features_per_group": usize::MAX}),
        ] {
            assert!(
                Config::from_checkpoint(variant, value.clone()).is_err(),
                "accepted invalid {variant:?} dimensions: {value}"
            );
        }
    }
    assert!(Config::from_checkpoint(Variant::V2, json!({"features_per_group": 3})).is_err());
    for variant in [Variant::V2_5, Variant::V2_6] {
        for value in [
            json!({"encoder_type": "mlp", "encoder_mlp_hidden_dim": 0}),
            json!({"encoder_type": "mlp", "encoder_mlp_hidden_dim": usize::MAX}),
            json!({"num_thinking_rows": usize::MAX}),
        ] {
            assert!(Config::from_checkpoint(variant, value).is_err());
        }
    }
}

#[test]
fn checkpoint_configuration_rejects_malformed_values() {
    for variant in [Variant::V2, Variant::V2_5, Variant::V2_6] {
        for value in [
            json!(null),
            json!([]),
            json!({"emsize": null}),
            json!({"emsize": "192"}),
            json!({"nhead": 1.5}),
            json!({"nlayers": -1}),
            json!({"features_per_group": true}),
            json!({"max_num_classes": 1e40}),
        ] {
            assert!(
                Config::from_checkpoint(variant, value.clone()).is_err(),
                "accepted malformed {variant:?} configuration: {value}"
            );
        }
    }
    for variant in [Variant::V2_5, Variant::V2_6] {
        for value in [
            json!({"num_thinking_rows": null}),
            json!({"encoder_type": "unknown"}),
            json!({"encoder_mlp_hidden_dim": null}),
        ] {
            assert!(Config::from_checkpoint(variant, value).is_err());
        }
    }
}

#[test]
fn checkpoint_configuration_accepts_unrelated_upstream_fields() -> Result<()> {
    for variant in [Variant::V2, Variant::V2_5, Variant::V2_6] {
        let config = Config::from_checkpoint(
            variant,
            json!({
                "emsize": 24,
                "nhead": 3,
                "seed": 42,
                "name": "TabPFN",
                "num_buckets": -1,
                "unrelated_training_option": {"enabled": true},
            }),
        )?;
        assert_eq!(config.embedding_dim(), 24);
        assert_eq!(config.seed(), if variant == Variant::V2 { 42 } else { 0 });
    }
    Ok(())
}
