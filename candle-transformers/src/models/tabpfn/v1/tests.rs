use super::{prepare_linear, Config, Model};
use candle::{Device, Result, Tensor, Var};
use candle_nn::{Linear, Module};

fn model() -> Result<Model> {
    let config = Config {
        num_features: 4,
        max_num_classes: 3,
        emsize: 8,
        nhead: 2,
        nhid_factor: 2,
        nlayers: 2,
    };
    Model::new(&config, super::super::test_utils::weights(&Device::Cpu))
}

fn tensor2(values: &[Vec<f32>]) -> Result<Tensor> {
    Tensor::from_vec(
        values.iter().flatten().copied().collect::<Vec<_>>(),
        (values.len(), values[0].len()),
        &Device::Cpu,
    )
}

fn assert_close(actual: &Tensor, expected: &Tensor) -> Result<()> {
    assert_eq!(actual.dims(), expected.dims());
    let actual = actual.flatten_all()?.to_vec1::<f32>()?;
    let expected = expected.flatten_all()?.to_vec1::<f32>()?;
    for (index, (actual, expected)) in actual.iter().zip(&expected).enumerate() {
        assert!(
            actual.is_finite() && (actual - expected).abs() < 1e-4,
            "element {index}: actual {actual}, expected {expected}"
        );
    }
    Ok(())
}

#[test]
fn preparing_trainable_weights_preserves_parameter_updates() -> Result<()> {
    let weight = Var::new(&[[1f32, 2.], [3., 4.]], &Device::Cpu)?;
    let layer = prepare_linear(Linear::new(weight.as_tensor().clone(), None))?;
    let input = Tensor::new(&[[1f32, 2.]], &Device::Cpu)?;
    assert_eq!(layer.forward(&input)?.to_vec2::<f32>()?, [[5., 11.]]);
    weight.set(&Tensor::new(&[[2f32, 4.], [6., 8.]], &Device::Cpu)?)?;
    assert_eq!(layer.forward(&input)?.to_vec2::<f32>()?, [[10., 22.]]);
    Ok(())
}

#[test]
fn query_only_layer_matches_full_layer_with_batched_inputs() -> Result<()> {
    let model = model()?;
    let hidden_size = model.config.emsize;
    for (batch, train_len, query_len) in [(1, 1, 1), (2, 3, 2), (2, 2, 5)] {
        let rows = train_len + query_len;
        let hidden = Tensor::from_vec(
            (0..batch * rows * hidden_size)
                .map(|index| ((index as f32 + 1.) * 0.17).sin())
                .collect::<Vec<_>>(),
            (batch, rows, hidden_size),
            &Device::Cpu,
        )?;
        for layer in &model.layers {
            let full = layer.forward(&hidden, train_len, false)?;
            let queries = layer.forward(&hidden, train_len, true)?;
            assert_close(&queries, &full.narrow(1, train_len, query_len)?)?;
        }
    }
    Ok(())
}

#[test]
fn invalid_tables_and_labels_return_errors() -> Result<()> {
    let model = model()?;
    let train_x = Tensor::new(
        &[
            [0f32, 1., 2.],
            [1., 2., 3.],
            [2., 3., 4.],
            [3., 4., 5.],
            [4., 5., 6.],
            [5., 6., 7.],
        ],
        &Device::Cpu,
    )?;
    let train_y = Tensor::new(&[0u32, 1, 2, 0, 1, 2], &Device::Cpu)?;
    let test_x = Tensor::new(&[[6f32, 7., 8.], [7., 8., 9.]], &Device::Cpu)?;

    for invalid_labels in [
        vec![0u32, 0, 0, 0, 0, 0],     // one class
        vec![0, 2, 2, 0, 0, 2],        // missing class 1
        vec![0, 1, 2, 3, 0, 1],        // exceeds checkpoint class count
        vec![0, 1, u32::MAX, 0, 1, 2], // must not overflow the class count
        vec![0, 1],                    // different number of rows
    ] {
        let labels = Tensor::new(invalid_labels.as_slice(), &Device::Cpu)?;
        assert!(model.predict_proba(&train_x, &labels, &test_x).is_err());
    }
    assert!(model
        .predict_proba(&train_x, &train_y, &test_x.narrow(1, 0, 2)?)
        .is_err());
    assert!(model
        .predict_proba(
            &train_x.narrow(0, 0, 1)?,
            &train_y.narrow(0, 0, 1)?,
            &test_x
        )
        .is_err());

    for bad_value in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        let mut invalid = test_x.to_vec2::<f32>()?;
        invalid[0][0] = bad_value;
        assert!(model
            .predict_proba(&train_x, &train_y, &tensor2(&invalid)?)
            .is_err());
    }
    Ok(())
}

// Reference the existing composed path so dispatch changes are checked against
// the same attention work, including the final layer's query-only slicing.
fn composed_attention(
    attention: &super::Attention,
    xs: &Tensor,
    train_len: usize,
    only_queries: bool,
) -> Result<Tensor> {
    let (batch, rows, hidden) = xs.dims3()?;
    let query_start = if only_queries { train_len } else { 0 };
    let query_rows = rows - query_start;
    let dim = hidden / attention.heads;
    let qkv = attention
        .qkv
        .forward(xs)?
        .reshape((batch, rows, 3, attention.heads, dim))?;
    let project = |index, start, len| -> Result<Tensor> {
        qkv.narrow(2, index, 1)?
            .squeeze(2)?
            .narrow(1, start, len)?
            .transpose(1, 2)
    };
    let q = (project(0, query_start, query_rows)? / (dim as f64).sqrt())?;
    let k = project(1, 0, train_len)?.contiguous()?;
    let v = project(2, 0, train_len)?.contiguous()?;
    let weights = candle_nn::ops::softmax_last_dim(&q.matmul(&k.t()?)?)?;
    attention.output.forward(
        &weights
            .matmul(&v)?
            .transpose(1, 2)?
            .reshape((batch, query_rows, hidden))?,
    )
}

#[test]
fn attention_private_pools_match_composed_full_and_query_only() -> Result<()> {
    let model = model()?;
    let attention = &model.layers[0].attention;
    for (batch, train_len, query_len) in [(1, 3, 2), (2, 31, 35)] {
        let rows = train_len + query_len;
        let hidden = model.config.emsize;
        let xs = Tensor::from_vec(
            (0..batch * rows * hidden)
                .map(|i| ((i as f32 + 1.) * 0.17).sin())
                .collect::<Vec<_>>(),
            (batch, rows, hidden),
            &Device::Cpu,
        )?;
        for only_queries in [false, true] {
            let expected = composed_attention(attention, &xs, train_len, only_queries)?;
            for threads in [1, 2] {
                let pool = rayon::ThreadPoolBuilder::new()
                    .num_threads(threads)
                    .build()
                    .unwrap();
                pool.install(|| -> Result<()> {
                    assert_eq!(rayon::current_num_threads(), threads);
                    assert_close(&attention.forward(&xs, train_len, only_queries)?, &expected)
                })?;
            }
        }
    }
    Ok(())
}

#[test]
fn tracked_attention_falls_back_to_existing_composed_behavior() -> Result<()> {
    let model = model()?;
    let attention = &model.layers[0].attention;
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(2)
        .build()
        .unwrap();
    let xs = Tensor::from_vec(
        (0..2 * 6 * 8)
            .map(|i| ((i as f32 + 1.) * 0.13).sin())
            .collect::<Vec<_>>(),
        (2, 6, 8),
        &Device::Cpu,
    )?;
    let input = Var::from_tensor(&xs)?;
    let derived = (input.as_tensor() * 1.125)?;
    assert!(derived.track_op() && !derived.is_variable());
    let weight = Var::from_tensor(attention.qkv.weight())?;
    let tracked_weights = super::Attention {
        qkv: Linear::new(weight.as_tensor().clone(), attention.qkv.bias().cloned()),
        output: attention.output.clone(),
        heads: attention.heads,
    };
    pool.install(|| -> Result<()> {
        for only_queries in [false, true] {
            for (attention, xs, variable) in [
                (attention, &derived, &input),
                (&tracked_weights, &xs, &weight),
            ] {
                let actual = attention.forward(xs, 4, only_queries)?;
                let expected = composed_attention(attention, xs, 4, only_queries)?;
                assert_close(&actual, &expected)?;
                assert!(actual.track_op());
                // Preserve the existing path's gradients; softmax_last_dim itself
                // is forward-only, so this is not a full-attention gradient claim.
                let actual_grad = actual.sqr()?.sum_all()?.backward()?;
                let expected_grad = expected.sqr()?.sum_all()?.backward()?;
                assert_close(
                    actual_grad
                        .get(variable)
                        .expect("missing fallback gradient"),
                    expected_grad
                        .get(variable)
                        .expect("missing composed gradient"),
                )?;
            }
        }
        Ok(())
    })
}
