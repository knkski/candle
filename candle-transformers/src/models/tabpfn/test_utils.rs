//! Deterministic synthetic weights for tests that do not use checkpoints.

use candle::{DType, Device, Result, Shape, Tensor};
use candle_nn::{var_builder::SimpleBackend, Init, VarBuilder};

struct Weights;

impl SimpleBackend for Weights {
    fn get(
        &self,
        shape: Shape,
        name: &str,
        _hint: Init,
        dtype: DType,
        device: &Device,
    ) -> Result<Tensor> {
        let offset = name.bytes().map(usize::from).sum::<usize>();
        let norm = name.contains("norm") && name.ends_with("weight");
        let values = (0..shape.elem_count())
            .map(|i| {
                let value = (((i + offset) * 17 % 37) as f32 - 18.) / 128.;
                if norm {
                    1. + value
                } else {
                    value
                }
            })
            .collect::<Vec<_>>();
        Tensor::from_vec(values, shape, device)?.to_dtype(dtype)
    }

    fn get_unchecked(&self, name: &str, _dtype: DType, _device: &Device) -> Result<Tensor> {
        candle::bail!("synthetic test weight requires a shape: {name}")
    }

    fn contains_tensor(&self, name: &str) -> bool {
        name.ends_with("weight") || name.ends_with("bias")
    }
}

pub(super) fn weights(device: &Device) -> VarBuilder<'static> {
    VarBuilder::from_backend(Box::new(Weights), DType::F32, device.clone())
}
