//! Inference-only packing of TabPFN's bias-free attention projections.

use candle::{DType, Result, Tensor};
use candle_nn::{Linear, Module};

#[derive(Debug, Clone)]
struct Packed {
    qkv: Linear,
    kv: Linear,
}

#[derive(Debug, Clone)]
pub(super) struct AttentionProjections {
    q: Linear,
    k: Linear,
    v: Linear,
    packed: Option<Packed>,
    heads: usize,
    kv_heads: usize,
    head_dim: usize,
}

impl AttentionProjections {
    pub(super) fn new(
        mut q: Linear,
        mut k: Linear,
        mut v: Linear,
        heads: usize,
        kv_heads: usize,
        head_dim: usize,
    ) -> Result<Self> {
        let q_width = heads.checked_mul(head_dim);
        let kv_width = kv_heads.checked_mul(head_dim);
        let total_width = kv_width
            .and_then(|width| width.checked_mul(2))
            .and_then(|width| q_width.and_then(|q_width| q_width.checked_add(width)));
        if heads == 0 || kv_heads == 0 || head_dim == 0 || total_width.is_none() {
            candle::bail!("invalid TabPFN attention projection dimensions")
        }
        let (q_width, kv_width) = (q_width.unwrap(), kv_width.unwrap());
        let (output, input) = q.weight().dims2()?;
        if output != q_width
            || k.weight().dims2()? != (kv_width, input)
            || v.weight().dims2()? != (kv_width, input)
        {
            candle::bail!("TabPFN attention projection weights do not match head dimensions")
        }
        let device = q.weight().device();
        let can_pack = device.is_cuda()
            && [&q, &k, &v].into_iter().all(|linear| {
                linear.bias().is_none()
                    && linear.weight().dtype() == DType::F32
                    && !linear.weight().track_op()
                    && linear.weight().device().same_device(device)
            });
        let packed = if can_pack {
            let weight = Tensor::cat(&[q.weight(), k.weight(), v.weight()], 0)?.contiguous()?;
            // All projections share this allocation. Keep separate views for
            // cross-attention Q and the tracked-input fallback, rather than
            // retaining a second copy of every original weight.
            q = Linear::new(weight.narrow(0, 0, q_width)?, None);
            k = Linear::new(weight.narrow(0, q_width, kv_width)?, None);
            v = Linear::new(weight.narrow(0, q_width + kv_width, kv_width)?, None);
            Some(Packed {
                kv: Linear::new(weight.narrow(0, q_width, 2 * kv_width)?, None),
                qkv: Linear::new(weight, None),
            })
        } else {
            None
        };
        Ok(Self {
            q,
            k,
            v,
            packed,
            heads,
            kv_heads,
            head_dim,
        })
    }

    fn row_major_matrices(x: &Tensor) -> bool {
        matches!(x.dims(), [_, _, width] if x.stride()[2] == 1 && x.stride()[1] == *width)
    }

    fn packed_for(&self, x: &Tensor, context: &Tensor) -> Option<&Packed> {
        self.packed.as_ref().filter(|_| {
            x.dtype() == DType::F32
                && context.dtype() == DType::F32
                && x.device().same_device(self.q.weight().device())
                && context.device().same_device(x.device())
                && !x.track_op()
                && !context.track_op()
        })
    }

    /// Returns Q/K/V as (batch, rows, heads, head_dim), retaining strided views.
    ///
    /// Self-attention is opt-in, and identity is checked before making either
    /// input contiguous. Prefix/cross attention projects KV only for context.
    pub(super) fn forward(
        &self,
        x: &Tensor,
        context: &Tensor,
        self_attention: bool,
    ) -> Result<(Tensor, Tensor, Tensor)> {
        let (batch, rows, _) = x.dims3()?;
        let count = context.dim(1)?;
        if let Some(packed) = self.packed_for(x, context) {
            if self_attention && x.id() == context.id() {
                let total_heads = self.heads + 2 * self.kv_heads;
                let projected = packed.qkv.forward(&x.contiguous()?)?.reshape((
                    batch,
                    rows,
                    total_heads,
                    self.head_dim,
                ))?;
                // Reshape the FULL contiguous projection before slicing heads.
                // Reshaping a narrowed output would copy its row-strided view.
                return Ok((
                    projected.narrow(2, 0, self.heads)?,
                    projected.narrow(2, self.heads, self.kv_heads)?,
                    projected.narrow(2, self.heads + self.kv_heads, self.kv_heads)?,
                ));
            }
            let q = self.q.forward(&x.contiguous()?)?.reshape((
                batch,
                rows,
                self.heads,
                self.head_dim,
            ))?;
            // A row-major matrix per batch may still have gaps between
            // batches, as when selecting training rows before query rows.
            // Linear can project that prefix with a strided-batched GEMM,
            // avoiding a full input copy. Other layouts retain materialization.
            let context = if Self::row_major_matrices(context) {
                context.clone()
            } else {
                context.contiguous()?
            };
            let kv = packed.kv.forward(&context)?.reshape((
                batch,
                count,
                2 * self.kv_heads,
                self.head_dim,
            ))?;
            return Ok((
                q,
                kv.narrow(2, 0, self.kv_heads)?,
                kv.narrow(2, self.kv_heads, self.kv_heads)?,
            ));
        }
        // Preserve ordinary Linear operations for CPU, other dtypes, biases,
        // and any tracked tensor. CUDA context views retain the original flat
        // projection. Materialize arbitrary matrix strides on every backend:
        // the CPU batched-matmul folding path assumes row-major matrices.
        let x = if Self::row_major_matrices(x) {
            x.clone()
        } else {
            x.contiguous()?
        };
        let context = if x.device().is_cuda() || !Self::row_major_matrices(context) {
            context.contiguous()?
        } else {
            context.clone()
        };
        Ok((
            self.q
                .forward(&x)?
                .reshape((batch, rows, self.heads, self.head_dim))?,
            self.k
                .forward(&context)?
                .reshape((batch, count, self.kv_heads, self.head_dim))?,
            self.v
                .forward(&context)?
                .reshape((batch, count, self.kv_heads, self.head_dim))?,
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle::{test_device, Device, Var};

    fn weight(rows: usize, columns: usize, offset: usize, device: &Device) -> Result<Tensor> {
        Tensor::from_vec(
            (0..rows * columns)
                .map(|i| (((i + offset) * 17 % 37) as f32 - 18.) / 13.)
                .collect::<Vec<_>>(),
            (rows, columns),
            device,
        )
    }

    fn projections(device: &Device, heads: usize, kv_heads: usize) -> Result<AttentionProjections> {
        AttentionProjections::new(
            Linear::new(weight(heads * 4, 8, 1, device)?, None),
            Linear::new(weight(kv_heads * 4, 8, 7, device)?, None),
            Linear::new(weight(kv_heads * 4, 8, 13, device)?, None),
            heads,
            kv_heads,
            4,
        )
    }

    fn close(actual: &Tensor, expected: &Tensor) -> Result<()> {
        assert_eq!(actual.dims(), expected.dims());
        let actual = actual.flatten_all()?.to_vec1::<f32>()?;
        let expected = expected.flatten_all()?.to_vec1::<f32>()?;
        for (a, b) in actual.iter().zip(expected) {
            assert!((a - b).abs() <= 2e-5 * (1. + b.abs()), "{a} != {b}");
        }
        Ok(())
    }

    fn packed_projections(device: &Device) -> Result<()> {
        for kv_heads in [1, 2] {
            let projected = projections(device, 2, kv_heads)?;
            assert_eq!(projected.packed.is_some(), device.is_cuda());
            let x = weight(14, 8, 17, device)?.reshape((2, 7, 8))?;
            // Same tensor identity, a prefix, and a distinct equal-valued input
            // exercise QKV, Q+KV, and the self-attention identity guard.
            for context in [x.clone(), x.narrow(1, 0, 3)?, x.copy()?] {
                let count = context.dim(1)?;
                let (q, k, v) = projected.forward(&x, &context, true)?;
                close(&q, &projected.q.forward(&x)?.reshape((2, 7, 2, 4))?)?;
                close(
                    &k,
                    &projected
                        .k
                        .forward(&context.contiguous()?)?
                        .reshape((2, count, kv_heads, 4))?,
                )?;
                close(
                    &v,
                    &projected
                        .v
                        .forward(&context.contiguous()?)?
                        .reshape((2, count, kv_heads, 4))?,
                )?;
                if device.is_cuda() {
                    let stride = if x.id() == context.id() {
                        (2 + 2 * kv_heads) * 4
                    } else {
                        2 * kv_heads * 4
                    };
                    assert_eq!(k.stride(), &[count * stride, stride, 4, 1]);
                    assert_eq!(v.stride(), k.stride());
                    assert!(!k.is_contiguous() && !v.is_contiguous());
                    if x.id() == context.id() {
                        assert_eq!(q.stride(), k.stride());
                    }
                }
            }
            // Query and context input views have nonzero offsets and strides.
            let x = weight(20, 8, 21, device)?
                .reshape((2, 10, 8))?
                .narrow(1, 1, 7)?;
            let context = x.narrow(1, 1, 3)?;
            let (q, k, v) = projected.forward(&x, &context, false)?;
            close(
                &q,
                &projected
                    .q
                    .forward(&x.contiguous()?)?
                    .reshape((2, 7, 2, 4))?,
            )?;
            close(
                &k,
                &projected
                    .k
                    .forward(&context.contiguous()?)?
                    .reshape((2, 3, kv_heads, 4))?,
            )?;
            close(
                &v,
                &projected
                    .v
                    .forward(&context.contiguous()?)?
                    .reshape((2, 3, kv_heads, 4))?,
            )?;

            // Arbitrary embedding/row strides cannot use the direct batched
            // projection. These layouts must retain the contiguous fallback.
            let contexts = [
                weight(16, 6, 27, device)?
                    .reshape((2, 8, 6))?
                    .transpose(1, 2)?,
                weight(12, 16, 31, device)?
                    .reshape((2, 6, 16))?
                    .narrow(2, 3, 8)?,
            ];
            for context in contexts {
                assert!(!context.is_contiguous());
                assert!(context.stride()[2] != 1 || context.stride()[1] != 8);
                let (q, k, v) = projected.forward(&x, &context, false)?;
                close(
                    &q,
                    &projected
                        .q
                        .forward(&x.contiguous()?)?
                        .reshape((2, 7, 2, 4))?,
                )?;
                close(
                    &k,
                    &projected
                        .k
                        .forward(&context.contiguous()?)?
                        .reshape((2, 6, kv_heads, 4))?,
                )?;
                close(
                    &v,
                    &projected
                        .v
                        .forward(&context.contiguous()?)?
                        .reshape((2, 6, kv_heads, 4))?,
                )?;
                // The ordinary Q fallback must handle the same layouts too.
                let (q, _, _) = projected.forward(&context, &x, false)?;
                close(
                    &q,
                    &projected
                        .q
                        .forward(&context.contiguous()?)?
                        .reshape((2, 6, 2, 4))?,
                )?;
            }
        }
        Ok(())
    }

    fn tracked_fallback(device: &Device) -> Result<()> {
        let projected = projections(device, 2, 1)?;
        let x = Var::from_tensor(&weight(10, 8, 3, device)?.reshape((2, 5, 8))?)?;
        let derived = (x.as_tensor() * 1.3)?;
        assert!(projected.packed_for(&derived, &derived).is_none());
        let (q, k, v) = projected.forward(&derived, &derived, true)?;
        let actual = (q.sum_all()? + k.sum_all()?)?
            .add(&v.sum_all()?)?
            .backward()?;
        let expected = (projected.q.forward(&derived)?.sum_all()?
            + projected.k.forward(&derived)?.sum_all()?)?
        .add(&projected.v.forward(&derived)?.sum_all()?)?
        .backward()?;
        close(actual.get(&x).unwrap(), expected.get(&x).unwrap())?;

        let plain = x.as_tensor().detach();
        assert!(projected.packed_for(&plain, &derived).is_none());
        let (_, k, v) = projected.forward(&plain, &derived, false)?;
        assert!((k.sum_all()? + v.sum_all()?)?.backward()?.get(&x).is_some());
        for tracked in 0..3 {
            let variable = Var::from_tensor(&weight(8, 8, 9, device)?)?;
            let derived_weight = (variable.as_tensor() * 1.1)?;
            let layers = (0..3)
                .map(|index| {
                    Linear::new(
                        if index == tracked {
                            derived_weight.clone()
                        } else {
                            variable.as_tensor().detach()
                        },
                        None,
                    )
                })
                .collect::<Vec<_>>();
            let projected = AttentionProjections::new(
                layers[0].clone(),
                layers[1].clone(),
                layers[2].clone(),
                2,
                2,
                4,
            )?;
            assert!(projected.packed.is_none());
            let (q, k, v) = projected.forward(&plain, &plain, true)?;
            assert!((q.sum_all()? + k.sum_all()?)?
                .add(&v.sum_all()?)?
                .backward()?
                .get(&variable)
                .is_some());
        }
        Ok(())
    }

    #[test]
    fn biased_and_other_dtype_fallback() -> Result<()> {
        for dtype in [DType::F32, DType::F64] {
            let weight = weight(8, 8, 2, &Device::Cpu)?.to_dtype(dtype)?;
            let bias = Tensor::ones(8, dtype, &Device::Cpu)?;
            let linear = Linear::new(weight, Some(bias));
            let projected =
                AttentionProjections::new(linear.clone(), linear.clone(), linear.clone(), 2, 2, 4)?;
            assert!(projected.packed.is_none());
            let x = Tensor::ones((2, 5, 8), dtype, &Device::Cpu)?;
            let (q, _, _) = projected.forward(&x, &x, true)?;
            let difference = q
                .reshape((2, 5, 8))?
                .sub(&linear.forward(&x)?)?
                .abs()?
                .sum_all()?;
            assert_eq!(difference.to_dtype(DType::F64)?.to_scalar::<f64>()?, 0.);
        }
        Ok(())
    }

    test_device!(
        packed_projections,
        packed_projections_cpu,
        packed_projections_cuda,
        packed_projections_metal
    );
    test_device!(
        tracked_fallback,
        tracked_fallback_cpu,
        tracked_fallback_cuda,
        tracked_fallback_metal
    );
}
