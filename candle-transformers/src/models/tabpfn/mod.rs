//! TabPFN transformers for tabular classification.
//!
//! Each architecture family exposes its own `Config` and `Model`.
//! Labeled training rows provide context for predictions without gradient updates.
//! The versions share components where their architectures overlap:
//!
//! - [`v2`] supports v2, v2.5, and v2.6 through [`v2::Variant`].
//! - [`v3_5`] reuses components from [`v3`] and supports both v3.5 and v3.5 Fast.
//!
//! The `tabpfn` example handles version selection and checkpoint downloads.

mod attention_projection;

#[cfg(test)]
mod test_utils;

pub mod v1;
pub mod v2;
pub mod v3;
pub mod v3_5;
