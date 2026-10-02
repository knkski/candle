//! Read architecture metadata without executing Python checkpoint objects.

use anyhow::{Context, Result};
use candle::pickle::{Object, Stack};
use serde_json::Value;
use std::{fs::File, io::BufReader, path::Path};

pub struct Metadata {
    pub config: Value,
    pub architecture: Option<String>,
    pub temperature: Option<f64>,
}

pub fn is_safetensors(path: &Path) -> bool {
    path.extension().is_some_and(|ext| ext == "safetensors")
}

pub fn metadata(path: &Path) -> Result<Metadata> {
    let fields = if is_safetensors(path) {
        let file = File::open(path)?;
        // The checkpoint is only read for the lifetime of this mapping.
        let mmap = unsafe { memmap2::MmapOptions::new().map(&file)? };
        let (_, header) = safetensors::SafeTensors::read_metadata(&mmap)?;
        let mut fields = serde_json::Map::new();
        if let Some(metadata) = header.metadata() {
            for name in ["config", "architecture_name", "inference_config"] {
                if let Some(value) = metadata.get(name) {
                    fields.insert(name.to_owned(), serde_json::from_str(value)?);
                }
            }
        }
        fields
    } else {
        let mut archive = zip::ZipArchive::new(BufReader::new(File::open(path)?))?;
        let names = archive
            .file_names()
            .filter(|name| name.ends_with("/data.pkl") || *name == "data.pkl")
            .map(str::to_owned)
            .collect::<Vec<_>>();
        anyhow::ensure!(names.len() == 1, "expected one PyTorch data.pkl entry");
        let mut stack = Stack::empty();
        stack.read_loop(&mut BufReader::new(archive.by_name(&names[0])?))?;
        let Object::Dict(entries) = stack.finalize()? else {
            anyhow::bail!("expected a dictionary checkpoint for TabPFN v2 or later")
        };
        let mut fields = serde_json::Map::new();
        for (key, value) in entries {
            if let Object::Unicode(name) = key {
                if matches!(
                    name.as_str(),
                    "config" | "architecture_name" | "inference_config"
                ) {
                    fields.insert(name, to_json(value)?);
                }
            }
        }
        fields
    };
    parse_fields(fields)
}

fn parse_fields(mut fields: serde_json::Map<String, Value>) -> Result<Metadata> {
    let config = fields
        .remove("config")
        .context("checkpoint has no config")?;
    anyhow::ensure!(config.is_object(), "checkpoint config must be a dictionary");
    let architecture = match fields.remove("architecture_name") {
        Some(Value::String(name)) => Some(name),
        None | Some(Value::Null) => None,
        _ => anyhow::bail!("checkpoint architecture_name must be a string"),
    };
    let temperature = fields
        .get("inference_config")
        .and_then(|config| config.get("SOFTMAX_TEMPERATURE"))
        .map(|value| {
            value
                .as_f64()
                .context("checkpoint temperature must be a number")
        })
        .transpose()?;
    Ok(Metadata {
        config,
        architecture,
        temperature,
    })
}

fn to_json(object: Object) -> Result<Value> {
    Ok(match object {
        Object::None => Value::Null,
        Object::Bool(value) => value.into(),
        Object::Int(value) => value.into(),
        Object::Long(value) => value.into(),
        Object::Float(value) => serde_json::Number::from_f64(value)
            .context("nonfinite checkpoint metadata")?
            .into(),
        Object::Unicode(value) => value.into(),
        Object::List(values) | Object::Tuple(values) => {
            Value::Array(values.into_iter().map(to_json).collect::<Result<_>>()?)
        }
        Object::Dict(entries) => {
            let mut values = serde_json::Map::new();
            for (key, value) in entries {
                let Object::Unicode(key) = key else {
                    anyhow::bail!("checkpoint metadata keys must be strings")
                };
                values.insert(key, to_json(value)?);
            }
            Value::Object(values)
        }
        _ => anyhow::bail!("unsupported Python object in checkpoint metadata"),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn config_is_required_but_optional_metadata_may_be_absent() -> Result<()> {
        for (fields, expected_error) in [
            (json!({}), "checkpoint has no config"),
            (
                json!({"config": null}),
                "checkpoint config must be a dictionary",
            ),
            (
                json!({"config": []}),
                "checkpoint config must be a dictionary",
            ),
        ] {
            let result = parse_fields(fields.as_object().unwrap().clone());
            assert_eq!(result.err().unwrap().to_string(), expected_error);
        }
        // Original v2 files have neither architecture_name nor inference_config;
        // v2.5 files may store inference_config as None.
        for fields in [
            json!({"config": {}}),
            json!({"config": {}, "inference_config": null}),
        ] {
            let parsed = parse_fields(fields.as_object().unwrap().clone())?;
            assert_eq!(parsed.config, json!({}));
            assert!(parsed.architecture.is_none());
            assert!(parsed.temperature.is_none());
        }
        Ok(())
    }
}
