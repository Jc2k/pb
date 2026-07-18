//! Bounded GGUF v3 directory parser used by the FlashMoe source adapters.
//!
//! This parser deliberately stops at the tensor payload. Model-family adapters
//! validate metadata and tensor semantics once, then publish the source into
//! FlashMoe's canonical resident and streamed-expert stores.

use std::collections::BTreeMap;
use std::fs::File;
use std::io::{BufReader, Read};
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};

const GGUF_MAGIC: u32 = 0x4655_4747;
const GGUF_VERSION: u32 = 3;
const MAX_DIRECTORY_ENTRIES: u64 = 1_000_000;
const MAX_ARRAY_ITEMS: u64 = 16_000_000;
const MAX_STRING_BYTES: u64 = 64 * 1024 * 1024;
const MAX_DIMS: u32 = 8;
const MAX_ARRAY_DEPTH: usize = 8;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub(crate) enum GgufMetadataType {
    Uint8 = 0,
    Int8 = 1,
    Uint16 = 2,
    Int16 = 3,
    Uint32 = 4,
    Int32 = 5,
    Float32 = 6,
    Bool = 7,
    String = 8,
    Array = 9,
    Uint64 = 10,
    Int64 = 11,
    Float64 = 12,
}

impl GgufMetadataType {
    fn from_u32(value: u32) -> Result<Self> {
        match value {
            0 => Ok(Self::Uint8),
            1 => Ok(Self::Int8),
            2 => Ok(Self::Uint16),
            3 => Ok(Self::Int16),
            4 => Ok(Self::Uint32),
            5 => Ok(Self::Int32),
            6 => Ok(Self::Float32),
            7 => Ok(Self::Bool),
            8 => Ok(Self::String),
            9 => Ok(Self::Array),
            10 => Ok(Self::Uint64),
            11 => Ok(Self::Int64),
            12 => Ok(Self::Float64),
            _ => bail!("unsupported GGUF metadata type {value}"),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) enum GgufValue {
    Uint8(u8),
    Int8(i8),
    Uint16(u16),
    Int16(i16),
    Uint32(u32),
    Int32(i32),
    Float32(f32),
    Bool(bool),
    String(String),
    Array {
        element_type: GgufMetadataType,
        values: Vec<GgufValue>,
    },
    Uint64(u64),
    Int64(i64),
    Float64(f64),
}

impl GgufValue {
    pub(crate) fn as_u64_compat(&self) -> Option<u64> {
        match self {
            Self::Uint32(value) => Some(u64::from(*value)),
            Self::Uint64(value) => Some(*value),
            _ => None,
        }
    }

    pub(crate) fn as_f64_compat(&self) -> Option<f64> {
        match self {
            Self::Float32(value) => Some(f64::from(*value)),
            Self::Float64(value) => Some(*value),
            Self::Uint32(value) => Some(f64::from(*value)),
            Self::Int32(value) => Some(f64::from(*value)),
            _ => None,
        }
    }

    pub(crate) fn as_bool(&self) -> Option<bool> {
        match self {
            Self::Bool(value) => Some(*value),
            _ => None,
        }
    }

    pub(crate) fn as_str(&self) -> Option<&str> {
        match self {
            Self::String(value) => Some(value),
            _ => None,
        }
    }

    pub(crate) fn as_array(&self) -> Option<(GgufMetadataType, &[GgufValue])> {
        match self {
            Self::Array {
                element_type,
                values,
            } => Some((*element_type, values)),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct GgufTensorType {
    pub(crate) id: u32,
    pub(crate) name: &'static str,
    pub(crate) block_elements: u64,
    pub(crate) block_bytes: u64,
}

impl GgufTensorType {
    pub(crate) const F32: Self = Self::new(0, "F32", 1, 4);
    pub(crate) const F16: Self = Self::new(1, "F16", 1, 2);
    pub(crate) const Q8_0: Self = Self::new(8, "Q8_0", 32, 34);
    pub(crate) const Q2_K: Self = Self::new(10, "Q2_K", 256, 84);
    pub(crate) const Q4_K: Self = Self::new(12, "Q4_K", 256, 144);
    pub(crate) const IQ2_XXS: Self = Self::new(16, "IQ2_XXS", 256, 66);
    pub(crate) const I32: Self = Self::new(26, "I32", 1, 4);

    const fn new(id: u32, name: &'static str, block_elements: u64, block_bytes: u64) -> Self {
        Self {
            id,
            name,
            block_elements,
            block_bytes,
        }
    }

    fn from_u32(value: u32) -> Result<Self> {
        let kind = match value {
            0 => Self::F32,
            1 => Self::F16,
            2 => Self::new(2, "Q4_0", 32, 18),
            3 => Self::new(3, "Q4_1", 32, 20),
            6 => Self::new(6, "Q5_0", 32, 22),
            7 => Self::new(7, "Q5_1", 32, 24),
            8 => Self::Q8_0,
            9 => Self::new(9, "Q8_1", 32, 40),
            10 => Self::Q2_K,
            11 => Self::new(11, "Q3_K", 256, 110),
            12 => Self::Q4_K,
            13 => Self::new(13, "Q5_K", 256, 176),
            14 => Self::new(14, "Q6_K", 256, 210),
            15 => Self::new(15, "Q8_K", 256, 292),
            16 => Self::IQ2_XXS,
            17 => Self::new(17, "IQ2_XS", 256, 74),
            18 => Self::new(18, "IQ3_XXS", 256, 98),
            19 => Self::new(19, "IQ1_S", 256, 110),
            20 => Self::new(20, "IQ4_NL", 256, 50),
            21 => Self::new(21, "IQ3_S", 256, 110),
            22 => Self::new(22, "IQ2_S", 256, 82),
            23 => Self::new(23, "IQ4_XS", 256, 136),
            24 => Self::new(24, "I8", 1, 1),
            25 => Self::new(25, "I16", 1, 2),
            26 => Self::I32,
            27 => Self::new(27, "I64", 1, 8),
            28 => Self::new(28, "F64", 1, 8),
            29 => Self::new(29, "IQ1_M", 256, 56),
            30 => Self::new(30, "BF16", 1, 2),
            _ => bail!("unsupported GGUF tensor type {value}"),
        };
        Ok(kind)
    }

    fn byte_len(self, elements: u64) -> Result<u64> {
        let blocks = elements
            .checked_add(self.block_elements - 1)
            .context("GGUF tensor block count overflow")?
            / self.block_elements;
        blocks
            .checked_mul(self.block_bytes)
            .context("GGUF tensor byte length overflow")
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct GgufTensorInfo {
    pub(crate) name: String,
    pub(crate) dimensions: Vec<u64>,
    pub(crate) tensor_type: GgufTensorType,
    pub(crate) relative_offset: u64,
    pub(crate) absolute_offset: u64,
    pub(crate) byte_len: u64,
}

#[derive(Debug)]
pub(crate) struct GgufFile {
    pub(crate) path: PathBuf,
    pub(crate) version: u32,
    pub(crate) alignment: u64,
    pub(crate) tensor_data_offset: u64,
    pub(crate) file_len: u64,
    pub(crate) metadata: BTreeMap<String, GgufValue>,
    pub(crate) tensors: BTreeMap<String, GgufTensorInfo>,
}

impl GgufFile {
    pub(crate) fn open(path: &Path) -> Result<Self> {
        let file = File::open(path)
            .with_context(|| format!("failed to open GGUF source {}", path.display()))?;
        let file_len = file
            .metadata()
            .with_context(|| format!("failed to stat GGUF source {}", path.display()))?
            .len();
        let mut cursor = GgufCursor::new(file, file_len);
        let magic = cursor.read_u32()?;
        if magic != GGUF_MAGIC {
            bail!("{} is not a GGUF file", path.display());
        }
        let version = cursor.read_u32()?;
        if version != GGUF_VERSION {
            bail!(
                "unsupported GGUF version {version} in {}; FlashMoe requires v{GGUF_VERSION}",
                path.display()
            );
        }
        let tensor_count = cursor.read_bounded_count("tensor", MAX_DIRECTORY_ENTRIES)?;
        let metadata_count = cursor.read_bounded_count("metadata", MAX_DIRECTORY_ENTRIES)?;

        let mut metadata = BTreeMap::new();
        for _ in 0..metadata_count {
            let key = cursor.read_string()?;
            let value_type = GgufMetadataType::from_u32(cursor.read_u32()?)?;
            let value = cursor.read_value(value_type, 0)?;
            if metadata.insert(key.clone(), value).is_some() {
                bail!("GGUF metadata contains duplicate key {key}");
            }
        }

        let alignment = metadata
            .get("general.alignment")
            .and_then(GgufValue::as_u64_compat)
            .unwrap_or(32);
        if alignment == 0 || !alignment.is_power_of_two() {
            bail!("GGUF general.alignment must be a non-zero power of two, got {alignment}");
        }

        let mut tensor_directory = Vec::with_capacity(
            usize::try_from(tensor_count).context("GGUF tensor count does not fit memory")?,
        );
        for _ in 0..tensor_count {
            let name = cursor.read_string()?;
            let dimension_count = cursor.read_u32()?;
            if dimension_count == 0 || dimension_count > MAX_DIMS {
                bail!("GGUF tensor {name} has unsupported dimension count {dimension_count}");
            }
            let mut dimensions = Vec::with_capacity(dimension_count as usize);
            let mut elements = 1u64;
            for _ in 0..dimension_count {
                let dimension = cursor.read_u64()?;
                if dimension == 0 {
                    bail!("GGUF tensor {name} has a zero dimension");
                }
                elements = elements
                    .checked_mul(dimension)
                    .with_context(|| format!("GGUF tensor {name} element count overflow"))?;
                dimensions.push(dimension);
            }
            let tensor_type = GgufTensorType::from_u32(cursor.read_u32()?)
                .with_context(|| format!("GGUF tensor {name} uses an unsupported encoding"))?;
            let relative_offset = cursor.read_u64()?;
            if !relative_offset.is_multiple_of(alignment) {
                bail!(
                    "GGUF tensor {name} relative offset {relative_offset} is not aligned to {alignment}"
                );
            }
            let byte_len = tensor_type
                .byte_len(elements)
                .with_context(|| format!("failed to size GGUF tensor {name}"))?;
            tensor_directory.push((name, dimensions, tensor_type, relative_offset, byte_len));
        }

        let tensor_data_offset = align_up(cursor.position(), alignment)?;
        let mut tensors = BTreeMap::new();
        let mut ranges = Vec::with_capacity(tensor_directory.len());
        for (name, dimensions, tensor_type, relative_offset, byte_len) in tensor_directory {
            let absolute_offset = tensor_data_offset
                .checked_add(relative_offset)
                .with_context(|| format!("GGUF tensor {name} absolute offset overflow"))?;
            let end = absolute_offset
                .checked_add(byte_len)
                .with_context(|| format!("GGUF tensor {name} end offset overflow"))?;
            if end > file_len {
                bail!(
                    "GGUF tensor {name} range {absolute_offset}..{end} exceeds file length {file_len}"
                );
            }
            let info = GgufTensorInfo {
                name: name.clone(),
                dimensions,
                tensor_type,
                relative_offset,
                absolute_offset,
                byte_len,
            };
            if tensors.insert(name.clone(), info).is_some() {
                bail!("GGUF tensor directory contains duplicate tensor {name}");
            }
            ranges.push((absolute_offset, end, name));
        }
        ranges.sort_by_key(|(start, _, _)| *start);
        for pair in ranges.windows(2) {
            let (_, previous_end, previous_name) = &pair[0];
            let (next_start, _, next_name) = &pair[1];
            if next_start < previous_end {
                bail!(
                    "GGUF tensor payloads overlap: {previous_name} ends at {previous_end}, {next_name} begins at {next_start}"
                );
            }
        }

        Ok(Self {
            path: path.to_path_buf(),
            version,
            alignment,
            tensor_data_offset,
            file_len,
            metadata,
            tensors,
        })
    }

    pub(crate) fn required_metadata(&self, key: &str) -> Result<&GgufValue> {
        self.metadata
            .get(key)
            .with_context(|| format!("required GGUF metadata key is missing: {key}"))
    }

    pub(crate) fn required_tensor(&self, name: &str) -> Result<&GgufTensorInfo> {
        self.tensors
            .get(name)
            .with_context(|| format!("required GGUF tensor is missing: {name}"))
    }
}

struct GgufCursor {
    reader: BufReader<File>,
    position: u64,
    file_len: u64,
}

impl GgufCursor {
    fn new(file: File, file_len: u64) -> Self {
        Self {
            reader: BufReader::new(file),
            position: 0,
            file_len,
        }
    }

    fn position(&self) -> u64 {
        self.position
    }

    fn read_exact<const N: usize>(&mut self) -> Result<[u8; N]> {
        let end = self
            .position
            .checked_add(N as u64)
            .context("GGUF cursor position overflow")?;
        if end > self.file_len {
            bail!("truncated GGUF file at byte {}", self.position);
        }
        let mut bytes = [0u8; N];
        self.reader
            .read_exact(&mut bytes)
            .with_context(|| format!("failed to read GGUF at byte {}", self.position))?;
        self.position = end;
        Ok(bytes)
    }

    fn read_u8(&mut self) -> Result<u8> {
        Ok(self.read_exact::<1>()?[0])
    }

    fn read_i8(&mut self) -> Result<i8> {
        Ok(self.read_u8()? as i8)
    }

    fn read_u16(&mut self) -> Result<u16> {
        Ok(u16::from_le_bytes(self.read_exact()?))
    }

    fn read_i16(&mut self) -> Result<i16> {
        Ok(i16::from_le_bytes(self.read_exact()?))
    }

    fn read_u32(&mut self) -> Result<u32> {
        Ok(u32::from_le_bytes(self.read_exact()?))
    }

    fn read_i32(&mut self) -> Result<i32> {
        Ok(i32::from_le_bytes(self.read_exact()?))
    }

    fn read_u64(&mut self) -> Result<u64> {
        Ok(u64::from_le_bytes(self.read_exact()?))
    }

    fn read_i64(&mut self) -> Result<i64> {
        Ok(i64::from_le_bytes(self.read_exact()?))
    }

    fn read_f32(&mut self) -> Result<f32> {
        Ok(f32::from_le_bytes(self.read_exact()?))
    }

    fn read_f64(&mut self) -> Result<f64> {
        Ok(f64::from_le_bytes(self.read_exact()?))
    }

    fn read_bounded_count(&mut self, label: &str, maximum: u64) -> Result<u64> {
        let count = self.read_u64()?;
        if count > maximum {
            bail!("GGUF {label} count {count} exceeds safety limit {maximum}");
        }
        Ok(count)
    }

    fn read_string(&mut self) -> Result<String> {
        let len = self.read_u64()?;
        if len > MAX_STRING_BYTES {
            bail!("GGUF string length {len} exceeds safety limit {MAX_STRING_BYTES}");
        }
        let end = self
            .position
            .checked_add(len)
            .context("GGUF string end offset overflow")?;
        if end > self.file_len {
            bail!("truncated GGUF string at byte {}", self.position);
        }
        let len = usize::try_from(len).context("GGUF string length does not fit memory")?;
        let mut bytes = vec![0u8; len];
        self.reader
            .read_exact(&mut bytes)
            .with_context(|| format!("failed to read GGUF string at byte {}", self.position))?;
        self.position = end;
        String::from_utf8(bytes).context("GGUF string is not valid UTF-8")
    }

    fn read_value(&mut self, value_type: GgufMetadataType, depth: usize) -> Result<GgufValue> {
        if depth > MAX_ARRAY_DEPTH {
            bail!("GGUF metadata array nesting exceeds {MAX_ARRAY_DEPTH}");
        }
        Ok(match value_type {
            GgufMetadataType::Uint8 => GgufValue::Uint8(self.read_u8()?),
            GgufMetadataType::Int8 => GgufValue::Int8(self.read_i8()?),
            GgufMetadataType::Uint16 => GgufValue::Uint16(self.read_u16()?),
            GgufMetadataType::Int16 => GgufValue::Int16(self.read_i16()?),
            GgufMetadataType::Uint32 => GgufValue::Uint32(self.read_u32()?),
            GgufMetadataType::Int32 => GgufValue::Int32(self.read_i32()?),
            GgufMetadataType::Float32 => GgufValue::Float32(self.read_f32()?),
            GgufMetadataType::Bool => match self.read_u8()? {
                0 => GgufValue::Bool(false),
                1 => GgufValue::Bool(true),
                value => bail!("GGUF boolean has invalid byte value {value}"),
            },
            GgufMetadataType::String => GgufValue::String(self.read_string()?),
            GgufMetadataType::Array => {
                let element_type = GgufMetadataType::from_u32(self.read_u32()?)?;
                let len = self.read_bounded_count("metadata array item", MAX_ARRAY_ITEMS)?;
                let mut values = Vec::with_capacity(
                    usize::try_from(len)
                        .context("GGUF metadata array length does not fit memory")?,
                );
                for _ in 0..len {
                    values.push(self.read_value(element_type, depth + 1)?);
                }
                GgufValue::Array {
                    element_type,
                    values,
                }
            }
            GgufMetadataType::Uint64 => GgufValue::Uint64(self.read_u64()?),
            GgufMetadataType::Int64 => GgufValue::Int64(self.read_i64()?),
            GgufMetadataType::Float64 => GgufValue::Float64(self.read_f64()?),
        })
    }
}

fn align_up(value: u64, alignment: u64) -> Result<u64> {
    let remainder = value % alignment;
    if remainder == 0 {
        return Ok(value);
    }
    value
        .checked_add(alignment - remainder)
        .context("GGUF tensor data alignment overflow")
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    fn push_u32(bytes: &mut Vec<u8>, value: u32) {
        bytes.extend_from_slice(&value.to_le_bytes());
    }

    fn push_u64(bytes: &mut Vec<u8>, value: u64) {
        bytes.extend_from_slice(&value.to_le_bytes());
    }

    fn push_string(bytes: &mut Vec<u8>, value: &str) {
        push_u64(bytes, value.len() as u64);
        bytes.extend_from_slice(value.as_bytes());
    }

    fn tiny_gguf(relative_offset: u64) -> Vec<u8> {
        let mut bytes = Vec::new();
        push_u32(&mut bytes, GGUF_MAGIC);
        push_u32(&mut bytes, GGUF_VERSION);
        push_u64(&mut bytes, 1);
        push_u64(&mut bytes, 2);

        push_string(&mut bytes, "general.alignment");
        push_u32(&mut bytes, GgufMetadataType::Uint32 as u32);
        push_u32(&mut bytes, 32);

        push_string(&mut bytes, "test.values");
        push_u32(&mut bytes, GgufMetadataType::Array as u32);
        push_u32(&mut bytes, GgufMetadataType::Uint32 as u32);
        push_u64(&mut bytes, 2);
        push_u32(&mut bytes, 4);
        push_u32(&mut bytes, 8);

        push_string(&mut bytes, "tensor.weight");
        push_u32(&mut bytes, 2);
        push_u64(&mut bytes, 32);
        push_u64(&mut bytes, 2);
        push_u32(&mut bytes, GgufTensorType::Q8_0.id);
        push_u64(&mut bytes, relative_offset);

        while !bytes.len().is_multiple_of(32) {
            bytes.push(0);
        }
        bytes.resize(bytes.len() + relative_offset as usize + 68, 0);
        bytes
    }

    #[test]
    fn parses_bounded_metadata_and_tensor_directory() {
        let temp = tempfile::tempdir().unwrap();
        let path = temp.path().join("tiny.gguf");
        fs::write(&path, tiny_gguf(0)).unwrap();

        let gguf = GgufFile::open(&path).unwrap();
        assert_eq!(gguf.version, 3);
        assert_eq!(gguf.alignment, 32);
        assert_eq!(gguf.file_len, fs::metadata(&path).unwrap().len());
        assert_eq!(gguf.path, path);
        assert_eq!(
            gguf.required_metadata("test.values")
                .unwrap()
                .as_array()
                .unwrap()
                .1,
            &[GgufValue::Uint32(4), GgufValue::Uint32(8)]
        );
        let tensor = gguf.required_tensor("tensor.weight").unwrap();
        assert_eq!(tensor.tensor_type, GgufTensorType::Q8_0);
        assert_eq!(tensor.dimensions, [32, 2]);
        assert_eq!(tensor.byte_len, 68);
        assert_eq!(tensor.absolute_offset, gguf.tensor_data_offset);
    }

    #[test]
    fn rejects_unaligned_and_out_of_bounds_tensor_ranges() {
        let temp = tempfile::tempdir().unwrap();
        let unaligned = temp.path().join("unaligned.gguf");
        fs::write(&unaligned, tiny_gguf(1)).unwrap();
        assert!(
            GgufFile::open(&unaligned)
                .unwrap_err()
                .to_string()
                .contains("not aligned")
        );

        let truncated = temp.path().join("truncated.gguf");
        let mut bytes = tiny_gguf(0);
        bytes.truncate(bytes.len() - 1);
        fs::write(&truncated, bytes).unwrap();
        assert!(
            GgufFile::open(&truncated)
                .unwrap_err()
                .to_string()
                .contains("exceeds file length")
        );
    }
}
