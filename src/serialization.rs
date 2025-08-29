use ndarray::Array2;
use safetensors::{serialize_to_file, Dtype, SafeTensorError, View};
use std::borrow::Cow;
use std::path::Path;

/// Convert a Vec<f32> into little-endian bytes
fn f32_vec_to_le_bytes(v: &[f32]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(v.len() * 4);
    for &f in v.iter() {
        bytes.extend_from_slice(&f.to_bits().to_le_bytes());
    }
    bytes
}

struct OwnedTensor {
    dtype: Dtype,
    shape: Vec<usize>,
    data: Vec<u8>,
}

impl View for OwnedTensor {
    fn dtype(&self) -> Dtype {
        self.dtype
    }

    fn shape(&self) -> &[usize] {
        &self.shape
    }

    fn data(&self) -> Cow<'_, [u8]> {
        (&self.data[..]).into()
    }

    fn data_len(&self) -> usize {
        self.data.len()
    }
}

/// Save a list of named Array2<f32> tensors into a safetensors file.
/// `tensors` is a slice of (name, reference to array).
pub fn save_safetensors<P: AsRef<Path>>(
    path: P,
    tensors: &[(&str, &Array2<f32>)],
) -> Result<(), SafeTensorError> {
    let mut owned: Vec<(&str, OwnedTensor)> = Vec::with_capacity(tensors.len());

    for (name, arr) in tensors.iter() {
        let (r, c) = arr.dim();
        let flat: Vec<f32> = arr.iter().cloned().collect();
        let data = f32_vec_to_le_bytes(&flat);

        let ot = OwnedTensor {
            dtype: Dtype::F32,
            shape: vec![r, c],
            data,
        };

        owned.push((name.as_ref(), ot));
    }

    serialize_to_file(owned.into_iter(), None, path.as_ref())
}
