use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize)]
pub struct SerializableModel {
  pub w1: Vec<f32>,
  pub w1_shape: (usize, usize),
  pub b1: Vec<f32>,
  pub b1_shape: (usize, usize),
  pub w2: Vec<f32>,
  pub w2_shape: (usize, usize),
  pub b2: Vec<f32>,
  pub b2_shape: (usize, usize),
}

use safetensors::SafeTensors;
use std::fs::File;
use std::io::Read;
use std::path::Path;

impl SerializableModel {
  pub fn load_from_safetensors<P: AsRef<Path>>(
    path: P,
  ) -> Result<Self, Box<dyn std::error::Error>> {
    // Read the safetensors file
    let mut file = File::open(path)?;
    let mut buffer = Vec::new();
    file.read_to_end(&mut buffer)?;

    // Parse the safetensors
    let tensors = SafeTensors::deserialize(&buffer)?;

    // Helper function to convert bytes to f32 vector
    let bytes_to_f32_vec = |data: &[u8]| -> Vec<f32> {
      data
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect()
    };

    // Extract w1
    let w1_view = tensors.tensor("w1")?;
    let w1_data = bytes_to_f32_vec(w1_view.data());
    let w1_shape = w1_view.shape();
    let w1_shape = (w1_shape[0], w1_shape[1]);

    // Extract b1
    let b1_view = tensors.tensor("b1")?;
    let b1_data = bytes_to_f32_vec(b1_view.data());
    let b1_shape = b1_view.shape();
    let b1_shape = (b1_shape[0], b1_shape[1]);

    // Extract w2
    let w2_view = tensors.tensor("w2")?;
    let w2_data = bytes_to_f32_vec(w2_view.data());
    let w2_shape = w2_view.shape();
    let w2_shape = (w2_shape[0], w2_shape[1]);

    // Extract b2
    let b2_view = tensors.tensor("b2")?;
    let b2_data = bytes_to_f32_vec(b2_view.data());
    let b2_shape = b2_view.shape();
    let b2_shape = (b2_shape[0], b2_shape[1]);

    Ok(SerializableModel {
      w1: w1_data,
      w1_shape,
      b1: b1_data,
      b1_shape,
      w2: w2_data,
      w2_shape,
      b2: b2_data,
      b2_shape,
    })
  }
}
