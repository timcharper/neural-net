use ndarray_rand::{RandomExt, rand_distr::Uniform};

use ndarray::Array2;

use crate::serializable_model::SerializableModel;

pub struct InferrableModel {
  pub w1: Array2<f32>,
  pub b1: Array2<f32>,
  pub w2: Array2<f32>,
  pub b2: Array2<f32>,
}

impl InferrableModel {
  pub fn new() -> Self {
    return InferrableModel {
      // --- Init weights ---
      // 128 hidden layers
      w1: Array2::<f32>::random((128, 784), Uniform::new(-0.5, 0.5)),
      b1: Array2::<f32>::zeros((128, 1)),
      // output layer (10 neurons, 10 digits)
      w2: Array2::<f32>::random((10, 128), Uniform::new(-0.5, 0.5)),
      b2: Array2::<f32>::zeros((10, 1)),
    };
  }

  pub fn from_serializable_model(model: &SerializableModel) -> Self {
    let w1 =
      Array2::from_shape_vec((model.w1_shape.0, model.w1_shape.1), model.w1.clone()).unwrap();
    let b1 =
      Array2::from_shape_vec((model.b1_shape.0, model.b1_shape.1), model.b1.clone()).unwrap();
    let w2 =
      Array2::from_shape_vec((model.w2_shape.0, model.w2_shape.1), model.w2.clone()).unwrap();
    let b2 =
      Array2::from_shape_vec((model.b2_shape.0, model.b2_shape.1), model.b2.clone()).unwrap();

    InferrableModel { w1, b1, w2, b2 }
  }

  pub fn to_serializable_model(&self) -> SerializableModel {
    let (w1_r, w1_c) = self.w1.dim();
    let (b1_r, b1_c) = self.b1.dim();
    let (w2_r, w2_c) = self.w2.dim();
    let (b2_r, b2_c) = self.b2.dim();

    SerializableModel {
      w1: self.w1.iter().cloned().collect(),
      w1_shape: (w1_r, w1_c),
      b1: self.b1.iter().cloned().collect(),
      b1_shape: (b1_r, b1_c),
      w2: self.w2.iter().cloned().collect(),
      w2_shape: (w2_r, w2_c),
      b2: self.b2.iter().cloned().collect(),
      b2_shape: (b2_r, b2_c),
    }
  }
}
