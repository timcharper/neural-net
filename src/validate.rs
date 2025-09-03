use ndarray::Array2;

use crate::inferrable_model::InferrableModel;
use crate::serializable_model::SerializableModel;
use mnist::MnistBuilder;

const TEST_SIZE: usize = 10_000; // whole test dataset

pub fn validate(model_path: &str) {
  let mnist = MnistBuilder::new()
    .label_format_digit()
    .training_set_length(TRAINING_SIZE as u32)
    .finalize();
  // Load the neural network model
  let model = match SerializableModel::load_from_safetensors(model_path) {
    Ok(serialized_model) => {
      let model = InferrableModel::from_serializable_model(&serialized_model);
      println!("Successfully loaded model from: {}", model_path);
      println!(
        "Model dimensions: w1={:?}, b1={:?}, w2={:?}, b2={:?}",
        model.w1.dim(),
        model.b1.dim(),
        model.w2.dim(),
        model.b2.dim()
      );
      model
    }
    Err(e) => {
      panic!("Failed to load model from {}: {}", model_path, e);
    }
  };

  let vest_img = Array2::from_shape_vec(
    (TEST_SIZE, 28 * 28),
    mnist
      .tst_img
      .clone()
      .into_iter()
      .map(|x| x as f32)
      .collect(),
  )
  .unwrap();

  // TODO: Implement validation logic here.
  // You can use the `model` variable above to access the loaded model.
  println!("Validation logic not implemented yet.");
}
