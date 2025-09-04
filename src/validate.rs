use indicatif::{ProgressBar, ProgressStyle};
use ndarray::{Array2, Axis};

use crate::inferrable_model::InferrableModel;
use crate::math::{sigmoid, softmax};
use crate::serializable_model::SerializableModel;
use mnist::MnistBuilder;

const TEST_SIZE: usize = 10_000; // whole test dataset

pub fn validate(model_path: &str) {
  let mnist = MnistBuilder::new()
    .label_format_digit()
    .test_set_length(TEST_SIZE as u32)
    .finalize();
  let _tst_lbl = mnist.tst_lbl.clone();
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

  let tst_img = Array2::from_shape_vec(
    (TEST_SIZE, 28 * 28),
    mnist
      .tst_img
      .clone()
      .into_iter()
      .map(|x| x as f32)
      .collect(),
  )
  .unwrap();
  let pb = ProgressBar::new(TEST_SIZE as u64);
  pb.set_style(
    ProgressStyle::with_template("[{elapsed_precise}] {bar:40.cyan/blue} {pos:>7}/{len:7} {msg}")
      .unwrap()
      .progress_chars("##-"),
  );
  let mut total_correct: i32 = 0;
  for (i, (image, &y)) in tst_img.outer_iter().zip(_tst_lbl.iter()).enumerate() {
    // 784x1
    let image = image.insert_axis(Axis(1));

    // multiply hidden layers
    let z1 = model.w1.dot(&image) + &model.b1;
    let a1 = sigmoid(&z1);
    let z2 = model.w2.dot(&a1) + &model.b2;
    let a2 = softmax(&z2);

    let max_probability = a2.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));

    let correct_probability = a2[[y as usize, 0]];
    let is_correct = correct_probability == max_probability;
    if is_correct {
      total_correct += 1;
    }
    pb.inc(1);
    pb.set_message(format!("Correct: {} / {}", total_correct, i + 1));
  }
  pb.finish();

  println!(
    "Validation accuracy: {:.2}%",
    (total_correct as f32 / TEST_SIZE as f32) * 100.0
  );
}
