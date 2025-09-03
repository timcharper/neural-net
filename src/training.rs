use crate::inferrable_model::InferrableModel;
use crate::serialization::save_safetensors;
use crate::stats::{RollingMean, TrainingStats};
use indicatif::{MultiProgress, ProgressBar, ProgressStyle};
use mnist::MnistBuilder;
use ndarray::{Array2, Axis};

use crate::math::{sigmoid, sigmoid_derivative, softmax};

const LR: f32 = 0.001;
pub const TRAINING_SIZE: usize = 60_000 /* whole dataset */;
const EPOCHS: usize = 15;
const ROLLING_MEAN_SIZE: usize = 1000;

pub fn run_train(model_path: &str) {
  let mnist = MnistBuilder::new()
    .label_format_digit()
    .training_set_length(TRAINING_SIZE as u32)
    .finalize();

  let trn_img = Array2::from_shape_vec(
    (TRAINING_SIZE, 28 * 28),
    mnist
      .trn_img
      .clone()
      .into_iter()
      .map(|x| x as f32)
      .collect(),
  )
  .unwrap();

  let trn_lbl = mnist.trn_lbl.clone();

  // --- Init weights ---
  // 128 hidden layers
  // let mut w1: ndarray::ArrayBase<ndarray::OwnedRepr<f32>, ndarray::Dim<[usize; 2]>> =
  //   Array2::<f32>::random((128, 784), Uniform::new(-0.5, 0.5));
  // -  let mut b1 = Array2::<f32>::zeros((128, 1));
  // -
  // -  // output layer (10 neurons, 10 digits)
  // -  let mut w2 = Array2::<f32>::random((10, 128), Uniform::new(-0.5, 0.5));
  // -  let mut b2 = Array2::<f32>::zeros((10, 1));

  let mut model = InferrableModel::new();

  // training loop

  // MultiProgress will hold one progress bar per epoch
  let m = MultiProgress::new();
  let sty =
    ProgressStyle::with_template("[{elapsed_precise}] {bar:40.cyan/blue} {pos:>7}/{len:7} {msg}")
      .unwrap()
      .progress_chars("##-");

  for epoch in 0..EPOCHS {
    println!("Epoch: {}", epoch);
    let rolling_entropy_loss = &mut RollingMean::new(ROLLING_MEAN_SIZE);

    let pb = m.add(ProgressBar::new(TRAINING_SIZE as u64));
    pb.set_style(sty.clone());
    pb.set_prefix(format!("Epoch {}/{}", epoch + 1, EPOCHS));
    pb.set_message(format!("loss={:.4}", rolling_entropy_loss.mean()));

    let stats = &mut TrainingStats::new();

    for (i, (image, &y)) in trn_img.outer_iter().zip(trn_lbl.iter()).enumerate() {
      // Forward
      // 128x784 * 784x1 = 128x1 - hidden layer
      let image = image.insert_axis(Axis(1)); // make it 1x784 ?
      // println!("image shape {:?}", image.dim());
      let z1 = &model.w1.dot(&image) + &model.b1;
      // println!("w1 dot image dim {:?}", &w1.dot(&image).dim());
      // println!("image dim {:?}", image.dim());
      // println!("w1 dim {:?}", w1.dim());
      // println!("b1 dim {:?}", b1.dim());
      // clamp / normalize 128x1
      let a1 = sigmoid(&z1);

      // 10*128 * 128x1 = 10x1; 10x1 + 10x1
      let z2 = &model.w2.dot(&a1) + &model.b2;

      // redistribute so all values sum up to 1
      let a2 = softmax(&z2);

      let correct_probability = a2[[y as usize, 0]];
      let entropy_loss = -((correct_probability + 1e-10).ln()); // add small value to avoid log(0)
      let max_probability = a2.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
      let is_correct = correct_probability == max_probability;

      // One-hot target (the correct probabilities)
      let mut y_vec = Array2::<f32>::zeros((10, 1));
      y_vec[[y as usize, 0]] = 1.0;

      // subtract the guesses by the actual answer; the more correct, the lower the values will be.
      // 10x1 with values -0.5..0.5
      let dz2 = &a2 - &y_vec;

      // 10x1 * 1x128 (transposed a1) = 10x128
      // how much did each of these hidden layers contribute to each neurons wrong guess?
      let dw2 = dz2.dot(&a1.t());
      // 10x128
      let db2 = dz2.clone();

      // Error

      // 128x10 * 10x1 = 128x1 * 128x1 (how saturated are these hidden neurons? If saturated they've learned a feature and dz1_x will be close to 0.)
      // derivative should be taken on the activated values (a1)
      let dz1 = model.w2.t().dot(&dz2) * sigmoid_derivative(&a1);

      // "flash" the image on to the hidden layer by multiplying it
      // 128x1 * 1x784 = 128x784 = one "flashed" (multiplied) hidden layer neuron. We'll back-propagate by this the error amount, except if the neuron is saturated.
      let dw1 = dz1.dot(&image.t());
      let db1 = dz1.clone();

      model.w2 = &model.w2 - &(LR * &dw2);
      model.b2 = &model.b2 - &(LR * &db2);
      model.w1 = &model.w1 - &(LR * &dw1);
      model.b1 = &model.b1 - &(LR * &db1);

      rolling_entropy_loss.push(entropy_loss);
      stats.update(entropy_loss, is_correct);

      // update the per-epoch progress bar: show rolling mean and iteration
      pb.inc(1);
      pb.set_message(format!(
        "loss={:.4} it={}/{}",
        rolling_entropy_loss.mean(),
        i + 1,
        TRAINING_SIZE
      ));
    }
    println!("Training stats: {:?}", stats.to_string());

    pb.finish_with_message("done");
  }

  println!("\nTraining finished, saving model to {}", model_path);

  let model = model.to_serializable_model();
  // Check for non-finite values. serde_json serializes NaN/Inf to null,
  // which is why you were seeing nulls in the JSON file.
  let check = |v: &Vec<f32>, name: &str| {
    let non_finite = v.iter().filter(|x| !x.is_finite()).count();
    if non_finite > 0 {
      eprintln!(
        "Model contains {} non-finite values in {}",
        non_finite, name
      );
      // print a few samples for diagnosis
      let sample: Vec<_> = v.iter().take(10).cloned().collect();
      eprintln!("{} sample: {:?}", name, sample);
    }
    non_finite
  };

  let nf_w1 = check(&model.w1, "w1");
  let nf_b1 = check(&model.b1, "b1");
  let nf_w2 = check(&model.w2, "w2");
  let nf_b2 = check(&model.b2, "b2");

  if nf_w1 + nf_b1 + nf_w2 + nf_b2 > 0 {
    eprintln!(
      "Aborting save because model contains non-finite values (NaN/Inf). Try using a smaller learning rate or stabilizing activations."
    );
    return;
  }

  // Save tensors to safetensors file
  let arr_w1 = Array2::from_shape_vec((model.w1_shape.0, model.w1_shape.1), model.w1.clone())
    .expect("w1 shape mismatch");
  let arr_b1 = Array2::from_shape_vec((model.b1_shape.0, model.b1_shape.1), model.b1.clone())
    .expect("b1 shape mismatch");
  let arr_w2 = Array2::from_shape_vec((model.w2_shape.0, model.w2_shape.1), model.w2.clone())
    .expect("w2 shape mismatch");
  let arr_b2 = Array2::from_shape_vec((model.b2_shape.0, model.b2_shape.1), model.b2.clone())
    .expect("b2 shape mismatch");

  match save_safetensors(
    model_path,
    &[
      ("w1", &arr_w1),
      ("b1", &arr_b1),
      ("w2", &arr_w2),
      ("b2", &arr_b2),
    ],
  ) {
    Ok(()) => println!("Model saved to {} as safetensors", model_path),
    Err(e) => eprintln!("Failed to write safetensors: {:?}", e),
  }
}
