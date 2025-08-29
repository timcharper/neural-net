use clap::{Parser, Subcommand};
use mnist::*;
use ndarray::{Array2, Axis};
use ndarray_rand::{RandomExt, rand_distr::Uniform};
use serde::{Deserialize, Serialize};
mod serialization;
use serialization::save_safetensors;

// sigmoid "clamps" values (in a fairly scaled way) to 0..1
fn sigmoid(x: &Array2<f32>) -> Array2<f32> {
    return x.mapv(|v| 1.0 / (1.0 + (-v).exp()));
}

fn softmax(z: &Array2<f32>) -> Array2<f32> {
    // Stable softmax: subtract max to avoid large exponents
    let max = z.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exps = z.mapv(|v| (v - max).exp());
    let sum = exps.sum();
    exps.mapv(|e| e / sum)
}

fn sigmoid_derivative(a: &Array2<f32>) -> Array2<f32> {
    a.mapv(|v| v * (1.0 - v))
}

const LR: f32 = 0.01;
const TRAINING_SIZE: usize = 20_000;
const EPOCHS: usize = 1;

fn main() {
    // CLI
    let cli = Cli::parse();

    match &cli.command {
        Commands::Train { out } => {
            let mnist = MnistBuilder::new()
                .label_format_digit()
                .training_set_length(TRAINING_SIZE as u32)
                .test_set_length(10_000)
                .finalize();

            run_train(&mnist, out);
        }

        Commands::Infer { .. } => {
            // Not implemented yet
            unimplemented!("infer is not implemented yet");
        }
    }
}

#[derive(Parser)]
#[command(author, version, about, long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Train a model
    Train {
        /// Output file to write model weights to
        #[arg(short, long, default_value = "model.safetensors")]
        out: String,
    },

    /// Run inference (not implemented yet)
    Infer {
        /// Input image or model path (placeholder)
        #[arg(short, long)]
        input: Option<String>,
    },
}

#[derive(Serialize, Deserialize)]
struct SerializableModel {
    w1: Vec<f32>,
    w1_shape: (usize, usize),
    b1: Vec<f32>,
    b1_shape: (usize, usize),
    w2: Vec<f32>,
    w2_shape: (usize, usize),
    b2: Vec<f32>,
    b2_shape: (usize, usize),
}

fn run_train(mnist: &Mnist, out: &str) {
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
    let mut w1 = Array2::<f32>::random((128, 784), Uniform::new(-0.5, 0.5));
    let mut b1 = Array2::<f32>::zeros((128, 1));

    // output layer (10 neurons, 10 digits)
    let mut w2 = Array2::<f32>::random((10, 128), Uniform::new(-0.5, 0.5));
    let mut b2 = Array2::<f32>::zeros((10, 1));

    // training loop

    for epoch in 0..EPOCHS {
        println!("Epoch {}", epoch);
        for (image, &y) in trn_img.outer_iter().zip(trn_lbl.iter()) {
            // Forward
            // 128x784 * 784x1 = 128x1 - hidden layer
            let image = image.insert_axis(Axis(1)); // make it 1x784 ?
            // println!("image shape {:?}", image.dim());
            let z1 = &w1.dot(&image) + &b1;
            // println!("w1 dot image dim {:?}", &w1.dot(&image).dim());
            // println!("image dim {:?}", image.dim());
            // println!("w1 dim {:?}", w1.dim());
            // println!("b1 dim {:?}", b1.dim());
            // clamp / normalize 128x1
            let a1 = sigmoid(&z1);

            // 10*128 * 128x1 = 10x1; 10x1 + 10x1
            let z2 = &w2.dot(&a1) + &b2;

            // redistribute so all values sum up to 1
            let a2 = softmax(&z2);

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
            let dz1 = w2.t().dot(&dz2) * sigmoid_derivative(&a1);

            // "flash" the image on to the hidden layer by multiplying it
            // 128x1 * 1x784 = 128x784 = one "flashed" (multiplied) hidden layer neuron. We'll back-propagate by this the error amount, except if the neuron is saturated.
            let dw1 = dz1.dot(&image.t());
            let db1 = dz1.clone();

            // Update; comments assume we are confidently wrong
            w2 = &w2 - &(LR * &dw2);
            b2 = &b2 - &(LR * &db2);
            w1 = &w1 - &(LR * &dw1);
            b1 = &b1 - &(LR * &db1);
            print!(".")
        }
    }

    println!("\nTraining finished, saving model to {}", out);

    let (w1_r, w1_c) = w1.dim();
    let (b1_r, b1_c) = b1.dim();
    let (w2_r, w2_c) = w2.dim();
    let (b2_r, b2_c) = b2.dim();

    let model = SerializableModel {
        w1: w1.iter().cloned().collect(),
        w1_shape: (w1_r, w1_c),
        b1: b1.iter().cloned().collect(),
        b1_shape: (b1_r, b1_c),
        w2: w2.iter().cloned().collect(),
        w2_shape: (w2_r, w2_c),
        b2: b2.iter().cloned().collect(),
        b2_shape: (b2_r, b2_c),
    };

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
        out,
        &[
            ("w1", &arr_w1),
            ("b1", &arr_b1),
            ("w2", &arr_w2),
            ("b2", &arr_b2),
        ],
    ) {
        Ok(()) => println!("Model saved to {} as safetensors", out),
        Err(e) => eprintln!("Failed to write safetensors: {:?}", e),
    }
}
