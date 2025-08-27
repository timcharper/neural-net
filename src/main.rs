use mnist::*;
use ndarray::Array2;
use ndarray_rand::{RandomExt, rand_distr::Uniform};

// sigmoid "clamps" values (in a fairly scaled way) to 0..1
fn sigmoid(x: &Array2<f32>) -> Array2<f32> {
    return x.mapv(|v| 1.0 / (1.0 + (-v).exp()));
}

fn main() {
    let mnist = MnistBuilder::new()
        .label_format_digit()
        .training_set_length(50_000)
        .test_set_length(10_000)
        .finalize();

    let trn_img = Array2::from_shape_vec(
        (50_000, 28 * 28),
        mnist.trn_img.into_iter().map(|x| x as f32).collect(),
    )
    .unwrap();

    let trn_lbl = mnist.trn_lbl;

    // --- Init weights ---
    // 128 hidden layers
    let mut w1 = Array2::<f32>::random((128, 784), Uniform::new(-0.5, 0.5));
    let mut b1 = Array2::<f32>::zeros((128, 1));

    // output layer (10 neurons, 10 digits)
    let mut w2 = Array2::<f32>::random((10, 128), Uniform::new(-0.5, 0.5));
    let mut b2 = Array2::<f32>::zeros((10, 1));

    // training loop

    for epoch in 0..5 {
        for (image, &y) in trn_img.outer_iter().zip(trn_lbl.iter()) {
            // lol
            // Forward
            let z1 = &w1.dot(&image) + &b1;
            let a1 = sigmoid(&z1);
            let z2 = &w2.dot(&a1) + &b2;
            let a2 = sigmoid(&z2); // for simplicity, not softmax yet
        }
    }

    println!("Hello, world!");
}
