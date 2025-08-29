use ndarray::Array1;
use ndarray::Array2;

// sigmoid "clamps" values (in a fairly scaled way) to 0..1
pub fn sigmoid(x: &Array2<f32>) -> Array2<f32> {
  return x.mapv(|v| 1.0 / (1.0 + (-v).exp()));
}

pub fn softmax(z: &Array2<f32>) -> Array2<f32> {
  // Stable softmax: subtract max to avoid large exponents
  let max = z.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
  let exps = z.mapv(|v| (v - max).exp());
  let sum = exps.sum();
  exps.mapv(|e| e / sum)
}

pub fn sigmoid_derivative(a: &Array2<f32>) -> Array2<f32> {
  a.mapv(|v| v * (1.0 - v))
}

/// Flatten a 2-dimensional array into a 1-dimensional array (row-major order).
///
/// Input: `Array2<f32>` with shape (r, c).
/// Output: `Array1<f32>` with length r*c, iterating rows then columns.
pub fn flatten_2d_to_1d(a: &Array2<f32>) -> Array1<f32> {
  // Collect into a Vec in row-major order, then convert to Array1
  let v: Vec<f32> = a.iter().cloned().collect();
  Array1::from(v)
}
