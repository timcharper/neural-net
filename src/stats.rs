use std::collections::VecDeque;

#[derive(Default, Debug)]
pub struct TrainingStats {
  pub total_loss: f32,
  pub total_correct: usize,
  pub total_samples: usize,
}

impl TrainingStats {
  pub fn new() -> Self {
    Self::default()
  }

  pub fn update(&mut self, loss: f32, correct: bool) -> () {
    self.total_loss += loss;
    self.total_samples += 1;
    if correct {
      self.total_correct += 1;
    }
  }

  pub fn to_string(&self) -> String {
    format!(
      "TrainingStats {{ mean loss: {}, accuracy: {}, samples: {} }}",
      self.total_loss / self.total_samples as f32,
      self.total_correct as f32 / self.total_samples as f32,
      self.total_samples
    )
  }
}

pub struct RollingMean {
  buf: VecDeque<f32>,
  sum: f32,
  cap: usize,
}

impl RollingMean {
  pub fn new(cap: usize) -> Self {
    Self {
      buf: VecDeque::with_capacity(cap),
      sum: 0.0,
      cap,
    }
  }

  pub fn push(&mut self, x: f32) -> () {
    self.buf.push_back(x);
    self.sum += x;

    if self.buf.len() > self.cap {
      let old = self.buf.pop_front().unwrap();
      self.sum -= old;
    }
  }

  pub fn mean(&self) -> f32 {
    self.sum / self.buf.len() as f32
  }
}
