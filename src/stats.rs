use std::collections::VecDeque;

struct RollingMean {
  buf: VecDeque<f32>,
  sum: f32,
  cap: usize,
}

impl RollingMean {
  fn new(cap: usize) -> Self {
    Self {
      buf: VecDeque::with_capacity(cap),
      sum: 0.0,
      cap,
    }
  }

  fn push(&mut self, x: f32) -> () {
    self.buf.push_back(x);
    self.sum += x;

    if self.buf.len() > self.cap {
      let old = self.buf.pop_front().unwrap();
      self.sum -= old;
    }
  }

  fn mean(&self) -> f32 {
    self.sum / self.buf.len() as f32
  }
}
