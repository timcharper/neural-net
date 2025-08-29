use cairo::{Context, Format, ImageSurface};
use gtk4::prelude::*;
use gtk4::{DrawingArea, GestureClick, GestureDrag};
use ndarray::Array2;
use std::cell::RefCell;
use std::rc::Rc;

const BRUSH_SIZE: f64 = 20.0;
const CANVAS_SIZE: i32 = 280;

pub struct DrawingAreaUI {
  drawing_area: DrawingArea,
  surface: Rc<RefCell<ImageSurface>>,
}

impl DrawingAreaUI {
  pub fn new() -> Rc<RefCell<Self>> {
    // Create an image surface to store the drawing
    let surface = Rc::new(RefCell::new(
      ImageSurface::create(Format::ARgb32, CANVAS_SIZE, CANVAS_SIZE)
        .expect("Failed to create image surface"),
    ));

    // Initialize surface with white background
    {
      let surface_ref = surface.borrow();
      let ctx = Context::new(&*surface_ref).expect("Failed to create context");
      ctx.set_source_rgb(1.0, 1.0, 1.0);
      ctx.paint().expect("Failed to paint initial background");
    }

    // Create a drawing area for the canvas
    let drawing_area = DrawingArea::builder()
      .width_request(CANVAS_SIZE)
      .height_request(CANVAS_SIZE)
      .can_focus(true)
      .build();

    let component = Rc::new(RefCell::new(DrawingAreaUI {
      drawing_area: drawing_area.clone(),
      surface: surface.clone(),
    }));

    // Set up the draw function to display the surface
    drawing_area.set_draw_func({
      let surface_weak = Rc::downgrade(&surface);
      move |_area, context, _width, _height| {
        if let Some(surface_rc) = surface_weak.upgrade() {
          let surface_ref = surface_rc.borrow();
          context
            .set_source_surface(&*surface_ref, 0.0, 0.0)
            .expect("Failed to set surface as source");
          context.paint().expect("Failed to paint surface");

          // Draw a simple border
          context.set_source_rgb(0.0, 0.0, 0.0);
          context.set_line_width(2.0);
          context.rectangle(1.0, 1.0, (CANVAS_SIZE - 2) as f64, (CANVAS_SIZE - 2) as f64);
          context.stroke().expect("Failed to draw border");
        }
      }
    });

    // Set up mouse event handling
    Self::setup_mouse_events(&drawing_area, &surface);

    component
  }

  fn setup_mouse_events(drawing_area: &DrawingArea, surface: &Rc<RefCell<ImageSurface>>) {
    let click_gesture = GestureClick::new();
    let drag_gesture = GestureDrag::new();

    let drawing_area_for_redraw = drawing_area.clone();
    let drawing_area_for_drag = drawing_area.clone();

    // Handle mouse press - draw a circle
    click_gesture.connect_pressed({
      let surface_weak = Rc::downgrade(surface);
      move |_gesture, _n_press, x, y| {
        if let Some(surface_rc) = surface_weak.upgrade() {
          let surface_ref = surface_rc.borrow();
          let ctx = Context::new(&*surface_ref).expect("Failed to create context");
          ctx.set_source_rgb(0.0, 0.0, 0.0); // Black brush
          ctx.arc(x, y, BRUSH_SIZE / 2.0, 0.0, 2.0 * std::f64::consts::PI);
          ctx.fill().expect("Failed to draw circle");
          drawing_area_for_redraw.queue_draw();
        }
      }
    });

    // Handle mouse drag - draw circles while dragging
    drag_gesture.connect_drag_update({
      let surface_weak = Rc::downgrade(surface);
      move |gesture, x, y| {
        if let Some(surface_rc) = surface_weak.upgrade() {
          if let Some((start_x, start_y)) = gesture.start_point() {
            let current_x = start_x + x;
            let current_y = start_y + y;

            let surface_ref = surface_rc.borrow();
            let ctx = Context::new(&*surface_ref).expect("Failed to create context");
            ctx.set_source_rgb(0.0, 0.0, 0.0); // Black brush
            ctx.arc(
              current_x,
              current_y,
              BRUSH_SIZE / 2.0,
              0.0,
              2.0 * std::f64::consts::PI,
            );
            ctx.fill().expect("Failed to draw circle");
            drawing_area_for_drag.queue_draw();
          }
        }
      }
    });

    // Add gesture controllers to the drawing area
    drawing_area.add_controller(click_gesture);
    drawing_area.add_controller(drag_gesture);
  }

  pub fn get_drawing_area(&self) -> &DrawingArea {
    &self.drawing_area
  }

  pub fn clear(&self) {
    // Clear the surface by painting it white
    let surface_ref = self.surface.borrow();
    let ctx = Context::new(&*surface_ref).expect("Failed to create context");
    ctx.set_source_rgb(1.0, 1.0, 1.0);
    ctx.paint().expect("Failed to clear surface");
    self.drawing_area.queue_draw();
  }

  /// Convert the drawn image to a 28x28 ndarray by extracting pixel data from the surface
  pub fn get_image_data(&self) -> Array2<f32> {
    let mut surface_ref = self.surface.borrow_mut();
    let data = surface_ref.data().expect("Failed to get surface data");

    // Convert ARGB32 data to grayscale and downsample
    let mut downsampled = Array2::<f32>::zeros((28, 28));

    for i in 0..28 {
      for j in 0..28 {
        let mut sum = 0.0;
        let mut count = 0;

        // Average over a 10x10 block
        for di in 0..10 {
          for dj in 0..10 {
            let y = i * 10 + di;
            let x = j * 10 + dj;
            if y < 280 && x < 280 {
              // ARGB32 format: each pixel is 4 bytes (B, G, R, A)
              let pixel_index = ((y * 280 + x) * 4) as usize;
              if pixel_index + 3 < data.len() {
                let b = data[pixel_index] as f32 / 255.0;
                let g = data[pixel_index + 1] as f32 / 255.0;
                let r = data[pixel_index + 2] as f32 / 255.0;

                // Convert to grayscale (inverted - black pixels = 1, white = 0)
                let gray = 1.0 - (0.299 * r + 0.587 * g + 0.114 * b);
                sum += gray;
                count += 1;
              }
            }
          }
        }

        downsampled[[i, j]] = if count > 0 { sum / count as f32 } else { 0.0 };
      }
    }

    downsampled
  }
}
