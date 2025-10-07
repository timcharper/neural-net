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
  surface: ImageSurface,
  on_drawing_updated_cb: Box<dyn Fn(Array2<u8>)>,
}

impl DrawingAreaUI {
  pub fn new(on_drawing_updated_cb: Box<dyn Fn(Array2<u8>)>) -> Rc<RefCell<Self>> {
    // Create an image surface to store the drawing
    let surface = ImageSurface::create(Format::ARgb32, CANVAS_SIZE, CANVAS_SIZE)
      .expect("Failed to create image surface");

    // Initialize surface with white background
    {
      let ctx = Context::new(&surface).expect("Failed to create context");
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
      drawing_area: drawing_area,
      surface: surface,
      on_drawing_updated_cb: on_drawing_updated_cb,
    }));

    // Set up the draw function to display the surface
    component.borrow().drawing_area.set_draw_func({
      let component_weak = Rc::downgrade(&component);
      move |_area, context, _width, _height| {
        if let Some(component_rc) = component_weak.upgrade() {
          let comp_ref = component_rc.borrow();
          context
            .set_source_surface(&*comp_ref.surface, 0.0, 0.0)
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
    Self::setup_mouse_events(&component);

    component
  }

  fn setup_mouse_events(component: &Rc<RefCell<DrawingAreaUI>>) {
    let click_gesture = GestureClick::new();
    let drag_gesture = GestureDrag::new();

    let component_weak = Rc::downgrade(component);

    // Handle mouse press - draw a circle
    click_gesture.connect_pressed({
      let component_weak = component_weak.clone();
      move |_gesture, _n_press, x, y| {
        if let Some(component_rc) = component_weak.upgrade() {
          let mut comp_ref = component_rc.borrow_mut();
          {
            // draw the circle but let go of surface_rc asap.
            let ctx = Context::new(comp_ref.surface.clone()).expect("Failed to create context");
            ctx.set_source_rgb(0.0, 0.0, 0.0); // Black brush
            ctx.arc(x, y, BRUSH_SIZE / 2.0, 0.0, 2.0 * std::f64::consts::PI);
            ctx.fill().expect("Failed to draw circle");
            comp_ref.drawing_area.queue_draw();
          }

          // Call the registered callback with owned image data
          let image = comp_ref.get_image_data();
          (comp_ref.on_drawing_updated_cb)(image);
        }
      }
    });

    // Handle mouse drag - draw circles while dragging
    drag_gesture.connect_drag_update({
      let component_weak = component_weak.clone();
      move |gesture, x, y| {
        if let Some(component_rc) = component_weak.upgrade() {
          let mut comp_ref = component_rc.borrow_mut();
          if let Some((start_x, start_y)) = gesture.start_point() {
            let current_x = start_x + x;
            let current_y = start_y + y;

            {
              let ctx = Context::new(comp_ref.surface.clone()).expect("Failed to create context");
              ctx.set_source_rgb(0.0, 0.0, 0.0); // Black brush
              ctx.arc(
                current_x,
                current_y,
                BRUSH_SIZE / 2.0,
                0.0,
                2.0 * std::f64::consts::PI,
              );
              ctx.fill().expect("Failed to draw circle");
            }
            comp_ref.drawing_area.queue_draw();

            // Call the registered callback with owned image data
            let image = comp_ref.get_image_data();
            (comp_ref.on_drawing_updated_cb)(image);
          }
        }
      }
    });

    // Add gesture controllers to the drawing area
    {
      let comp_ref = component.borrow();
      comp_ref.drawing_area.add_controller(click_gesture);
      comp_ref.drawing_area.add_controller(drag_gesture);
    }
  }

  pub fn get_drawing_area(&self) -> &DrawingArea {
    &self.drawing_area
  }

  pub fn clear(&self) {
    // Clear the surface by painting it white
    let ctx = Context::new(&self.surface).expect("Failed to create context");
    ctx.set_source_rgb(1.0, 1.0, 1.0);
    ctx.paint().expect("Failed to clear surface");
    self.drawing_area.queue_draw();
  }

  /// Convert the drawn image to a 28x28 ndarray by extracting pixel data from the surface
  pub fn get_image_data(&mut self) -> Array2<u8> {
    let data = self.surface.data().expect("Failed to get surface data");

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

        downsampled[[i, j]] = if count > 0 {
          //(sum / count as f32 * 255.0) as u8
          sum / count as f32
        } else {
          0.0
        };
      }
    }

      // Center by center of mass (MNIST preprocessing)
    let mut com_row = 0.0;
    let mut com_col = 0.0;
    let mut total_mass = 0.0;

    for i in 0..28 {
      for j in 0..28 {
        let mass = downsampled[[i, j]];
        com_row += i as f32 * mass;
        com_col += j as f32 * mass;
        total_mass += mass;
      }
    }

    if total_mass > 0.0 {
      com_row /= total_mass;
      com_col /= total_mass;

      // Shift to center at (13.5, 13.5)
      let shift_row = 13.5 - com_row;
      let shift_col = 13.5 - com_col;

      let mut centered = Array2::<f32>::zeros((28, 28));
      for i in 0..28 {
        for j in 0..28 {
          let src_row = i as f32 - shift_row;
          let src_col = j as f32 - shift_col;

          // Bilinear interpolation
          if src_row >= 0.0 && src_row < 27.0 && src_col >= 0.0 && src_col < 27.0 {
            let r0 = src_row.floor() as usize;
            let c0 = src_col.floor() as usize;
            let r1 = (r0 + 1).min(27);
            let c1 = (c0 + 1).min(27);

            let dr = src_row - r0 as f32;
            let dc = src_col - c0 as f32;

            centered[[i, j]] =
              downsampled[[r0, c0]] * (1.0 - dr) * (1.0 - dc) +
              downsampled[[r0, c1]] * (1.0 - dr) * dc +
              downsampled[[r1, c0]] * dr * (1.0 - dc) +
              downsampled[[r1, c1]] * dr * dc;
          }
        }
      }

      centered.mapv(|v| (v * 255.0).min(255.0) as u8)
    } else {
      downsampled.mapv(|v| (v * 255.0) as u8)
    }
  }
}
