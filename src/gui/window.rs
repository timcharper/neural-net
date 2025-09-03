use gio::ApplicationFlags;
use glib::clone::Downgrade;
use gtk4::prelude::*;
use gtk4::{Application, ApplicationWindow, Box, Button, Label, Orientation};
use ndarray::{Array, Array2, Axis, Ix2};
use std::rc::Rc;

use crate::inferrable_model::InferrableModel;
use crate::math::{flatten_2d_to_1d, sigmoid, softmax};
use crate::serializable_model::SerializableModel;

use super::drawing_area_ui::DrawingAreaUI;

fn run_inference(model: &InferrableModel, image_data: &Array2<u8>) -> Array<f32, Ix2> {
  let image = flatten_2d_to_1d(image_data)
    .mapv(|x| x as f32)
    .insert_axis(Axis(1));

  // Forward pass
  let z1 = &model.w1.dot(&image) + &model.b1;
  let a1 = sigmoid(&z1);
  let z2 = model.w2.dot(&a1) + &model.b2;
  softmax(&z2)
}

pub fn create_window(model_path: &str) {
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
      Rc::new(model)
    }
    Err(e) => {
      panic!("Failed to load model from {}: {}", model_path, e);
    }
  };

  // Create the GTK4 application
  let app = Application::builder()
    .application_id("com.example.neural-net")
    .flags(ApplicationFlags::FLAGS_NONE)
    .build();

  app.connect_activate(move |app| {
    // Clone model for use in the closure
    let model = model.clone();
    // Create the main window
    let window = ApplicationWindow::builder()
      .application(app)
      .title("Neural Network Visualizer")
      .default_width(500)
      .default_height(400)
      .resizable(true)
      .build();

    // Create the main layout - vertical box
    // Create the drawing area component will be created after the output label so the
    // callback can capture the label and the model.

    let main_vbox = Box::new(Orientation::Vertical, 10);
    main_vbox.set_margin_top(10);
    main_vbox.set_margin_bottom(10);
    main_vbox.set_margin_start(10);
    main_vbox.set_margin_end(10);

    let content_hbox = Box::new(Orientation::Horizontal, 20);

    let output_text = "Neural Network Output:

Model loaded successfully!
Draw a digit to see prediction...";
    let output_label = Label::new(Some(output_text));
    output_label.set_halign(gtk4::Align::Start);
    output_label.set_valign(gtk4::Align::Start);

    // Create the drawing area component with a required callback that performs
    // inference and updates the output label. The closure takes ownership of
    // clones of `model` and `output_label`.
    let drawing_area_ui = {
      let model_for_cb = model.clone();
      let output_label_for_cb = output_label.clone();
      DrawingAreaUI::new(std::boxed::Box::new(move |image_data_2d: Array2<u8>| {
        let a2 = run_inference(&model_for_cb, &image_data_2d);

        // Format predictions and update label on the main thread
        let mut s = String::new();
        s.push_str("Neural Network Output:\n\n");
        s.push_str("Predictions:\n");
        for (i, v) in a2.iter().enumerate() {
          s.push_str(&format!("{}: {:.3}\n", i, v));
        }
        output_label_for_cb.set_text(&s);
      }))
    };

    content_hbox.append(drawing_area_ui.borrow().get_drawing_area());

    content_hbox.append(&output_label);

    // Create horizontal box for buttons
    let button_hbox = Box::new(Orientation::Horizontal, 10);

    // Create clear button
    let clear_button = Button::with_label("Clear Canvas");

    // Set up clear button event handler
    let drawing_area_ui_clear = drawing_area_ui.clone();
    clear_button.connect_clicked(move |_| {
      drawing_area_ui_clear.borrow().clear();
    });

    // Create debug button
    let debug_button = Button::with_label("Debug");

    // Set up debug button event handler
    debug_button.connect_clicked({
      let drawing_area_ui_weak = drawing_area_ui.downgrade();
      let model = model.downgrade();
      move |_| {
        if let (Some(drawing_area_ui), Some(model)) =
          (drawing_area_ui_weak.upgrade(), model.upgrade())
        {
          let image_data_2d = drawing_area_ui.borrow_mut().get_image_data();

          for r in 0..28 {
            for c in 0..28 {
              let v = image_data_2d[[r, c]];
              print!("{:4} ", v);
            }
            println!();
          }

          let a2 = run_inference(&model, &image_data_2d);

          println!("Predictions: {:?}", a2)
        }
      }
    });

    // Add buttons to button box
    button_hbox.append(&clear_button);
    button_hbox.append(&debug_button);

    // Add components to main layout
    main_vbox.append(&content_hbox);
    main_vbox.append(&button_hbox);

    // Add the main layout to the window
    window.set_child(Some(&main_vbox));

    // Show the window
    window.present();
  });

  // Run the application
  app.run_with_args(&[] as &[&str]);
}
