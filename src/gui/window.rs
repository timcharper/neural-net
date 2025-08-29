use gio::ApplicationFlags;
use glib::clone::Downgrade;
use gtk4::prelude::*;
use gtk4::{Application, ApplicationWindow, Box, Button, Label, Orientation};
use ndarray::Axis;
use std::rc::Rc;

use crate::inferrable_model::InferrableModel;
use crate::math::{flatten_2d_to_1d, sigmoid, softmax};
use crate::serializable_model::SerializableModel;

use super::drawing_area_ui::DrawingAreaUI;

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

    // Create the drawing area component
    let drawing_area_ui = DrawingAreaUI::new();

    // Create the main layout - vertical box
    let main_vbox = Box::new(Orientation::Vertical, 10);
    main_vbox.set_margin_top(10);
    main_vbox.set_margin_bottom(10);
    main_vbox.set_margin_start(10);
    main_vbox.set_margin_end(10);

    // Create horizontal box for drawing area and neural net output
    let content_hbox = Box::new(Orientation::Horizontal, 20);

    // Add drawing area to the horizontal box
    {
      let drawing_area_borrowed = drawing_area_ui.borrow();
      let drawing_area = drawing_area_borrowed.get_drawing_area();
      content_hbox.append(drawing_area);
    }

    // Create placeholder for neural network output
    let output_text = "Neural Network Output:

Model loaded successfully!
Draw a digit to see prediction...";
    let output_label = Label::new(Some(output_text));
    output_label.set_halign(gtk4::Align::Start);
    output_label.set_valign(gtk4::Align::Start);
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
          let image_data_2d = drawing_area_ui.borrow().get_image_data();
          let mut image = flatten_2d_to_1d(&image_data_2d).insert_axis(Axis(1));

          let values: Vec<f64> = image.iter().map(|v| (*v as f64)).collect();
          for r in 0..28 {
            for c in 0..28 {
              let v = values[r * 28 + c];
              print!("{:5.2} ", v);
            }
            println!();
          }

          // 784x1 * 128x784 = 128x1
          let z1 = &model.w1.dot(&image) + &model.b1;
          let a1 = sigmoid(&z1);

          // apply hidden activations to classification neurons
          // 128x1 * 10x128 = 10x1
          let z2 = model.w2.dot(&a1) + &model.b2;
          let a2 = softmax(&z2);

          println!("Predictions: {:?}", a2)
          /*
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

              */
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
