use clap::{Parser, Subcommand};
use mnist::*;
use neural_net::{TRAINING_SIZE, run_train};

// sigmoid "clamps" values (in a fairly scaled way) to 0..1
// Training logic moved to `training.rs`

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

    Commands::GUI { model } => {
      // Create a GUI window
      neural_net::gui::window::create_window(model);
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

  /// Create a GUI window
  GUI {
    #[arg(short, long, default_value = "model.safetensors")]
    model: String,
  },
}
