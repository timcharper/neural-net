use clap::{Parser, Subcommand};
use neural_net::training::run_train;

// sigmoid "clamps" values (in a fairly scaled way) to 0..1
// Training logic moved to `training.rs`

fn main() {
  // CLI
  let cli = Cli::parse();

  match &cli.command {
    Commands::Train { out } => {
      run_train(out);
    }

    Commands::Infer { .. } => {
      // Not implemented yet
      unimplemented!("infer is not implemented yet");
    }

    Commands::GUI { model } => {
      // Create a GUI window
      neural_net::gui::window::create_window(model);
    }

    Commands::Validate { model } => {
      neural_net::validate::validate(model);
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

  /// Validate a model
  Validate {
    #[arg(short, long, default_value = "model.safetensors")]
    model: String,
  },
}
