use std::process::ExitCode;

use clap::{Parser, Subcommand};
use ollama_worker::{DEFAULT_OLLAMA_BASE_URL, DEFAULT_TIMEOUT_SECS};
use tracing_subscriber::EnvFilter;

#[derive(Debug, Parser)]
#[command(
    name = "ollama-worker",
    version,
    about = "Lightweight local LLM worker for Ollama and AI agent orchestration"
)]
struct Cli {
    /// Base URL of the local Ollama HTTP API.
    #[arg(
        long,
        global = true,
        env = "OLLAMA_BASE_URL",
        default_value = DEFAULT_OLLAMA_BASE_URL
    )]
    base_url: String,

    /// Request timeout in seconds.
    #[arg(
        long,
        global = true,
        env = "OLLAMA_WORKER_TIMEOUT_SECS",
        default_value_t = DEFAULT_TIMEOUT_SECS
    )]
    timeout: u64,

    #[command(subcommand)]
    command: Commands,
}

#[derive(Debug, Subcommand)]
enum Commands {
    /// Check Ollama server availability and loaded models.
    Health {
        /// Output machine-readable JSON to stdout.
        #[arg(long)]
        json: bool,
    },
    /// List models installed on the Ollama server.
    Models {
        /// Output machine-readable JSON to stdout.
        #[arg(long)]
        json: bool,
    },
    /// Execute a single local inference task and emit structured JSON to stdout.
    Run {
        /// Target Ollama model name (e.g., qwen2.5-coder:3b).
        #[arg(long, env = "OLLAMA_WORKER_MODEL")]
        model: Option<String>,

        /// Inline task instruction for quick CLI invocations.
        #[arg(long)]
        task: Option<String>,
    },
}

fn init_logging() {
    let filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("warn"));
    tracing_subscriber::fmt()
        .with_env_filter(filter)
        .with_writer(std::io::stderr)
        .without_time()
        .init();
}

#[tokio::main(flavor = "current_thread")]
async fn main() -> ExitCode {
    init_logging();

    let cli = Cli::parse();
    tracing::debug!(
        base_url = %cli.base_url,
        timeout_secs = cli.timeout,
        command = ?cli.command,
        "parsed CLI arguments"
    );

    match cli.command {
        Commands::Health { .. } | Commands::Models { .. } | Commands::Run { .. } => {
            eprintln!("error: subcommand is not implemented yet in the initial skeleton");
            ExitCode::from(1)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::CommandFactory;

    #[test]
    fn cli_definition_is_valid() {
        Cli::command().debug_assert();
    }

    #[test]
    fn cli_parses_run_command_with_defaults() {
        let cli = Cli::try_parse_from([
            "ollama-worker",
            "run",
            "--model",
            "qwen2.5-coder:3b",
            "--task",
            "Write is_prime in Python",
        ])
        .expect("CLI should parse valid run arguments");

        assert_eq!(cli.base_url, DEFAULT_OLLAMA_BASE_URL);
        assert_eq!(cli.timeout, DEFAULT_TIMEOUT_SECS);
        match cli.command {
            Commands::Run { model, task } => {
                assert_eq!(model.as_deref(), Some("qwen2.5-coder:3b"));
                assert_eq!(task.as_deref(), Some("Write is_prime in Python"));
            }
            other => panic!("expected Run command, got {other:?}"),
        }
    }
}
