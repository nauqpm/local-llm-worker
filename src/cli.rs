use std::process::ExitCode;

use clap::{Args, Parser, Subcommand};
use ollama_worker::{
    DEFAULT_OLLAMA_BASE_URL, DEFAULT_TIMEOUT_SECS, ErrorDetail, HttpOllamaClient, InferenceOptions,
    InstalledModel, TaskExecutor, TaskRequest, TaskResult, WorkerConfig, WorkerError,
};
use serde::Serialize;

const DEFAULT_CLI_TASK_ID: &str = "cli-task";

#[derive(Debug, Parser)]
#[command(
    name = "ollama-worker",
    version,
    about = "Lightweight local LLM worker for Ollama and AI agent orchestration"
)]
pub struct Cli {
    /// Base URL of the local Ollama HTTP API.
    #[arg(
        long,
        global = true,
        env = "OLLAMA_BASE_URL",
        default_value = DEFAULT_OLLAMA_BASE_URL
    )]
    pub base_url: String,

    /// Request timeout in seconds.
    #[arg(
        long,
        global = true,
        env = "OLLAMA_WORKER_TIMEOUT_SECS",
        default_value_t = DEFAULT_TIMEOUT_SECS
    )]
    pub timeout: u64,

    /// Explicitly permit connecting to a non-loopback Ollama server.
    #[arg(long, global = true, default_value_t = false)]
    pub allow_remote_ollama: bool,

    /// Pretty-print JSON output on stdout.
    #[arg(long, global = true, default_value_t = false)]
    pub pretty: bool,

    #[command(subcommand)]
    pub command: Commands,
}

#[derive(Debug, Subcommand)]
pub enum Commands {
    /// Check Ollama server availability and currently loaded models.
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
    Run(Box<RunArgs>),
}

#[derive(Debug, Args)]
pub struct RunArgs {
    /// Target Ollama model name (e.g., qwen2.5-coder:3b).
    #[arg(long, env = "OLLAMA_WORKER_MODEL")]
    pub model: Option<String>,

    /// Inline task instruction for quick CLI invocations.
    #[arg(long)]
    pub task: Option<String>,

    /// Task identifier included in the structured JSON result.
    #[arg(long, default_value = DEFAULT_CLI_TASK_ID)]
    pub task_id: String,

    /// Optional custom system prompt overriding the built-in concise default.
    #[arg(long)]
    pub system: Option<String>,

    /// Sampling temperature in 0.0..=2.0.
    #[arg(long)]
    pub temperature: Option<f32>,

    /// Maximum tokens to generate (`options.num_predict`).
    #[arg(long)]
    pub max_tokens: Option<u32>,

    /// Explicit context window override (`options.num_ctx`).
    #[arg(long)]
    pub num_ctx: Option<u32>,

    /// Deterministic sampling seed when supported by Ollama.
    #[arg(long)]
    pub seed: Option<i64>,

    /// Per-request model keep-alive duration (e.g., `5m`, `0`, `-1`).
    #[arg(long)]
    pub keep_alive: Option<String>,

    /// Skip the pre-flight `/api/tags` model existence check.
    #[arg(long, default_value_t = false)]
    pub skip_model_check: bool,
}

#[derive(Debug, Serialize)]
struct ModelsOutput {
    models: Vec<InstalledModel>,
}

#[derive(Debug, Serialize)]
struct CommandErrorOutput {
    status: &'static str,
    error: ErrorDetail,
}

pub async fn execute_cli(cli: Cli) -> ExitCode {
    let pretty = cli.pretty;
    match cli.command {
        Commands::Health { json } => {
            run_health(
                cli.base_url,
                cli.timeout,
                cli.allow_remote_ollama,
                json,
                pretty,
            )
            .await
        }
        Commands::Models { json } => {
            run_models(
                cli.base_url,
                cli.timeout,
                cli.allow_remote_ollama,
                json,
                pretty,
            )
            .await
        }
        Commands::Run(args) => {
            run_task(
                cli.base_url,
                cli.timeout,
                cli.allow_remote_ollama,
                *args,
                pretty,
            )
            .await
        }
    }
}

async fn run_health(
    base_url: String,
    timeout: u64,
    allow_remote_ollama: bool,
    json_output: bool,
    pretty: bool,
) -> ExitCode {
    let config = match WorkerConfig::new(&base_url, timeout, allow_remote_ollama, None) {
        Ok(cfg) => cfg,
        Err(err) => return emit_command_error(&err, json_output, pretty),
    };

    let client = match HttpOllamaClient::new(config.clone()) {
        Ok(c) => c,
        Err(err) => return emit_command_error(&err, json_output, pretty),
    };

    let executor = TaskExecutor::new(client, config.verify_model());
    match executor.health(config.base_url_str()).await {
        Ok(report) => {
            if json_output {
                emit_json(&report, pretty)
            } else {
                println!(
                    "Ollama OK (version {}, endpoint {})",
                    report.ollama_version, report.base_url
                );
                if report.running_models.is_empty() {
                    println!("Loaded models: none");
                } else {
                    println!("Loaded models ({}):", report.running_models.len());
                    for model in &report.running_models {
                        println!("  - {} ({:.1} MiB)", model.name, bytes_to_mib(model.size));
                    }
                }
                ExitCode::SUCCESS
            }
        }
        Err(err) => emit_command_error(&err, json_output, pretty),
    }
}

async fn run_models(
    base_url: String,
    timeout: u64,
    allow_remote_ollama: bool,
    json_output: bool,
    pretty: bool,
) -> ExitCode {
    let config = match WorkerConfig::new(&base_url, timeout, allow_remote_ollama, None) {
        Ok(cfg) => cfg,
        Err(err) => return emit_command_error(&err, json_output, pretty),
    };

    let client = match HttpOllamaClient::new(config.clone()) {
        Ok(c) => c,
        Err(err) => return emit_command_error(&err, json_output, pretty),
    };

    let executor = TaskExecutor::new(client, config.verify_model());
    match executor.models().await {
        Ok(models) => {
            if json_output {
                emit_json(&ModelsOutput { models }, pretty)
            } else if models.is_empty() {
                println!("No models installed.");
                ExitCode::SUCCESS
            } else {
                for model in &models {
                    let quant = model
                        .details
                        .as_ref()
                        .and_then(|d| d.quantization_level.as_deref())
                        .unwrap_or("-");
                    let params = model
                        .details
                        .as_ref()
                        .and_then(|d| d.parameter_size.as_deref())
                        .unwrap_or("-");
                    println!(
                        "{}\t{}\t{}\t{:.1} MiB",
                        model.name,
                        params,
                        quant,
                        bytes_to_mib(model.size)
                    );
                }
                ExitCode::SUCCESS
            }
        }
        Err(err) => emit_command_error(&err, json_output, pretty),
    }
}

async fn run_task(
    base_url: String,
    timeout: u64,
    allow_remote_ollama: bool,
    args: RunArgs,
    pretty: bool,
) -> ExitCode {
    let requested_model = args.model.clone().unwrap_or_default();
    let config =
        match WorkerConfig::new(&base_url, timeout, allow_remote_ollama, args.model.clone()) {
            Ok(cfg) => cfg.with_verify_model(!args.skip_model_check),
            Err(err) => {
                let result = TaskResult::from_error(&args.task_id, &requested_model, &err);
                return emit_task_result(&result, pretty);
            }
        };

    let Some(model) = args.model.filter(|m| !m.trim().is_empty()) else {
        let err = WorkerError::InvalidInput {
            message: "missing required --model (or OLLAMA_WORKER_MODEL)".into(),
        };
        let result = TaskResult::from_error(&args.task_id, "", &err);
        return emit_task_result(&result, pretty);
    };

    let Some(instruction) = args.task.filter(|t| !t.trim().is_empty()) else {
        let err = WorkerError::InvalidInput {
            message: "missing required --task instruction".into(),
        };
        let result = TaskResult::from_error(&args.task_id, &model, &err);
        return emit_task_result(&result, pretty);
    };

    let overrides = InferenceOptions {
        temperature: args.temperature,
        num_ctx: args.num_ctx,
        max_tokens: args.max_tokens,
        seed: args.seed,
        keep_alive: args.keep_alive,
    };
    let options = InferenceOptions::default().merge_override(&overrides);

    let mut request = TaskRequest::new(args.task_id, model, instruction);
    request.system_prompt = args.system;
    request.options = options;

    let client = match HttpOllamaClient::new(config.clone()) {
        Ok(c) => c,
        Err(err) => {
            let result = TaskResult::from_error(&request.task_id, &request.model, &err);
            return emit_task_result(&result, pretty);
        }
    };

    let executor = TaskExecutor::new(client, config.verify_model());
    let result = executor.execute(&request).await;
    emit_task_result(&result, pretty)
}

fn emit_task_result(result: &TaskResult, pretty: bool) -> ExitCode {
    let code = emit_json(result, pretty);
    if code != ExitCode::SUCCESS {
        return code;
    }
    ExitCode::from(result.exit_code())
}

fn emit_command_error(err: &WorkerError, json_output: bool, pretty: bool) -> ExitCode {
    if json_output {
        let payload = CommandErrorOutput {
            status: "error",
            error: err.to_detail(),
        };
        let _ = emit_json(&payload, pretty);
    } else {
        eprintln!("error: {err}");
    }
    ExitCode::from(err.exit_code())
}

fn emit_json<T: Serialize>(value: &T, pretty: bool) -> ExitCode {
    let rendered = if pretty {
        serde_json::to_string_pretty(value)
    } else {
        serde_json::to_string(value)
    };

    match rendered {
        Ok(text) => {
            println!("{text}");
            ExitCode::SUCCESS
        }
        Err(err) => {
            eprintln!("error: failed to serialize JSON output: {err}");
            ExitCode::from(1)
        }
    }
}

fn bytes_to_mib(bytes: u64) -> f64 {
    (bytes as f64) / (1024.0 * 1024.0)
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
    fn cli_parses_run_command_with_defaults_and_overrides() {
        let cli = Cli::try_parse_from([
            "ollama-worker",
            "--pretty",
            "run",
            "--model",
            "qwen2.5-coder:3b",
            "--task",
            "Write is_prime in Python",
            "--task-id",
            "task-042",
            "--temperature",
            "0.2",
            "--max-tokens",
            "256",
            "--num-ctx",
            "4096",
            "--seed",
            "7",
            "--skip-model-check",
        ])
        .expect("CLI should parse valid run arguments");

        assert_eq!(cli.base_url, DEFAULT_OLLAMA_BASE_URL);
        assert_eq!(cli.timeout, DEFAULT_TIMEOUT_SECS);
        assert!(!cli.allow_remote_ollama);
        assert!(cli.pretty);

        match cli.command {
            Commands::Run(args) => {
                assert_eq!(args.model.as_deref(), Some("qwen2.5-coder:3b"));
                assert_eq!(args.task.as_deref(), Some("Write is_prime in Python"));
                assert_eq!(args.task_id, "task-042");
                assert_eq!(args.temperature, Some(0.2));
                assert_eq!(args.max_tokens, Some(256));
                assert_eq!(args.num_ctx, Some(4096));
                assert_eq!(args.seed, Some(7));
                assert!(args.skip_model_check);
            }
            other => panic!("expected Run command, got {other:?}"),
        }
    }

    #[test]
    fn cli_accepts_remote_opt_in_flag() {
        let cli = Cli::try_parse_from([
            "ollama-worker",
            "--base-url",
            "http://100.64.0.10:11434",
            "--allow-remote-ollama",
            "health",
            "--json",
        ])
        .expect("CLI should parse remote opt-in flag");

        assert!(cli.allow_remote_ollama);
        assert_eq!(cli.base_url, "http://100.64.0.10:11434");
        assert!(matches!(cli.command, Commands::Health { json: true }));
    }
}
