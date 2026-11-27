use std::fs::{self, File};
use std::io::{self, Read};
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::str::FromStr;

use clap::{Args, Parser, Subcommand};
use ollama_worker::{
    BenchmarkConfig, BenchmarkEndpoint, BenchmarkRunner, BenchmarkSuite, DEFAULT_OLLAMA_BASE_URL,
    DEFAULT_TIMEOUT_SECS, ErrorDetail, HttpOllamaClient, InferenceOptions, InstalledModel,
    TaskExecutor, TaskRequest, TaskResult, TaskType, WorkerConfig, WorkerError, WorkerResult,
};
use serde::Serialize;

const DEFAULT_CLI_TASK_ID: &str = "cli-task";

/// Hard upper bound (1 MiB) for `--input` files and `--stdin` payloads.
const MAX_INPUT_BYTES: usize = 1024 * 1024;

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

    /// Pretty-print JSON output on stdout and --output files.
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
    /// Run a reproducible benchmark suite and emit structured metrics JSON.
    Benchmark(Box<BenchmarkArgs>),
}

#[derive(Debug, Args)]
pub struct BenchmarkArgs {
    /// Target Ollama model name to benchmark (e.g., qwen2.5-coder:3b).
    #[arg(long, env = "OLLAMA_WORKER_MODEL")]
    pub model: String,

    /// Optional path to a custom JSON benchmark suite (defaults to built-in S1/S2/S3 suite).
    #[arg(long)]
    pub suite: Option<PathBuf>,

    /// Ollama endpoint to benchmark: chat, generate, or both.
    #[arg(long, default_value = "chat")]
    pub endpoint: String,

    /// Number of iterations per scenario (1..=20).
    #[arg(long, default_value_t = 1)]
    pub iterations: usize,

    /// Enable NDJSON streaming for `/api/chat` to record TTFT (`ttft_ms`).
    #[arg(long, default_value_t = false)]
    pub stream: bool,

    /// Unload the model (`keep_alive: "0"`) before iteration 1 of each scenario to measure cold start.
    #[arg(long, default_value_t = false)]
    pub cold_start: bool,

    /// Deterministic temperature for benchmark runs (default 0.0).
    #[arg(long, default_value_t = 0.0)]
    pub temperature: f32,

    /// Deterministic seed for benchmark runs (default 42).
    #[arg(long, default_value_t = 42)]
    pub seed: i64,

    /// Optional explicit `num_ctx` override (e.g. 4096 vs 8192).
    #[arg(long)]
    pub num_ctx: Option<u32>,

    /// Optional file path to write the benchmark report JSON.
    #[arg(long)]
    pub output: Option<PathBuf>,

    /// Suppress stdout JSON when --output is used.
    #[arg(long, default_value_t = false, requires = "output")]
    pub quiet: bool,
}

#[derive(Debug, Args)]
pub struct RunArgs {
    /// Path to a JSON task file (mutually exclusive with --task and --stdin).
    #[arg(long, conflicts_with_all = ["task", "stdin"])]
    pub input: Option<PathBuf>,

    /// Read a JSON task payload from stdin (mutually exclusive with --input and --task).
    #[arg(long, default_value_t = false, conflicts_with_all = ["input", "task"])]
    pub stdin: bool,

    /// Inline task instruction for quick CLI invocations.
    #[arg(long, conflicts_with_all = ["input", "stdin"])]
    pub task: Option<String>,

    /// Optional file path to write the structured JSON output.
    #[arg(long)]
    pub output: Option<PathBuf>,

    /// Suppress stdout JSON when --output is used.
    #[arg(long, default_value_t = false, requires = "output")]
    pub quiet: bool,

    /// Validate input, build prompt, and estimate tokens without calling Ollama.
    #[arg(long, default_value_t = false)]
    pub dry_run: bool,

    /// Target Ollama model name (overrides Task JSON; falls back to OLLAMA_WORKER_MODEL).
    #[arg(long)]
    pub model: Option<String>,

    /// Task type (freeform, generate_function, generate_unit_test, generate_docstring, explain_code, refactor_function, fix_bug, generate_json).
    #[arg(long = "type")]
    pub task_type: Option<String>,

    /// Target programming language for inline tasks or overrides.
    #[arg(long = "lang")]
    pub language: Option<String>,

    /// Task identifier (overrides Task JSON; defaults to "cli-task" for inline tasks).
    #[arg(long)]
    pub task_id: Option<String>,

    /// Optional custom system prompt overriding the built-in task template.
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

    /// Enable streaming NDJSON inference to measure Time to First Token (TTFT).
    #[arg(long, default_value_t = false)]
    pub stream: bool,

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
        Commands::Benchmark(args) => {
            run_benchmark(
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

async fn run_benchmark(
    base_url: String,
    timeout: u64,
    allow_remote_ollama: bool,
    args: BenchmarkArgs,
    pretty: bool,
) -> ExitCode {
    let config = match WorkerConfig::new(&base_url, timeout, allow_remote_ollama, None) {
        Ok(cfg) => cfg,
        Err(err) => return emit_command_error(&err, true, pretty),
    };

    let endpoint = match BenchmarkEndpoint::from_str(&args.endpoint) {
        Ok(ep) => ep,
        Err(err) => return emit_command_error(&err, true, pretty),
    };

    let suite = match &args.suite {
        Some(path) => match read_limited_file(path, MAX_INPUT_BYTES).and_then(|raw| {
            serde_json::from_str::<BenchmarkSuite>(&raw).map_err(|err| WorkerError::InvalidInput {
                message: format!("failed to parse benchmark suite JSON: {err}"),
            })
        }) {
            Ok(s) => s,
            Err(err) => return emit_command_error(&err, true, pretty),
        },
        None => BenchmarkSuite::default_suite(),
    };

    let bench_config = BenchmarkConfig {
        model: args.model,
        endpoint,
        iterations: args.iterations,
        stream: args.stream,
        cold_start: args.cold_start,
        temperature: args.temperature,
        seed: Some(args.seed),
        num_ctx: args.num_ctx,
    };

    let client = match HttpOllamaClient::new(config.clone()) {
        Ok(c) => c,
        Err(err) => return emit_command_error(&err, true, pretty),
    };

    let runner = BenchmarkRunner::new(client);
    match runner
        .run(&bench_config, &suite, config.base_url_str())
        .await
    {
        Ok(report) => emit_payload(&report, args.output.as_deref(), args.quiet, pretty, 0),
        Err(err) => emit_command_error(&err, true, pretty),
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
                emit_json_stdout(&report, pretty)
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
                emit_json_stdout(&ModelsOutput { models }, pretty)
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
    let fallback_task_id = args
        .task_id
        .clone()
        .unwrap_or_else(|| DEFAULT_CLI_TASK_ID.to_owned());
    let fallback_model = args.model.clone().unwrap_or_default();
    let output_path = args.output.clone();
    let quiet = args.quiet;

    let config = match WorkerConfig::new(&base_url, timeout, allow_remote_ollama, None) {
        Ok(cfg) => cfg.with_verify_model(!args.skip_model_check),
        Err(err) => {
            let result = TaskResult::from_error(&fallback_task_id, &fallback_model, &err);
            return emit_task_result(&result, output_path.as_deref(), quiet, pretty);
        }
    };

    let request = match resolve_task_request(&args) {
        Ok(req) => req,
        Err(err) => {
            let result = TaskResult::from_error(&fallback_task_id, &fallback_model, &err);
            return emit_task_result(&result, output_path.as_deref(), quiet, pretty);
        }
    };

    let client = match HttpOllamaClient::new(config.clone()) {
        Ok(c) => c,
        Err(err) => {
            let result = TaskResult::from_error(&request.task_id, &request.model, &err);
            return emit_task_result(&result, output_path.as_deref(), quiet, pretty);
        }
    };

    let executor = TaskExecutor::new(client, config.verify_model());

    if args.dry_run {
        return match executor.dry_run(&request) {
            Ok(report) => emit_payload(&report, output_path.as_deref(), quiet, pretty, 0),
            Err(err) => {
                let result = TaskResult::from_error(&request.task_id, &request.model, &err);
                emit_task_result(&result, output_path.as_deref(), quiet, pretty)
            }
        };
    }

    let result = executor
        .execute_with_cancel(&request, shutdown_signal())
        .await;
    emit_task_result(&result, output_path.as_deref(), quiet, pretty)
}

async fn shutdown_signal() {
    #[cfg(unix)]
    {
        use tokio::signal::unix::{SignalKind, signal};
        match signal(SignalKind::terminate()) {
            Ok(mut sigterm) => {
                tokio::select! {
                    _ = tokio::signal::ctrl_c() => {}
                    _ = sigterm.recv() => {}
                }
            }
            Err(_) => {
                let _ = tokio::signal::ctrl_c().await;
            }
        }
    }

    #[cfg(not(unix))]
    {
        let _ = tokio::signal::ctrl_c().await;
    }
}

fn resolve_task_request(args: &RunArgs) -> WorkerResult<TaskRequest> {
    let env_model = std::env::var("OLLAMA_WORKER_MODEL")
        .ok()
        .map(|v| v.trim().to_owned())
        .filter(|v| !v.is_empty());

    let mut request = if let Some(path) = &args.input {
        let raw = read_limited_file(path, MAX_INPUT_BYTES)?;
        parse_task_json(&raw)?
    } else if args.stdin {
        let raw = read_limited_reader(io::stdin().lock(), MAX_INPUT_BYTES, "stdin")?;
        parse_task_json(&raw)?
    } else if let Some(instruction) = &args.task {
        let model = args
            .model
            .clone()
            .or(env_model.clone())
            .filter(|m| !m.trim().is_empty())
            .ok_or_else(|| WorkerError::InvalidInput {
                message: "missing required --model (or OLLAMA_WORKER_MODEL)".into(),
            })?;
        let task_id = args
            .task_id
            .clone()
            .unwrap_or_else(|| DEFAULT_CLI_TASK_ID.to_owned());
        TaskRequest::new(task_id, model, instruction)
    } else {
        return Err(WorkerError::InvalidInput {
            message: "provide one of --input <FILE>, --stdin, or --task <INSTRUCTION>".into(),
        });
    };

    // Precedence: CLI Flags > Task JSON > Env > Built-in Defaults.
    if let Some(model) = &args.model {
        request.model = model.clone();
    } else if request.model.trim().is_empty()
        && let Some(env_m) = env_model
    {
        request.model = env_m;
    }

    if let Some(task_id) = &args.task_id {
        request.task_id = task_id.clone();
    }

    if let Some(raw_type) = &args.task_type {
        request.task_type = TaskType::from_str(raw_type)?;
    }

    if let Some(lang) = &args.language {
        request.language = Some(lang.clone());
    }

    if let Some(system) = &args.system {
        request.system_prompt = Some(system.clone());
    }

    let cli_overrides = InferenceOptions {
        temperature: args.temperature,
        num_ctx: args.num_ctx,
        max_tokens: args.max_tokens,
        seed: args.seed,
        keep_alive: args.keep_alive.clone(),
        stream: args.stream,
    };
    request.options = request.options.merge_override(&cli_overrides);

    Ok(request)
}

fn parse_task_json(raw: &str) -> WorkerResult<TaskRequest> {
    if raw.trim().is_empty() {
        return Err(WorkerError::InvalidInput {
            message: "task JSON input is empty".into(),
        });
    }
    serde_json::from_str::<TaskRequest>(raw).map_err(|err| WorkerError::InvalidInput {
        message: format!("failed to parse task JSON: {err}"),
    })
}

fn read_limited_file(path: &Path, max_bytes: usize) -> WorkerResult<String> {
    let metadata = fs::metadata(path).map_err(|err| WorkerError::InvalidInput {
        message: format!("cannot access input file '{}': {err}", path.display()),
    })?;
    if !metadata.is_file() {
        return Err(WorkerError::InvalidInput {
            message: format!("input path '{}' is not a regular file", path.display()),
        });
    }
    if metadata.len() > max_bytes as u64 {
        return Err(WorkerError::InvalidInput {
            message: format!(
                "input file '{}' exceeds maximum allowed size of {max_bytes} bytes",
                path.display()
            ),
        });
    }

    let file = File::open(path).map_err(|err| WorkerError::InvalidInput {
        message: format!("failed to open input file '{}': {err}", path.display()),
    })?;
    read_limited_reader(file, max_bytes, &path.display().to_string())
}

fn read_limited_reader<R: Read>(reader: R, max_bytes: usize, label: &str) -> WorkerResult<String> {
    let mut limited = reader.take((max_bytes as u64) + 1);
    let mut buffer = Vec::new();
    limited
        .read_to_end(&mut buffer)
        .map_err(|err| WorkerError::InvalidInput {
            message: format!("failed to read {label}: {err}"),
        })?;
    if buffer.len() > max_bytes {
        return Err(WorkerError::InvalidInput {
            message: format!("{label} exceeds maximum allowed size of {max_bytes} bytes"),
        });
    }
    String::from_utf8(buffer).map_err(|err| WorkerError::InvalidInput {
        message: format!("{label} is not valid UTF-8: {err}"),
    })
}

fn write_output_file(path: &Path, contents: &str) -> WorkerResult<()> {
    if path.is_dir() {
        return Err(WorkerError::InvalidInput {
            message: format!("output path '{}' is a directory", path.display()),
        });
    }
    if let Some(parent) = path.parent()
        && !parent.as_os_str().is_empty()
        && !parent.exists()
    {
        return Err(WorkerError::InvalidInput {
            message: format!(
                "parent directory '{}' for --output does not exist",
                parent.display()
            ),
        });
    }
    fs::write(path, contents).map_err(|err| WorkerError::Internal {
        message: format!("failed to write output file '{}': {err}", path.display()),
    })
}

fn emit_task_result(
    result: &TaskResult,
    output_path: Option<&Path>,
    quiet: bool,
    pretty: bool,
) -> ExitCode {
    emit_payload(result, output_path, quiet, pretty, result.exit_code())
}

fn emit_payload<T: Serialize>(
    value: &T,
    output_path: Option<&Path>,
    quiet: bool,
    pretty: bool,
    exit_code: u8,
) -> ExitCode {
    let rendered = match render_json(value, pretty) {
        Ok(text) => text,
        Err(err) => {
            eprintln!("error: failed to serialize JSON output: {err}");
            return ExitCode::from(1);
        }
    };

    if let Some(path) = output_path
        && let Err(err) = write_output_file(path, &format!("{rendered}\n"))
    {
        eprintln!("error: {err}");
        return ExitCode::from(err.exit_code());
    }

    if !quiet {
        println!("{rendered}");
    }

    ExitCode::from(exit_code)
}

fn emit_command_error(err: &WorkerError, json_output: bool, pretty: bool) -> ExitCode {
    if json_output {
        let payload = CommandErrorOutput {
            status: "error",
            error: err.to_detail(),
        };
        let _ = emit_json_stdout(&payload, pretty);
    } else {
        eprintln!("error: {err}");
    }
    ExitCode::from(err.exit_code())
}

fn emit_json_stdout<T: Serialize>(value: &T, pretty: bool) -> ExitCode {
    match render_json(value, pretty) {
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

fn render_json<T: Serialize>(value: &T, pretty: bool) -> Result<String, serde_json::Error> {
    if pretty {
        serde_json::to_string_pretty(value)
    } else {
        serde_json::to_string(value)
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
            "--type",
            "generate_function",
            "--lang",
            "python",
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
            "--dry-run",
        ])
        .expect("CLI should parse valid run arguments");

        assert_eq!(cli.base_url, DEFAULT_OLLAMA_BASE_URL);
        assert_eq!(cli.timeout, DEFAULT_TIMEOUT_SECS);
        assert!(!cli.allow_remote_ollama);
        assert!(cli.pretty);

        match cli.command {
            Commands::Run(args) => {
                assert_eq!(args.model.as_deref(), Some("qwen2.5-coder:3b"));
                assert_eq!(args.task_type.as_deref(), Some("generate_function"));
                assert_eq!(args.language.as_deref(), Some("python"));
                assert_eq!(args.task.as_deref(), Some("Write is_prime in Python"));
                assert_eq!(args.task_id.as_deref(), Some("task-042"));
                assert_eq!(args.temperature, Some(0.2));
                assert_eq!(args.max_tokens, Some(256));
                assert_eq!(args.num_ctx, Some(4096));
                assert_eq!(args.seed, Some(7));
                assert!(args.skip_model_check);
                assert!(args.dry_run);
            }
            other => panic!("expected Run command, got {other:?}"),
        }
    }

    #[test]
    fn cli_enforces_mutually_exclusive_task_input_sources() {
        let conflict =
            Cli::try_parse_from(["ollama-worker", "run", "--input", "task.json", "--stdin"]);
        assert!(conflict.is_err());

        let quiet_without_output =
            Cli::try_parse_from(["ollama-worker", "run", "--stdin", "--quiet"]);
        assert!(quiet_without_output.is_err());
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

    #[test]
    fn limited_reader_rejects_oversized_payload() {
        let oversized = [b'a'; 64];
        let err = read_limited_reader(&oversized[..], 32, "stdin")
            .expect_err("payload over limit must fail");
        assert_eq!(err.exit_code(), 2);
    }
}
