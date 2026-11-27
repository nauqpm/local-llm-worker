//! Worker runtime configuration, loopback security checks, and inference options.

use std::time::Duration;

use serde::{Deserialize, Serialize};
use url::{Host, Url};

use crate::error::{WorkerError, WorkerResult};
use crate::{DEFAULT_OLLAMA_BASE_URL, DEFAULT_TIMEOUT_SECS};

/// Upper bound for request timeout (1 hour) to guard against misconfiguration.
pub const MAX_TIMEOUT_SECS: u64 = 3600;

/// Default temperature tuned for deterministic coding tasks on a 3B model.
pub const DEFAULT_TEMPERATURE: f32 = 0.1;

/// Default maximum predicted tokens to prevent runaway CPU generation.
pub const DEFAULT_MAX_TOKENS: u32 = 512;

/// Hard upper bound for per-request predicted tokens.
pub const MAX_PREDICT_TOKENS: u32 = 8192;

/// Minimum sensible context window when `num_ctx` is explicitly specified.
pub const MIN_NUM_CTX: u32 = 256;

/// Validated runtime configuration for communicating with Ollama.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WorkerConfig {
    base_url: Url,
    timeout: Duration,
    allow_remote_ollama: bool,
    verify_model: bool,
    default_model: Option<String>,
}

impl Default for WorkerConfig {
    fn default() -> Self {
        Self::new(DEFAULT_OLLAMA_BASE_URL, DEFAULT_TIMEOUT_SECS, false, None)
            .expect("built-in default WorkerConfig must always be valid")
    }
}

impl WorkerConfig {
    /// Builds and validates a `WorkerConfig` from raw configuration parts.
    ///
    /// Enforces:
    /// - Valid `http` or `https` URL without embedded credentials, query, or fragment.
    /// - Loopback-only host (`127.0.0.0/8`, `::1`, or `localhost`) unless `allow_remote_ollama` is true.
    /// - Bounded positive timeout (`1..=MAX_TIMEOUT_SECS`).
    /// - Trimmed non-empty `default_model` if supplied.
    pub fn new(
        base_url: &str,
        timeout_secs: u64,
        allow_remote_ollama: bool,
        default_model: Option<String>,
    ) -> WorkerResult<Self> {
        let parsed_url = validate_base_url(base_url, allow_remote_ollama)?;

        if timeout_secs == 0 || timeout_secs > MAX_TIMEOUT_SECS {
            return Err(WorkerError::InvalidConfig {
                message: format!(
                    "timeout must be between 1 and {MAX_TIMEOUT_SECS} seconds, got {timeout_secs}"
                ),
            });
        }

        let normalized_model = match default_model {
            Some(raw) => {
                let trimmed = raw.trim();
                if trimmed.is_empty() {
                    return Err(WorkerError::InvalidConfig {
                        message: "default model name must not be empty or whitespace".into(),
                    });
                }
                Some(trimmed.to_owned())
            }
            None => None,
        };

        Ok(Self {
            base_url: parsed_url,
            timeout: Duration::from_secs(timeout_secs),
            allow_remote_ollama,
            verify_model: true,
            default_model: normalized_model,
        })
    }

    /// Returns the validated, normalized base URL.
    #[must_use]
    pub fn base_url(&self) -> &Url {
        &self.base_url
    }

    /// Returns the normalized base URL as a string without a trailing slash.
    #[must_use]
    pub fn base_url_str(&self) -> &str {
        self.base_url.as_str().trim_end_matches('/')
    }

    /// Resolves a native Ollama API path (for example, `"/api/chat"`) against `base_url`.
    pub fn endpoint_url(&self, api_path: &str) -> WorkerResult<Url> {
        let trimmed = api_path.trim_start_matches('/');
        let base_with_slash = format!("{}/", self.base_url_str());
        let base = Url::parse(&base_with_slash).map_err(|err| WorkerError::Internal {
            message: format!("failed to reconstruct normalized base URL: {err}"),
        })?;
        base.join(trimmed)
            .map_err(|err| WorkerError::InvalidConfig {
                message: format!("invalid API endpoint path '{api_path}': {err}"),
            })
    }

    /// Returns the configured per-request timeout.
    #[must_use]
    pub const fn timeout(&self) -> Duration {
        self.timeout
    }

    /// Returns whether non-loopback Ollama endpoints are explicitly permitted.
    #[must_use]
    pub const fn allow_remote_ollama(&self) -> bool {
        self.allow_remote_ollama
    }

    /// Returns whether target model existence is verified via `/api/tags` before inference.
    #[must_use]
    pub const fn verify_model(&self) -> bool {
        self.verify_model
    }

    /// Sets whether target model existence should be checked before inference.
    #[must_use]
    pub const fn with_verify_model(mut self, verify_model: bool) -> Self {
        self.verify_model = verify_model;
        self
    }

    /// Returns the optional default model name.
    #[must_use]
    pub fn default_model(&self) -> Option<&str> {
        self.default_model.as_deref()
    }
}

/// Per-request Ollama inference options.
///
/// `num_ctx` is intentionally `None` by default so requests do not force Ollama to
/// unload and reload a model already resident in memory with a different context size.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct InferenceOptions {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,

    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub num_ctx: Option<u32>,

    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u32>,

    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub seed: Option<i64>,

    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub keep_alive: Option<String>,
}

impl Default for InferenceOptions {
    fn default() -> Self {
        Self {
            temperature: Some(DEFAULT_TEMPERATURE),
            num_ctx: None,
            max_tokens: Some(DEFAULT_MAX_TOKENS),
            seed: None,
            keep_alive: None,
        }
    }
}

impl InferenceOptions {
    /// Validates inference parameters before dispatching to Ollama.
    pub fn validate(&self) -> WorkerResult<()> {
        if let Some(temp) = self.temperature
            && (!temp.is_finite() || !(0.0..=2.0).contains(&temp))
        {
            return Err(WorkerError::InvalidInput {
                message: format!("temperature must be a finite number in 0.0..=2.0, got {temp}"),
            });
        }

        if let Some(ctx) = self.num_ctx
            && ctx < MIN_NUM_CTX
        {
            return Err(WorkerError::InvalidInput {
                message: format!("num_ctx must be at least {MIN_NUM_CTX}, got {ctx}"),
            });
        }

        if let Some(max_tokens) = self.max_tokens
            && (max_tokens == 0 || max_tokens > MAX_PREDICT_TOKENS)
        {
            return Err(WorkerError::InvalidInput {
                message: format!(
                    "max_tokens must be between 1 and {MAX_PREDICT_TOKENS}, got {max_tokens}"
                ),
            });
        }

        if let Some(keep_alive) = &self.keep_alive
            && keep_alive.trim().is_empty()
        {
            return Err(WorkerError::InvalidInput {
                message: "keep_alive must not be empty or whitespace".into(),
            });
        }

        Ok(())
    }

    /// Merges higher-precedence overrides on top of `self`.
    #[must_use]
    pub fn merge_override(&self, overrides: &Self) -> Self {
        Self {
            temperature: overrides.temperature.or(self.temperature),
            num_ctx: overrides.num_ctx.or(self.num_ctx),
            max_tokens: overrides.max_tokens.or(self.max_tokens),
            seed: overrides.seed.or(self.seed),
            keep_alive: overrides
                .keep_alive
                .clone()
                .or_else(|| self.keep_alive.clone()),
        }
    }
}

fn validate_base_url(raw_url: &str, allow_remote_ollama: bool) -> WorkerResult<Url> {
    let trimmed = raw_url.trim();
    if trimmed.is_empty() {
        return Err(WorkerError::InvalidConfig {
            message: "base_url must not be empty".into(),
        });
    }

    let parsed = Url::parse(trimmed).map_err(|err| WorkerError::InvalidConfig {
        message: format!("invalid base_url '{trimmed}': {err}"),
    })?;

    if !matches!(parsed.scheme(), "http" | "https") {
        return Err(WorkerError::InvalidConfig {
            message: format!(
                "base_url scheme must be 'http' or 'https', got '{}'",
                parsed.scheme()
            ),
        });
    }

    if !parsed.username().is_empty() || parsed.password().is_some() {
        return Err(WorkerError::InvalidConfig {
            message: "base_url must not contain embedded credentials".into(),
        });
    }

    if parsed.query().is_some() || parsed.fragment().is_some() {
        return Err(WorkerError::InvalidConfig {
            message: "base_url must not include query parameters or fragments".into(),
        });
    }

    let Some(host) = parsed.host() else {
        return Err(WorkerError::InvalidConfig {
            message: format!("base_url '{trimmed}' is missing a host"),
        });
    };

    if !allow_remote_ollama && !is_loopback_host(&host) {
        return Err(WorkerError::RemoteEndpointBlocked {
            url: parsed.to_string(),
        });
    }

    Ok(parsed)
}

fn is_loopback_host(host: &Host<&str>) -> bool {
    match host {
        Host::Ipv4(addr) => addr.is_loopback(),
        Host::Ipv6(addr) => addr.is_loopback(),
        Host::Domain(domain) => domain.eq_ignore_ascii_case("localhost"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::ErrorCategory;

    #[test]
    fn default_config_uses_loopback_and_120s_timeout() {
        let cfg = WorkerConfig::default();
        assert_eq!(cfg.base_url_str(), "http://127.0.0.1:11434");
        assert_eq!(cfg.timeout(), Duration::from_secs(120));
        assert!(!cfg.allow_remote_ollama());
        assert!(cfg.verify_model());
        assert_eq!(cfg.default_model(), None);
    }

    #[test]
    fn accepts_valid_loopback_hosts_and_resolves_endpoints() {
        for raw in [
            "http://127.0.0.1:11434",
            "http://127.0.0.1:11434/",
            "http://127.0.0.2:11434",
            "http://localhost:11434",
            "http://LOCALHOST:11434/",
            "http://[::1]:11434",
        ] {
            let cfg = WorkerConfig::new(raw, 60, false, Some(" qwen2.5-coder:3b ".into()))
                .unwrap_or_else(|err| panic!("expected {raw} to be valid: {err}"));
            let chat_url = cfg
                .endpoint_url("/api/chat")
                .expect("endpoint_url should resolve");
            assert!(chat_url.as_str().ends_with("/api/chat"));
            assert_eq!(cfg.default_model(), Some("qwen2.5-coder:3b"));
        }
    }

    #[test]
    fn blocks_remote_and_spoofed_loopback_hosts_by_default() {
        for raw in [
            "http://192.168.1.50:11434",
            "http://10.0.0.5:11434",
            "http://127.0.0.1.evil.example:11434",
            "http://localhost.evil.example:11434",
            "https://ollama.example.com",
        ] {
            let err = WorkerConfig::new(raw, 60, false, None)
                .expect_err("remote URL must be blocked without explicit opt-in");
            assert_eq!(err.category(), ErrorCategory::InvalidInput);
            assert_eq!(err.exit_code(), 2);
            assert!(matches!(err, WorkerError::RemoteEndpointBlocked { .. }));
        }
    }

    #[test]
    fn allows_remote_host_only_when_explicitly_opted_in() {
        let cfg = WorkerConfig::new("http://100.64.0.10:11434/", 90, true, None)
            .expect("remote URL should succeed when allow_remote_ollama is true");
        assert!(cfg.allow_remote_ollama());
        assert_eq!(cfg.base_url_str(), "http://100.64.0.10:11434");
    }

    #[test]
    fn rejects_embedded_credentials_bad_schemes_and_invalid_timeouts() {
        for raw in [
            "",
            "   ",
            "ftp://127.0.0.1:11434",
            "file:///etc/passwd",
            "http://localhost@evil.example:11434",
            "http://user:pass@127.0.0.1:11434",
            "http://127.0.0.1:11434?foo=bar",
            "http://127.0.0.1:11434#frag",
        ] {
            let err = WorkerConfig::new(raw, 60, true, None)
                .expect_err("malformed or unsafe URL must fail");
            assert_eq!(err.exit_code(), 2);
        }

        assert!(WorkerConfig::new("http://127.0.0.1:11434", 0, false, None).is_err());
        assert!(
            WorkerConfig::new("http://127.0.0.1:11434", MAX_TIMEOUT_SECS + 1, false, None).is_err()
        );
        assert!(
            WorkerConfig::new("http://127.0.0.1:11434", 30, false, Some("   ".into())).is_err()
        );
    }

    #[test]
    fn inference_options_defaults_validation_and_merging() {
        let base = InferenceOptions::default();
        assert_eq!(base.temperature, Some(0.1));
        assert_eq!(base.num_ctx, None);
        assert_eq!(base.max_tokens, Some(512));
        base.validate().expect("default options should be valid");

        let overrides = InferenceOptions {
            temperature: Some(0.0),
            num_ctx: Some(4096),
            max_tokens: None,
            seed: Some(42),
            keep_alive: Some("5m".into()),
        };
        let merged = base.merge_override(&overrides);
        assert_eq!(merged.temperature, Some(0.0));
        assert_eq!(merged.num_ctx, Some(4096));
        assert_eq!(merged.max_tokens, Some(512));
        assert_eq!(merged.seed, Some(42));
        assert_eq!(merged.keep_alive.as_deref(), Some("5m"));
        merged.validate().expect("merged options should be valid");
    }

    #[test]
    fn inference_options_rejects_out_of_range_values() {
        let bad_temp = InferenceOptions {
            temperature: Some(f32::NAN),
            ..Default::default()
        };
        assert!(bad_temp.validate().is_err());

        let high_temp = InferenceOptions {
            temperature: Some(2.5),
            ..Default::default()
        };
        assert!(high_temp.validate().is_err());

        let zero_tokens = InferenceOptions {
            max_tokens: Some(0),
            ..Default::default()
        };
        assert!(zero_tokens.validate().is_err());

        let tiny_ctx = InferenceOptions {
            num_ctx: Some(64),
            ..Default::default()
        };
        assert!(tiny_ctx.validate().is_err());

        let blank_keep_alive = InferenceOptions {
            keep_alive: Some("  ".into()),
            ..Default::default()
        };
        assert!(blank_keep_alive.validate().is_err());
    }
}
