//! Static, non-executing output validation for local LLM responses.
//!
//! Never executes generated code or shell commands; treats all model outputs
//! strictly as untrusted text.

use serde_json::Value;

use crate::error::{WorkerError, WorkerResult};
use crate::task::{OutputFormat, TaskRequest, ValidationCheck, ValidationReport};

/// Result of running all configured static checks on a raw model response.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ValidationOutcome {
    pub normalized_output: Option<String>,
    pub report: ValidationReport,
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum FenceExtraction {
    NoFence,
    Extracted(String),
    UnclosedFence,
}

/// Non-executing validator enforcing structural, length, fence, and JSON schema checks.
pub struct OutputValidator;

impl OutputValidator {
    /// Pre-validates any caller-supplied JSON Schema before calling Ollama.
    pub fn validate_request_schema(request: &TaskRequest) -> WorkerResult<()> {
        if let Some(schema) = &request.output_expectation.json_schema {
            jsonschema::validator_for(schema).map_err(|err| WorkerError::InvalidInput {
                message: format!("invalid output_expectation.json_schema: {err}"),
            })?;
        }
        Ok(())
    }

    /// Runs static validation checks on `raw_output` and produces a [`ValidationOutcome`].
    #[must_use]
    pub fn validate(
        request: &TaskRequest,
        raw_output: &str,
        done_reason: Option<&str>,
        mut warnings: Vec<String>,
    ) -> ValidationOutcome {
        let mut checks = Vec::new();
        let trimmed = raw_output.trim();

        if trimmed.is_empty() {
            checks.push(ValidationCheck::fail(
                "non_empty",
                "model returned an empty response",
            ));
            return ValidationOutcome {
                normalized_output: None,
                report: ValidationReport {
                    passed: false,
                    extracted_code_block: false,
                    checks,
                    warnings,
                },
            };
        }
        checks.push(ValidationCheck::pass("non_empty"));

        if done_reason.is_some_and(|reason| reason.eq_ignore_ascii_case("length")) {
            checks.push(ValidationCheck::fail(
                "not_truncated",
                "generation stopped due to token limit (done_reason = length); output may be incomplete",
            ));
        } else {
            checks.push(ValidationCheck::pass("not_truncated"));
        }

        let effective_format = request
            .output_expectation
            .effective_format(request.task_type);
        let should_extract_fence = match effective_format {
            OutputFormat::Code => request.output_expectation.extract_code_block,
            OutputFormat::Json => true,
            OutputFormat::Text | OutputFormat::Auto => false,
        };

        let mut extracted_code_block = false;
        let candidate = if should_extract_fence {
            match extract_first_fenced_block(trimmed) {
                FenceExtraction::Extracted(inner) => {
                    extracted_code_block = true;
                    if inner.trim().is_empty() {
                        checks.push(ValidationCheck::fail(
                            "code_fence",
                            "extracted markdown code block is empty",
                        ));
                    } else {
                        checks.push(ValidationCheck::pass("code_fence"));
                    }
                    inner.trim().to_owned()
                }
                FenceExtraction::UnclosedFence => {
                    checks.push(ValidationCheck::fail(
                        "code_fence",
                        "malformed or unclosed markdown code fence in model output",
                    ));
                    trimmed.to_owned()
                }
                FenceExtraction::NoFence => trimmed.to_owned(),
            }
        } else {
            trimmed.to_owned()
        };

        if let Some(max_chars) = request.output_expectation.max_output_chars {
            let actual_chars = candidate.chars().count();
            if actual_chars > max_chars {
                checks.push(ValidationCheck::fail(
                    "max_length",
                    format!(
                        "output length ({actual_chars} chars) exceeds max_output_chars ({max_chars})"
                    ),
                ));
            } else {
                checks.push(ValidationCheck::pass("max_length"));
            }
        }

        let requires_json = effective_format == OutputFormat::Json
            || !request.output_expectation.required_fields.is_empty()
            || request.output_expectation.json_schema.is_some();

        if requires_json {
            validate_json_output(&candidate, request, &mut checks, &mut warnings);
        }

        let passed = checks.iter().all(|c| c.passed);
        ValidationOutcome {
            normalized_output: passed.then_some(candidate),
            report: ValidationReport {
                passed,
                extracted_code_block,
                checks,
                warnings,
            },
        }
    }
}

fn validate_json_output(
    candidate: &str,
    request: &TaskRequest,
    checks: &mut Vec<ValidationCheck>,
    warnings: &mut Vec<String>,
) {
    let parsed: Value = match serde_json::from_str(candidate) {
        Ok(val) => {
            checks.push(ValidationCheck::pass("valid_json"));
            val
        }
        Err(err) => {
            checks.push(ValidationCheck::fail(
                "valid_json",
                format!("output is not valid JSON: {err}"),
            ));
            return;
        }
    };

    if !request.output_expectation.required_fields.is_empty() {
        match parsed.as_object() {
            Some(obj) => {
                let missing: Vec<&str> = request
                    .output_expectation
                    .required_fields
                    .iter()
                    .map(String::as_str)
                    .filter(|field| !obj.contains_key(*field))
                    .collect();
                if missing.is_empty() {
                    checks.push(ValidationCheck::pass("required_fields"));
                } else {
                    checks.push(ValidationCheck::fail(
                        "required_fields",
                        format!("missing required JSON fields: {}", missing.join(", ")),
                    ));
                }
            }
            None => {
                checks.push(ValidationCheck::fail(
                    "required_fields",
                    "expected top-level JSON object to check required_fields",
                ));
            }
        }
    }

    if let Some(schema) = &request.output_expectation.json_schema {
        match jsonschema::validator_for(schema) {
            Ok(validator) => {
                let errors: Vec<String> = validator
                    .iter_errors(&parsed)
                    .take(3)
                    .map(|e| e.to_string())
                    .collect();
                if errors.is_empty() {
                    checks.push(ValidationCheck::pass("json_schema"));
                } else {
                    checks.push(ValidationCheck::fail("json_schema", errors.join("; ")));
                }
            }
            Err(err) => {
                warnings.push(format!("failed to compile json_schema: {err}"));
                checks.push(ValidationCheck::fail(
                    "json_schema",
                    format!("invalid json_schema: {err}"),
                ));
            }
        }
    }
}

/// Extracts the first markdown triple-backtick code block if present.
fn extract_first_fenced_block(text: &str) -> FenceExtraction {
    let mut lines = text.lines();
    let mut inside = false;
    let mut block_lines = Vec::new();

    for line in lines.by_ref() {
        let trimmed = line.trim_start();
        if !inside {
            if trimmed.starts_with("```") {
                inside = true;
            }
        } else if trimmed.starts_with("```") {
            return FenceExtraction::Extracted(block_lines.join("\n"));
        } else {
            block_lines.push(line);
        }
    }

    if inside {
        FenceExtraction::UnclosedFence
    } else {
        FenceExtraction::NoFence
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;
    use crate::task::TaskType;

    #[test]
    fn extracts_fenced_code_block_for_code_tasks() {
        let mut req = TaskRequest::new("t1", "qwen2.5-coder:3b", "Write is_prime");
        req.task_type = TaskType::GenerateFunction;

        let raw =
            "Here is the code:\n```python\ndef is_prime(n: int) -> bool:\n    return n > 1\n```\n";
        let outcome = OutputValidator::validate(&req, raw, Some("stop"), vec![]);
        assert!(outcome.report.passed);
        assert!(outcome.report.extracted_code_block);
        assert_eq!(
            outcome.normalized_output.as_deref(),
            Some("def is_prime(n: int) -> bool:\n    return n > 1")
        );
    }

    #[test]
    fn fails_on_unclosed_code_fence_and_length_truncation() {
        let mut req = TaskRequest::new("t2", "qwen2.5-coder:3b", "Write is_prime");
        req.task_type = TaskType::GenerateFunction;

        let unclosed = "```python\ndef is_prime(n):";
        let outcome = OutputValidator::validate(&req, unclosed, Some("length"), vec![]);
        assert!(!outcome.report.passed);
        assert!(outcome.normalized_output.is_none());
        assert!(
            outcome
                .report
                .checks
                .iter()
                .any(|c| c.name == "not_truncated" && !c.passed)
        );
        assert!(
            outcome
                .report
                .checks
                .iter()
                .any(|c| c.name == "code_fence" && !c.passed)
        );
    }

    #[test]
    fn enforces_max_output_chars_limit() {
        let mut req = TaskRequest::new("t3", "qwen2.5-coder:3b", "Explain briefly");
        req.output_expectation.max_output_chars = Some(10);

        let outcome =
            OutputValidator::validate(&req, "This explanation is too long", Some("stop"), vec![]);
        assert!(!outcome.report.passed);
        assert!(
            outcome
                .report
                .checks
                .iter()
                .any(|c| c.name == "max_length" && !c.passed)
        );
    }

    #[test]
    fn validates_json_required_fields_and_json_schema() {
        let mut req = TaskRequest::new("t4", "qwen2.5-coder:3b", "Emit summary JSON");
        req.task_type = TaskType::GenerateJson;
        req.output_expectation.required_fields = vec!["name".into(), "score".into()];
        req.output_expectation.json_schema = Some(json!({
            "type": "object",
            "properties": {
                "name": { "type": "string" },
                "score": { "type": "integer", "minimum": 0, "maximum": 100 }
            },
            "required": ["name", "score"],
            "additionalProperties": false
        }));

        OutputValidator::validate_request_schema(&req).expect("schema should compile");

        let ok_raw = "```json\n{\"name\": \"prime\", \"score\": 95}\n```";
        let ok_outcome = OutputValidator::validate(&req, ok_raw, Some("stop"), vec![]);
        assert!(ok_outcome.report.passed);
        assert!(ok_outcome.report.extracted_code_block);
        assert_eq!(
            ok_outcome.normalized_output.as_deref(),
            Some("{\"name\": \"prime\", \"score\": 95}")
        );

        let bad_schema_raw = "{\"name\": \"prime\", \"score\": 150}";
        let bad_outcome = OutputValidator::validate(&req, bad_schema_raw, Some("stop"), vec![]);
        assert!(!bad_outcome.report.passed);
        assert!(
            bad_outcome
                .report
                .checks
                .iter()
                .any(|c| c.name == "json_schema" && !c.passed)
        );

        let missing_field_raw = "{\"name\": \"prime\"}";
        let missing_outcome =
            OutputValidator::validate(&req, missing_field_raw, Some("stop"), vec![]);
        assert!(!missing_outcome.report.passed);
        assert!(
            missing_outcome
                .report
                .checks
                .iter()
                .any(|c| c.name == "required_fields" && !c.passed)
        );
    }
}
