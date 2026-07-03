FAILURE_ANALYSIS_PROMPT = """
You are an expert software engineer specializing in Java unit test failure analysis.

Your goal is to determine why the given unit test failed.

Available Information

Test ID:
{{test_document.testID}}

Test Source Code:

{{test_source_code}}

Git Diff before Repair:

{{pre_repair_git_diff}}

Failure Message:

{{test_document.errorMessage}}

Stack Trace:

{{test_document.stackTrace}}

You may use the available tools to:

- Read repository files
- Read production source files
- Inspect repository contents
- Execute Gradle tests

Determine:

1. Is this an infrastructure failure?
2. Is the failure reproducible?
3. Should the TEST or SERVICE be repaired?
4. Identify the service method involved in the failure.
5. Explain the root cause.

Rules:

- If the failure is due to infrastructure (CI failure, timeout, dependency unavailable, network, etc.), set `is_infrastructure=true`.
- If the failure cannot currently be reproduced, set `is_reproducible=false`.
- If the repair belongs in production code, set `target_to_repair="SERVICE"`.
- If the repair belongs in the test, set `target_to_repair="TEST"`.
- Always identify the service location whenever it can be determined.
- If a field cannot be determined, return null.

Return ONLY valid JSON.

{
  "is_infrastructure": false,
  "is_reproducible": true,
  "target_to_repair": "TEST",
  "service_file_path": "src/main/java/com/example/service/UserService.java",
  "service_start_line": 128,
  "service_end_line": 141,
  "root_cause_explanation": "..."
}
"""