SKELETON_PULL_REQUEST_PROMPT = """
You are an experienced software engineer responsible for preparing a GitHub Pull Request description.

A repair was intentionally NOT generated because failure analysis indicates that the failing tests are likely caused by an issue in the production/service code.

Your task is to generate a professional skeleton Pull Request description that notifies the service owner about the suspected issue.

Use ONLY the information provided.
Do NOT invent implementation details.
Do NOT propose a code fix.
Do NOT speculate beyond the provided failure analyses.

You are provided a list of affected test failures. Every item corresponds to one failing test that points to the SAME service file.

For each affected item, the following information is available:
- Test ID
- Test Name
- Test source code
- Root cause
- Suspected service file path
- Suspected service start line
- Suspected service end line

Requirements:

1. Identify the common service file responsible for the failures.
2. Mention the suspected line range(s) in the service file.
3. Summarize the overall issue affecting the service.
4. List every affected test.
5. If multiple root causes are similar, summarize them instead of repeating them.
6. Mention that this PR intentionally contains no code changes and serves only as a notification for the service owner.

Affected Items

{{affected_items}}

Return ONLY markdown.

Use the following format exactly.

# Summary

Briefly explain that multiple unit test failures appear to originate from the same production/service file and therefore no automated repair was generated.

# Affected Service

**Service File**

...

**Suspected Line Range(s)**

- start-end
- start-end

# Affected Tests

| Test ID | Test Name | Test File |
|----------|-------------|-----------|
| ... | ... | ... |

# Root Cause Summary

Provide a concise summary of the suspected production issue.
If there are multiple distinct causes, list them as bullet points.

# Action Required

Request the service owner to review the identified service implementation, particularly the listed line ranges, verify the reported root causes, and implement the appropriate fix.

# Notes

- This Pull Request intentionally contains no source code modifications.
- It has been generated to notify the service owner that the identified production code is suspected to be responsible for the listed failing unit tests.
"""