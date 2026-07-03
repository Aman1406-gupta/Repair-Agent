PULL_REQUEST_PROMPT = """
You are an experienced software engineer responsible for preparing a GitHub Pull Request description.

The repair process has completed successfully.

Your task is to generate a concise, professional Pull Request description summarizing the repairs that were generated.

Use ONLY the provided information.
Do NOT invent implementation details.
Do NOT mention changes that are not reflected in the generated repair patches.
If multiple repairs address the same underlying issue, summarize them instead of repeating them.

Each repair item contains:
- Test ID
- Test name
- Test file
- Root cause
- Repair target (TEST or SERVICE)
- Service file
- Generated repair patch
- Validation result
- Relevant git diff

Repair Items

{{repair_items}}

Return ONLY markdown.

Use the following format exactly.

# Summary

Briefly describe what was repaired and why.

# Root Cause

Summarize the identified root cause(s).

# Changes Made

Describe the generated repairs at a high level.

Do NOT include code snippets.

Group similar changes together.

# Affected Tests

| Test ID | Test Name | Test File
|----------|-----------|----------
| ... | ... | ... |

# Validation

Summarize the validation results.

If every repair passed validation, state that all generated repairs passed validation.

Otherwise mention which repairs require further investigation.

# Notes

Mention that the generated repairs should be reviewed before merging.
"""