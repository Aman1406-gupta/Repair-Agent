PULL_REQUEST_PROMPT = """
Generate a professional GitHub Pull Request description.
Keep the description concise.
Do not invent implementation details.
Mention only changes actually present in generated patch

Test ID:

{{test_document.testID}}

File Modified:

{{target_file_path}}

Root Cause:

{{analysis_result.root_cause_explanation}}

Validation Result:

{{validation_passed}}

Return ONLY markdown.

Format:

## Summary

## Root Cause

## Changes Made

## Validation
"""