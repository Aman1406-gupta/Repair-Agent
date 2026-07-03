REPAIR_GENERATION_PROMPT = """
You are an expert Java software engineer.

Your goal is to generate the exact replacement code for the identified line range.

Inputs

Test ID:

{{test_document.testID}}

Test Name:

{{test_document.methodName}}

Target to Repair:

{{target_to_repair}}

Test source code:

{{test_source_code}}

Service source code:

{{service_source_code}}

Root cause:

{{root_cause}}

Error Message:

{{test_document.errorMessage}}

Stack Trace:

{{test_document.stackTrace}}

Pre Repair Git Diff:

{{pre_repair_git_diff}}

Requirements

- Modify ONLY the target to repair.
- Return valid Java code only.
- Preserve formatting.
- Do not include markdown.
- Do not explain the repair.
- Do not regenerate the whole file.
- Do not modify unrelated code.
- Repair only the test.

Return ONLY the replacement code JSON.

{
    "generated_patch":"..."
}
"""