REPAIR_GENERATION_PROMPT = """
You are an expert Java software engineer.

Your goal is to generate the minimal repair code for the given test code in identified line range.

Inputs

Test ID:

{{testID}}

Test Name:

{{methodName}}

Target to Repair:

{{target_to_repair}}

Test source code:

{{test_source_code}}

Root cause:

{{root_cause}}

Error Message:

{{errorMessage}}

Stack Trace:

{{stackTrace}}

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