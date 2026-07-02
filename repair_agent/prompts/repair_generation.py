REPAIR_GENERATION_PROMPT = """
You are an expert Java software engineer.

Generate the minimal repair required to fix the failing unit test.

Target:

{{analysis_result.target_to_repair}}

Root Cause:

{{analysis_result.root_cause_explanation}}

Target Source Code:

{{target_source_code}}

Failure Message:

{{test_document.errorMessage}}

Stack Trace:

{{test_document.stackTrace}}

Rules:

- Fix ONLY the identified bug.
- Preserve formatting.
- Do not modify unrelated code.
- Return ONLY the corrected code for the specified target lines.

Return ONLY JSON.

Do not include:
- Markdown
- ```json
- Explanations
- Additional text

{
    "generated_patch":"..."
}
"""