"""
eval/parsers.py

parse_mmlu_response(mmlu_example, model_output) → str | None
    Extract answer letter (A/B/C/D) from model output.

parse_gsm8k_response(model_output) → str | None
    Extract final numeric answer (last number in the text).
"""
