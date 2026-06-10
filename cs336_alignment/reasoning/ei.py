"""
reasoning/ei.py

expert_iteration_loop(args)
    Core loop for Expert Iteration:
    1. Generate rollouts from current policy
    2. Filter correct trajectories via reward_fn
    3. SFT on filtered data
    4. Repeat

convert_reasoning_records_to_sft_jsonl(source_path, output_path, limit, seed)
    Convert reasoning traces to SFT {prompt, response} format.
"""
