"""
assistant/packing.py

get_packed_sft_dataset(tokenizer, dataset_path, seq_length, shuffle)
    → list of {"input_ids": Tensor, "labels": Tensor}

    Pack multiple SFT examples into fixed-length sequences,
    using EOS as separator between examples.
"""
