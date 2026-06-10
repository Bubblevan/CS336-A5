"""
core/batching.py

iterate_batches(dataset, batch_size, shuffle) → generator of dict batches
    Yield batches of {input_ids, labels, ...} from a list of examples.

split_microbatches(batch, grad_accum_steps) → list of dict
    Split a batch into micro-batches for gradient accumulation.

collate_fn(batch_examples) → dict[str, torch.Tensor]
    Collate a list of examples into a batched dict with padding.
"""
