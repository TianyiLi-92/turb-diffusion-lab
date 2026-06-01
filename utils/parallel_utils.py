
def reverse_gpu_parallel_indices(num_data, num_gpus, batch_size_per_gpu):
    num_data_per_gpu = num_data//num_gpus
    gpu_wise_indices = [
        [i*num_data_per_gpu+j for j in range(num_data_per_gpu)]
        for i in range(num_gpus)
    ]

    all_gather_order = []
    for i in range(0, len(gpu_wise_indices[0]), batch_size_per_gpu):
        for gpu_indices in gpu_wise_indices:
            all_gather_order.extend(gpu_indices[i:i+batch_size_per_gpu])

    reverse_order = [0] * len(all_gather_order)
    for original_idx, transformed_idx in enumerate(all_gather_order):
        reverse_order[transformed_idx] = original_idx

    return reverse_order
