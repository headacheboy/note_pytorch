# pytorch nn.DataParallel will not split list, dict or other data automatically. 
# Therefore, to use list data with multi gpus (i.e. list[i] correspond to batch[i]), we can add a torch.arange(batch_size).unsqueeze(1) to index batch number

def check(a, b):
    # to check a == b
    pass

def multi_gpu_use(list_data, tensor_data, index):
    # index: torch.arange(batch_size).unsqueeze(1), and it will be automatically split by pytorch when using multi-gpus
    # tensor_data: [batch / gpu_num, ...]
    # list_data: [batch, ...]
    batch_size_single_gpu = tensor_data.size(0)
    for i in range(batch_size_single_gpu):
        assert check(list_data[index[i]], tensor_data)
