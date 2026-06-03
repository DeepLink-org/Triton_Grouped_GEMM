import random
import torch


def gmm(a, b, batch_sizes, trans_b=False):
    batch_sizes = batch_sizes.numpy()

    out = []
    start = 0
    for i, size in enumerate(batch_sizes):
        rhs = b[i, :, :].t() if trans_b else b[i, :, :]
        out.append(a[start:start + size, :] @ rhs)
        start += size
    return torch.cat(out)

def allclose(x, y, pct=2.0):
    mask = torch.isclose(x, y, rtol=1e-5)
    pct_diff = (mask.numel() - mask.sum()) / mask.numel() * 100
    if pct_diff > pct:
        print(x[torch.logical_not(mask)], y[torch.logical_not(mask)])
        print("{:.2f}% of values not close.".format(pct_diff))
        return False
    return True

def generate_random_list(length, total_sum):
    # 生成一个长度为length的列表，元素之和为total_sum
    # 先生成一个平均分配的列表
    avg = total_sum // length
    lst = [0] * length
    # 随机调整数值，确保总和不变
    for i in range(length):
        # 随机选择两个不同的位置
        lst[i] = random.randint(0, 2*int(avg))
    ratio = total_sum / sum(lst)
    lst = [int(x * ratio) for x in lst]

    diff = total_sum - sum(lst)
    lst[-1] += diff
    return lst

def row_max_normalization(tensor):
    row_maxs = tensor.abs().max(dim = -1).values + 1e-9
    tensor_normalized = tensor / row_maxs.unsqueeze(dim = -1)
    return tensor_normalized
