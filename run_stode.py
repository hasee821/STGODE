import os
import random
import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR, OneCycleLR
import time
from tqdm import tqdm
from loguru import logger

from args import args
from model import ODEGCN
from utils import generate_dataset, read_data, get_normalized_adj
from eval import masked_mae_np, masked_mape_np, masked_rmse_np

#DTW距离是一种时间序列相似度度量方法，它可以在时间序列中存在缩放、平移、略微变形的情况下，仍然能够准确地计算时间序列之间的相似度。
'''
dtw距离计算方法：DTW距离的计算方法如下：
首先，我们有两个时间序列，分别是序列A和序列B，其中序列A的长度为n，序列B的长度为m。

创建一个n行m列的矩阵，称为DTW矩阵。矩阵的每个元素表示在对齐序列A的第i个元素和序列B的第j个元素时的距离。

初始化DTW矩阵的第一行和第一列。第一行和第一列的元素表示对齐序列A或序列B的前缀子序列时的距离。可以使用欧氏距离或其他距离度量方法来计算这些初始值。

从DTW矩阵的第二行和第二列开始，逐行逐列计算矩阵中的每个元素。对于矩阵中的每个元素(i, j)，计算以下三个相邻元素的最小值：

(i-1, j)：表示对齐序列A的第i-1个元素和序列B的第j个元素时的距离。
(i, j-1)：表示对齐序列A的第i个元素和序列B的第j-1个元素时的距离。
(i-1, j-1)：表示对齐序列A的第i-1个元素和序列B的第j-1个元素时的距离。
然后，将这三个值与对齐序列A的第i个元素和序列B的第j个元素之间的距离相加，得到当前元素(i, j)的距离。

继续计算DTW矩阵的剩余元素，直到计算完最后一个元素(n, m)。

最后，DTW矩阵的右下角元素即为序列A和序列B之间的DTW距离。
'''
## torch.unsqueeze(input, dim, out=None) → Tensor:返回一个新的张量，对输入的制定位置插入维度 1
## torch.mul(input, other, out=None) → Tensor:对输入张量input逐元素相乘，与*号等价
## torch.contiguous() → Tensor:返回一个内存连续的有相同数据的张量,防止内存不连续导致某些操作如view()等不支持
## torch.clamp(input, min, max, out=None) → Tensor:将输入input张量每个元素的夹紧到区间 [min, max]，超过范围的截断为min或max，并返回结果到一个新张量
## torch.einsum(equation, *operands) → Tensor:根据方程式计算张量的乘积

'''
torch.einsum是PyTorch库中的一个函数，它提供了一种执行Einstein summation convention的方法。
Einstein summation convention是一种简化多维数组（如矩阵）操作的表示方法，它可以用来描述各种复杂的数组操作，包括矩阵的乘法、转置、迹、对角线等。

torch.einsum函数的参数是一个表示操作的字符串和一个或多个张量。字符串中的每个字符代表一个维度，字符的顺序表示张量的形状。
例如，字符串"ij"表示一个二维张量，其中"i"是第一维，"j"是第二维。字符串中的逗号用来分隔不同的张量，箭头"->"用来指示输出的形状。

equation规则
规则一，equation 箭头左边，在不同输入之间重复出现的索引表示，输入张量沿着该维度做乘法操作，比如还是以上面矩阵乘法为例， "ik,kj->ij"，k 在输入中重复出现，所以就是把 a 和 b 沿着 k 这个维度作相乘操作；
规则二，只出现在 equation 箭头左边的索引，表示中间计算结果需要在这个维度上求和；
规则三，equation 箭头右边的索引顺序可以是任意的，比如上面的 "ik,kj->ij" 如果写成 "ik,kj->ji"，那么就是返回输出结果的转置，用户只需要定义好索引的顺序，转置操作会在 einsum 内部完成。
'''




def train(loader, model, optimizer, criterion, device):
    batch_loss = 0
    accu_step = 4                    # 梯度累积,每4个batch更新一次梯度,减小显存占用，相当于batch_size*4
    for idx, (inputs, targets) in enumerate(tqdm(loader)):
        model.train()
        inputs = inputs.to(device)
        targets = targets.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        if(idx+1) % accu_step == 0:
            optimizer.step()
            optimizer.zero_grad()
        batch_loss += loss.detach().cpu().item() 
    return batch_loss / (idx + 1)


@torch.no_grad()
def eval(loader, model, std, mean, device):
    batch_rmse_loss = 0  
    batch_mae_loss = 0
    batch_mape_loss = 0
    for idx, (inputs, targets) in enumerate(tqdm(loader)):
        model.eval()

        inputs = inputs.to(device)
        targets = targets.to(device)
        output = model(inputs)
        
        out_unnorm = output.detach().cpu().numpy()*std + mean
        target_unnorm = targets.detach().cpu().numpy()*std + mean

        mae_loss = masked_mae_np(target_unnorm, out_unnorm, 0)
        rmse_loss = masked_rmse_np(target_unnorm, out_unnorm, 0)
        mape_loss = masked_mape_np(target_unnorm, out_unnorm, 0)
        batch_rmse_loss += rmse_loss
        batch_mae_loss += mae_loss
        batch_mape_loss += mape_loss

    return batch_rmse_loss / (idx + 1), batch_mae_loss / (idx + 1), batch_mape_loss / (idx + 1)


def main(args):
    # random seed
    seed = 2
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)

    device = torch.device('cuda:'+str(args.num_gpu)) if torch.cuda.is_available() else torch.device('cpu')

    if args.log:
        logger.add('log_{time}.log')
    options = vars(args)
    if args.log:
        logger.info(options)
    else:
        print(options)

    data, mean, std, dtw_matrix, sp_matrix = read_data(args)
    train_loader, valid_loader, test_loader = generate_dataset(data, args)
    A_sp_wave = get_normalized_adj(sp_matrix).to(device)
    A_se_wave = get_normalized_adj(dtw_matrix).to(device)

    net = ODEGCN(num_nodes=data.shape[1], 
                num_features=data.shape[2], 
                num_timesteps_input=args.his_length, 
                num_timesteps_output=args.pred_length, 
                A_sp_hat=A_sp_wave, 
                A_se_hat=A_se_wave)
    net = net.to(device)
    lr = args.lr
    optimizer = torch.optim.AdamW(net.parameters(), lr=lr)
    criterion = nn.SmoothL1Loss()

    best_valid_rmse = 1000 
    scheduler = StepLR(optimizer, step_size=50, gamma=0.5)

    for epoch in range(1, args.epochs+1):
        print("=====Epoch {}=====".format(epoch))
        print('Training...')
        start_time=time.time()
        loss = train(train_loader, net, optimizer, criterion, device)
        print(time.time()-start_time)
        print('Evaluating...')
        train_rmse, train_mae, train_mape = eval(train_loader, net, std, mean, device)
        valid_rmse, valid_mae, valid_mape = eval(valid_loader, net, std, mean, device)
        print(time.time()-start_time)

        if valid_rmse < best_valid_rmse:
            best_valid_rmse = valid_rmse
            print('New best results!')
            torch.save(net.state_dict(), f'net_params_{args.filename}_{args.num_gpu}.pkl')

        if args.log:
            logger.info(f'\n##on train data## loss: {loss}, \n' + 
                        f'##on train data## rmse loss: {train_rmse}, mae loss: {train_mae}, mape loss: {train_mape}\n' +
                        f'##on valid data## rmse loss: {valid_rmse}, mae loss: {valid_mae}, mape loss: {valid_mape}\n')
        else:
            print(f'\n##on train data## loss: {loss}, \n' + 
                f'##on train data## rmse loss: {train_rmse}, mae loss: {train_mae}, mape loss: {train_mape}\n' +
                f'##on valid data## rmse loss: {valid_rmse}, mae loss: {valid_mae}, mape loss: {valid_mape}\n')
        
        scheduler.step()

    net.load_state_dict(torch.load(f'net_params_{args.filename}_{args.num_gpu}.pkl'))
    test_rmse, test_mae, test_mape = eval(test_loader, net, std, mean, device)
    print(f'##on test data## rmse loss: {test_rmse}, mae loss: {test_mae}, mape loss: {test_mape}')


if __name__ == '__main__':
    main(args)
