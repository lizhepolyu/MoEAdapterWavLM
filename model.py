'''
This is the ECAPA-TDNN model.
This model is modified and combined based on the following three projects:
  1. https://github.com/clovaai/voxceleb_trainer/issues/86
  2. https://github.com/lawlict/ECAPA-TDNN/blob/master/ecapa_tdnn.py
  3. https://github.com/speechbrain/speechbrain/blob/96077e9a1afff89d3f5ff47cab4bca0202770e4f/speechbrain/lobes/models/ECAPA_TDNN.py

'''

import math, torch, torchaudio
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import KMeans

from WavLM import WavLM, WavLMConfig
checkpoint = torch.load('WavLM-Large.pt')
cfg = WavLMConfig(checkpoint['cfg'])
model = WavLM(cfg)
model.load_state_dict(checkpoint['model'], strict=False)  


# --------------------
# 1. 定义 SpectralAdapter
# --------------------
class SpectralAdapter(nn.Module):
    """
    利用低秩矩阵对原始权重矩阵进行可学习的微调。
    """
    def __init__(self, initial_weight: torch.Tensor, rank: int = 256, adapter_rank: int = 16):
        """
        Args:
            initial_weight (torch.Tensor): 原始权重 (out_dim, in_dim).
            rank (int): 保留的主要奇异值数目 (SVD 截断).
            adapter_rank (int): 额外用于微调的低秩维度 (默认16).
        """
        super(SpectralAdapter, self).__init__()
        
        # 1) 对初始权重做 SVD
        assert rank <= min(initial_weight.size()), (
            f"Rank {rank} cannot exceed min dimension of weight {initial_weight.size()}."
        )
        U, S, Vh = torch.linalg.svd(initial_weight, full_matrices=False)
        
        # 2) 截断到指定 rank
        self.U1 = U[:, :rank].detach()       # (out_dim, rank)
        self.S = torch.diag(S[:rank]).detach()  # (rank, rank)
        self.Vh1 = Vh[:rank, :].detach()     # (rank, in_dim)
        
        # 3) 定义可学习的微调参数 (低秩修正)
        in_dim = initial_weight.shape[1]
        
        self.UA = nn.Parameter(torch.randn(in_dim, adapter_rank))   # (in_dim, adapter_rank)
        self.UB = nn.Parameter(torch.randn(adapter_rank, rank))     # (adapter_rank, rank)
        
        self.VhA = nn.Parameter(torch.randn(rank, adapter_rank))    # (rank, adapter_rank)
        self.VhB = nn.Parameter(torch.randn(adapter_rank, in_dim))  # (adapter_rank, in_dim)
        
        # 4) 参数初始化
        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.UA)
        nn.init.zeros_(self.UB)
        nn.init.kaiming_uniform_(self.VhA)
        nn.init.zeros_(self.VhB)
    
    def forward(self):
        """
        Returns:
            torch.Tensor: 通过低秩微调后的新权重 (out_dim, in_dim).
        """
        U_new = self.U1 + torch.matmul(self.UA, self.UB)       # (out_dim, rank)
        Vh_new = self.Vh1 + torch.matmul(self.VhA, self.VhB)   # (rank, in_dim)
        
        # 重建完整权重: W = U_new * S * Vh_new
        return torch.matmul(torch.matmul(U_new, self.S), Vh_new)


# --------------------
# 2. 将 SpectralAdapter 整合到您的 WavLMPtm
# --------------------
class WavLMPtm(nn.Module):
    """
    使用 SpectralAdapter 微调 WavLM，仅更新每层自注意力的 Q、K 投影矩阵。
    其他参数（除 prompt / gating / adapter 等）默认冻结。
    """
    def __init__(self):
        """
        Args:
            model: 预训练好的 WavLM 模型
            cfg:   配置对象 (假设包含 .encoder_layers, .normalize 等)
            rank:  SVD 截断的奇异值数
        """
        super(WavLMPtm, self).__init__()
        self.model = model

        # 1) 获取编码器层数，按需求初始化可学习的层权重
        num_layers = self.model.cfg.encoder_layers
        # 建立一个递减序列 [1.0, ..., 0.1]，再 softmax
        decreasing_values = torch.linspace(start=1.0, end=0.1, steps=num_layers+1)
        self.weights = nn.Parameter(torch.softmax(decreasing_values, dim=0))  # (num_layers,)

        # self.instance_norm = nn.InstanceNorm1d(1024)

        # 2) 为每一层的 Q、K 投影创建 SpectralAdapter (这里只演示 Q、K；V 不做适配)
#        self.spectral_adapters_q = nn.ModuleList()
#        self.spectral_adapters_k = nn.ModuleList()
#
#        for i in range(num_layers):
#            # 注意：这里的访问方式适用于 fairseq 版本的 wavlm
#            q_weight = self.model.encoder.layers[i].self_attn.q_proj.weight
#            k_weight = self.model.encoder.layers[i].self_attn.k_proj.weight
#
#            adapter_q = SpectralAdapter(q_weight).cuda()
#            adapter_k = SpectralAdapter(k_weight).cuda()
#
#            self.spectral_adapters_q.append(adapter_q)
#            self.spectral_adapters_k.append(adapter_k)

        # 3) 冻结除 spectral/prompt/adapter_experts/gating_network 外的所有参数
        for name, param in self.model.named_parameters():
            # 如果您的模型中 prompt、adapter_experts 等关键字不同，需自行更改
            if ('prompt' in name or
                'spectral' in name or
                'adapter_experts' in name or
                'gating_network' in name):
                param.requires_grad = True
            else:
                param.requires_grad = False

    def forward(self, x):

        # 1) 在 forward 中，将 SpectralAdapter 产生的新权重写回 Q、K
        num_layers = self.model.cfg.encoder_layers

#        for i in range(num_layers):
#            # 1) 普通计算图中: 调用 SpectralAdapter 的 forward，允许参数更新
#            q_weight_new = self.spectral_adapters_q[i]()  # 会追踪梯度
#            k_weight_new = self.spectral_adapters_k[i]()  # 会追踪梯度
#
#            # 2) 在 no_grad 下，将结果拷贝到原模型
#            with torch.no_grad():
#                self.model.encoder.layers[i].self_attn.q_proj.weight.copy_(q_weight_new)
#                self.model.encoder.layers[i].self_attn.k_proj.weight.copy_(k_weight_new)


        # 2) 可选归一化
        if cfg.normalize:
            x = F.layer_norm(x, x.shape)

        # 3) 提取多层特征
        #    Fairseq 中：extract_features 返回 (x, layer_results)
        #    layer_results 是 [(layer_out, attn_weight), (layer_out, attn_weight), ...]
        rep, layer_results = self.model.extract_features(
            x, 
            output_layer=self.model.cfg.encoder_layers, 
            ret_layer_results=True
        )[0]
        # 拆出所有层的输出 (batch_size, seq_len)
        layer_reps = [lr[0].transpose(0, 1) for lr in layer_results]  # 变为 (batch_size, seq_len, dim)

        # 4) 使用可学习的 self.weights 对各层输出加权求和
        #    self.weights.shape = (num_layers,)
        #    先 softmax 再相乘
        w_normalized = torch.softmax(self.weights, dim=0)
        weighted_sum = sum(w * rep for w, rep in zip(w_normalized, layer_reps))  # (batch_size, seq_len, dim)

        # 5) (batch_size, seq_len, dim) -> (batch_size, dim, seq_len)
        x = weighted_sum.permute(0, 2, 1)

        # x = self.instance_norm(x)  # 如有需要可启用

        return x


class SEModule(nn.Module):
    def __init__(self, channels, bottleneck=128):
        super(SEModule, self).__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(channels, bottleneck, kernel_size=1, padding=0),
            nn.ReLU(),
            # nn.BatchNorm1d(bottleneck), # I remove this layer
            nn.Conv1d(bottleneck, channels, kernel_size=1, padding=0),
            nn.Sigmoid(),
        )

    def forward(self, input):
        x = self.se(input)
        return input * x


class Bottle2neck(nn.Module):

    def __init__(self, inplanes, planes, kernel_size=None, dilation=None, scale=8):
        super(Bottle2neck, self).__init__()
        width = int(math.floor(planes / scale))
        self.conv1 = nn.Conv1d(inplanes, width * scale, kernel_size=1)
        self.bn1 = nn.BatchNorm1d(width * scale)
        self.nums = scale - 1
        convs = []
        bns = []
        num_pad = math.floor(kernel_size / 2) * dilation
        for i in range(self.nums):
            convs.append(nn.Conv1d(width, width, kernel_size=kernel_size, dilation=dilation, padding=num_pad))
            bns.append(nn.BatchNorm1d(width))
        self.convs = nn.ModuleList(convs)
        self.bns = nn.ModuleList(bns)
        self.conv3 = nn.Conv1d(width * scale, planes, kernel_size=1)
        self.bn3 = nn.BatchNorm1d(planes)
        self.relu = nn.ReLU()
        self.width = width
        self.se = SEModule(planes)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.relu(out)
        out = self.bn1(out)

        spx = torch.split(out, self.width, 1)
        for i in range(self.nums):
            if i == 0:
                sp = spx[i]
            else:
                sp = sp + spx[i]
            sp = self.convs[i](sp)
            sp = self.relu(sp)
            sp = self.bns[i](sp)
            if i == 0:
                out = sp
            else:
                out = torch.cat((out, sp), 1)
        out = torch.cat((out, spx[self.nums]), 1)

        out = self.conv3(out)
        out = self.relu(out)
        out = self.bn3(out)

        out = self.se(out)
        out += residual
        return out


class PreEmphasis(torch.nn.Module):

    def __init__(self, coef: float = 0.97):
        super().__init__()
        self.coef = coef
        self.register_buffer(
            'flipped_filter', torch.FloatTensor([-self.coef, 1.]).unsqueeze(0).unsqueeze(0)
        )

    def forward(self, input: torch.tensor) -> torch.tensor:
        input = input.unsqueeze(1)
        input = F.pad(input, (1, 0), 'reflect')
        return F.conv1d(input, self.flipped_filter).squeeze(1)


class FbankAug(nn.Module):

    def __init__(self, freq_mask_width=(0, 8), time_mask_width=(0, 10)):
        self.time_mask_width = time_mask_width
        self.freq_mask_width = freq_mask_width
        super().__init__()

    def mask_along_axis(self, x, dim):
        original_size = x.shape
        batch, fea, time = x.shape
        if dim == 1:
            D = fea
            width_range = self.freq_mask_width
        else:
            D = time
            width_range = self.time_mask_width

        mask_len = torch.randint(width_range[0], width_range[1], (batch, 1), device=x.device).unsqueeze(2)
        mask_pos = torch.randint(0, max(1, D - mask_len.max()), (batch, 1), device=x.device).unsqueeze(2)
        arange = torch.arange(D, device=x.device).view(1, 1, -1)
        mask = (mask_pos <= arange) * (arange < (mask_pos + mask_len))
        mask = mask.any(dim=1)

        if dim == 1:
            mask = mask.unsqueeze(2)
        else:
            mask = mask.unsqueeze(1)

        x = x.masked_fill_(mask, 0.0)
        return x.view(*original_size)

    def forward(self, x):
        x = self.mask_along_axis(x, dim=2)
        x = self.mask_along_axis(x, dim=1)
        return x


class ECAPA_TDNN(nn.Module):

    def __init__(self, C):
        super(ECAPA_TDNN, self).__init__()
        self.WavLMPtm = WavLMPtm()
        self.specaug = FbankAug()
        self.conv1 = nn.Conv1d(1024, C, kernel_size=5, stride=1, padding=2)
        self.relu = nn.ReLU()
        self.bn1 = nn.BatchNorm1d(C)
        self.layer1 = Bottle2neck(C, C, kernel_size=3, dilation=2, scale=8)
        self.layer2 = Bottle2neck(C, C, kernel_size=3, dilation=3, scale=8)
        self.layer3 = Bottle2neck(C, C, kernel_size=3, dilation=4, scale=8)
        # I fixed the shape of the output from MFA layer, that is close to the setting from ECAPA paper.
        self.layer4 = nn.Conv1d(3 * C, 1536, kernel_size=1)
        self.attention = nn.Sequential(
            nn.Conv1d(4608, 256, kernel_size=1),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Tanh(),  # I add this layer
            nn.Conv1d(256, 1536, kernel_size=1),
            nn.Softmax(dim=2),
        )
        self.bn5 = nn.BatchNorm1d(3072)
        self.fc6 = nn.Linear(3072, 192)
        self.bn6 = nn.BatchNorm1d(192)


    def forward(self, x, aug):

        x = self.WavLMPtm(x)

        if aug == True:
            x = self.specaug(x)

        x = self.conv1(x)
        x = self.relu(x)
        x = self.bn1(x)

        x1 = self.layer1(x)
        x2 = self.layer2(x + x1)
        x3 = self.layer3(x + x1 + x2)

        x = self.layer4(torch.cat((x1, x2, x3), dim=1))
        x = self.relu(x)

        t = x.size()[-1]

        global_x = torch.cat((x, torch.mean(x, dim=2, keepdim=True).repeat(1, 1, t),
                              torch.sqrt(torch.var(x, dim=2, keepdim=True).clamp(min=1e-4)).repeat(1, 1, t)), dim=1)

        w = self.attention(global_x)

        mu = torch.sum(x * w, dim=2)
        sg = torch.sqrt((torch.sum((x ** 2) * w, dim=2) - mu ** 2).clamp(min=1e-4))

        x = torch.cat((mu, sg), 1)
        x = self.bn5(x)
        x = self.fc6(x)
        x = self.bn6(x)

        return x