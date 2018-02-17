import torch
import torch.nn as nn
import torch.nn.init as init
from transformer.Modules import BottleLinear as Linear
from transformer.Modules import ScaledDotProductAttention
#from transformer.Modules import BottleLayerNormalization as LayerNormalization
from transformer.Modules import LayerNormalization

class MultiHeadAttention(nn.Module):
    """多头注意力模块
    """
    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        """初始化多头
        
        Arguments:
            n_head {int} -- 头的数量
            d_model {int} -- 模型总维度
            d_k {int} -- Query和Key分别的子头维度
            d_v {int} -- Value的子头维度
    
        """
        super(MultiHeadAttention, self).__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Parameter(torch.FloatTensor(n_head, d_model, d_k))
        self.w_ks = nn.Parameter(torch.FloatTensor(n_head, d_model, d_k))
        self.w_vs = nn.Parameter(torch.FloatTensor(n_head, d_model, d_v))

        self.attention = ScaledDotProductAttention(d_model)
        self.layer_norm = LayerNormalization(d_model)
        self.proj = Linear(n_head*d_v, d_model)

        self.dropout = nn.Dropout(dropout)

        init.xavier_normal(self.w_qs)
        init.xavier_normal(self.w_ks)
        init.xavier_normal(self.w_vs)

    def forward(self, q, k, v, attn_mask=None):
        """[summary]
        
        Arguments:
            q {Tensor(batch_size, seq_len, d_model)} -- Query
            k {Tensor(batch_size, seq_len, d_model)} -- Key
            v {Tensor(batch_size, seq_len, d_model)} -- Value
        
        Returns:
            output -- Value 和 注意力分数加权后结果
            attn -- 注意力分数
        """

        d_k, d_v = self.d_k, self.d_v
        n_head = self.n_head

        residual = q

        mb_size, len_q, d_model = q.size()
        mb_size, len_k, d_model = k.size()
        mb_size, len_v, d_model = v.size()

        # treat as a (n_head) size batch
        q_s = q.repeat(n_head, 1, 1).view(n_head, -1, d_model) # n_head x (mb_size*len_q) x d_model
        k_s = k.repeat(n_head, 1, 1).view(n_head, -1, d_model) # n_head x (mb_size*len_k) x d_model
        v_s = v.repeat(n_head, 1, 1).view(n_head, -1, d_model) # n_head x (mb_size*len_v) x d_model

        # treat the result as a (n_head * mb_size) size batch
        q_s = torch.bmm(q_s, self.w_qs).view(-1, len_q, d_k)   # (n_head*mb_size) x len_q x d_k
        k_s = torch.bmm(k_s, self.w_ks).view(-1, len_k, d_k)   # (n_head*mb_size) x len_k x d_k
        v_s = torch.bmm(v_s, self.w_vs).view(-1, len_v, d_v)   # (n_head*mb_size) x len_v x d_v

        # 执行注意力, result size = (n_head * mb_size) x len_q x d_v
        if attn_mask==None:
            outputs, attns = self.attention(q_s, k_s, v_s)
        else:
            outputs, attns = self.attention(q_s, k_s, v_s, attn_mask=attn_mask.repeat(n_head, 1, 1))
        
        # back to original mb_size batch, result size = mb_size x len_q x (n_head*d_v)
        outputs = torch.cat(torch.split(outputs, mb_size, dim=0), dim=-1) 

        # project back to residual size
        outputs = self.proj(outputs)
        outputs = self.dropout(outputs)

        return self.layer_norm(outputs + residual), attns

class PositionwiseFeedForward(nn.Module):
    ''' 双层的位置前馈网络，可以写成下x*W，也可以写成conv1d，
        同时执行了res和layernorm和dropout 
    '''

    def __init__(self, d_hid, d_inner_hid, dropout=0.1):
        """[summary]
        
        Arguments:
            d_hid {int} -- 输出维度，等于输入维度
            d_inner_hid {int} -- 中间隐藏层维度，一般比输入大
        
        """
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Conv1d(d_hid, d_inner_hid, 1) # position-wise
        self.w_2 = nn.Conv1d(d_inner_hid, d_hid, 1) # position-wise
        self.layer_norm = LayerNormalization(d_hid)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, x):
        residual = x
        output = self.relu(self.w_1(x.transpose(1, 2)))
        output = self.w_2(output).transpose(2, 1)
        output = self.dropout(output)
        return self.layer_norm(output + residual)
