from turtle import back
from unicodedata import bidirectional
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class BiLSTMLayer(nn.Module):
    """
    多层双向LSTM层，用于将token序列编码为语句表示
    Args:
        embedding_dim (int): 输入嵌入维度
        hidden_dim (int): 隐藏层维度
        num_layers (int): LSTM层数，默认为2
    """
    def __init__(self, embedding_dim: int, hidden_dim: int, num_layers: int = 2):
        super().__init__()
        self.num_layers = num_layers
        # 双向LSTM层，输入维度为embedding_dim，输出维度为hidden_dim，层数为num_layers
        self.bilstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            bidirectional=True,
            batch_first=True
        )
        # 投影层，将双向LSTM的输出映射到embedding_dim维度
        self.projection = nn.Sequential(
            nn.Linear(2 * embedding_dim, embedding_dim),
            nn.Tanh()
        )

    """
    前向传播：处理token序列生成语句表示
    Args:
        x (torch.Tensor): 输入token序列，形状为(batch_size, seq_len, embedding_dim)
        token_mask (torch.Tensor): token掩码，形状为(batch_size, seq_len)
    Returns:
        torch.Tensor: 语句表示，形状为(batch_size, embedding_dim)
    """
    def forward(self, x: torch.Tensor, token_mask: torch.Tensor) -> torch.Tensor:
        # 输入形状为(batch_size, M, N, d)
        batch_size, M, N, d = x.shape
        # 将输入展平为(batch_size * M, N, d)
        x_flat = x.view(-1, N, d)
        # 计算每个序列的有效长度，形状为(batch_size * M)
        seq_lens = (~token_mask).sum(dim=-1).view(-1).cpu()
        # 使用pack_padded_sequence和pad_packed_sequence处理变长序列
        packed_x = pack_padded_sequence(x_flat, seq_lens, batch_first=True, enforce_sorted=False)
        packed_out, _ = self.bilstm(packed_x)
        lstm_out, _ = pad_packed_sequence(packed_out, batch_first=True, total_length=N)
        lstm_out = lstm_out.view(batch_size, M, N, 2 * (d//2))
        # 提取每个序列的最后一个有效token的隐藏状态
        last_indices = (seq_lens.view(batch_size, M) - 1).clamp(min=0)
        last_hidden = torch.gather(lstm_out, dim=2,
                                    index=last_indices.view(batch_size, M, 1, 1).repeat(1, 1, 1, d)).squeeze(2)
        # 投影到embedding_dim维度
        return self.projection(last_hidden)

class TransformerLayer(nn.Module):
    """
    单层Transformer编码器，用于将语句表示编码为函数表示
    Args:
        d_model (int): 输入嵌入维度
        nhead (int): 多头注意力头数，默认为6
        dim_feedforward (int): 前馈网络隐藏层维度，默认为512
    """
    def __init__(self, d_model: int, nhead: int = 6, dim_feedforward: int = 512):
        super().__init__()
        # 位置嵌入层，用于将位置信息加入语句表示
        self.pos_embedding = nn.Embedding(num_embeddings=30, embedding_dim=d_model)
        # Transformer编码器层，输入维度为d_model，多头注意力头数为nhead，前馈网络隐藏层维度为dim_feedforward
        self.transformer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            batch_first=True
        )
        # 层归一化层，用于规范化输入
        self.norm = nn.LayerNorm(d_model)

    """
    前向传播：处理语句劽生成函数表示
    Args:
        x (torch.Tensor): 输入语句表示，形状为(batch_size, M, d_model)
        statement_mask (torch.Tensor): 语句掩码，形状为(batch_size, M)
    """
    def forward(self, x: torch.Tensor, statement_mask: torch.Tensor) -> torch.Tensor:
        # 输入形状为(batch_size, M, d)
        batch_size, M, _ = x.shape
        # 生成位置索引，形状为(batch_size, M)
        pos_indices = torch.arange(M, device=x.device).repeat(batch_size, 1)
        # 生成位置嵌入，形状为(batch_size, M, d)
        pos_emb = self.pos_embedding(pos_indices)
        # 将位置嵌入加到语句表示上，形状为(batch_size, M, d)
        x_with_pos = self.norm(x + pos_emb)
        # 生成注意力掩码，形状为(batch_size, M, M)
        attn_mask = statement_mask.unsqueeze(1).repeat(1, M, 1)
        # 使用Transformer编码器处理语句表示，形状为(batch_size, M, d)
        x_encoded = self.transformer(x_with_pos, src_key_padding_mask=~attn_mask)
        # 生成有效语句掩码，形状为(batch_size, M, 1)
        valid_mask = ~statement_mask.unsqueeze(-1)
        # 计算有效语句的表示，形状为(batch_size, d)
        sum_encoded = (x_encoded * valid_mask).sum(dim=1)
        # 计算有效语句的数量，形状为(batch_size)
        count_valid = valid_mask.sum(dim=1)
        # 计算函数表示，形状为(batch_size, d)
        return sum_encoded / count_valid.clamp(min=1)

class HLTModel(nn.Module):
    """
    完整HLT模型：组合BiLSTM和Transformer的分层漏洞检测模型
    Args:
        vocab_size (int): 词汇表大小
        embedding_dim (int): 嵌入维度，默认为128
        num_layers (int): LSTM层数，默认为2
    """
    def __init__(self, vocab_size: int, embedding_dim: int = 128, num_layers: int = 2):
        super().__init__()
        # 嵌入层，用于将token映射为嵌入向量
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=0
        )
        # 双向LSTM层，用于将token序列编码为语句表示
        self.bilstm = BiLSTMLayer(
            embedding_dim=embedding_dim,
            hidden_dim=embedding_dim//2,
            num_layers=num_layers
        )
        # Transformer层，用于将语句表示编码为函数表示
        self.transformer = TransformerLayer(
            d_model=embedding_dim,
            nhead=6,
            dim_feedforward=512
        )
        # 分类器，用于将函数表示映射为漏洞检测结果
        self.classifier = nn.Linear(embedding_dim, 2)

    """
    完整前向传播流程
    Args:
        token_ids (torch.Tensor): 输入token序列，形状为(batch_size, seq_len)
        token_mask (torch.Tensor): token掩码，形状为(batch_size, seq_len)
        statement_mask (torch.Tensor): 语句掩码，形状为(batch_size, M)
    """
    def forward(self, token_ids: torch.Tensor, token_mask: torch.Tensor, statement_mask: torch.Tensor) -> torch.Tensor:
        # 输入形状为(batch_size, seq_len)
        x_emb = self.embedding(token_ids)
        # 输入形状为(batch_size, seq_len, embedding_dim)
        stmt_expr = self.bilstm(x_emb, token_mask)
        # 输入形状为(batch_size, M, embedding_dim)
        func_expr = self.transformer(stmt_expr, statement_mask)
        # 输入形状为(batch_size, embedding_dim)
        return self.classifier(func_expr)