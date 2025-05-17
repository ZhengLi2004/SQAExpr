from nt import lseek
import torch
from models.hlt_model import HLTModel
from typing import List, Tuple

"""
加载预训练的HLT模型
Args:
    vocab_size (int): 词汇表大小
    embedding_dim (int): 嵌入维度，默认为128
    num_layers (int): LSTM层数，默认为2
    checkpoint_path (str): 检查点路径，默认为"checkpoints/hlt_model.pth"
"""
def load_pretrained_model(
    vocab_size: int,
    embedding_dim: int = 128,
    num_layers: int = 2,
    checkpoint_path: str = "checkpoints/hlt_model.pth"
) -> HLTModel:
    model = HLTModel(
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        num_layers=num_layers
    )
    model.load_state_dict(torch.load(checkpoint_path))
    model.eval()
    return model

"""
预处理代码样本，生成模型所需的输入张量
Args:
    code_samples (List[List[List[str]]]): 代码样本，形状为(batch_size, M, N)，其中M为语句数，N为语句中的token数
    vocab (dict): 词汇表，用于将token映射为索引
    max_statements (int): 最大语句数，默认为30
    max_tokens_per_statement (int): 最大token数，默认为30
"""
def preprocess_code(
    code_samples: List[List[List[str]]],
    vocab: dict,
    max_statements: int = 30,
    max_tokens_per_statement: int = 30
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    batch_size = len(code_samples)
    token_ids = torch.zeros((batch_size, max_statements * max_tokens_per_statement), dtype=torch.long)
    token_mask = torch.ones((batch_size, max_statements * max_tokens_per_statement), dtype=torch.bool)
    statement_mask = torch.ones((batch_size, max_statements), dtype=torch.bool)

    for i, sample in enumerate(code_samples):
        statements = sample[:max_statements] + [[]] * max(0, max_statements - len(sample))

        for stmt_idx, stmt_tokens in enumerate(statements):
            tokens = stmt_tokens[:max_tokens_per_statement] + ['<pad>'] * max(0, max_tokens_per_statement - len(stmt_tokens))
            start = stmt_idx * max_tokens_per_statement
            end = start + max_tokens_per_statement
            token_ids[i, start:end] = torch.tensor([vocab.get(token, vocab['<unk>']) for token in tokens], dtype=torch.long)
            token_mask[i, start:end] = torch.tensor([t == '<pad>' for token in tokens], dtype=torch.bool)

            if len(stmt_tokens) == 0:
                statement_mask[i, stmt_idx] = True
            else:
                statement_mask[i, stmt_idx] = False

            return token_ids, token_mask, statement_mask

"""
对代码样本进行漏洞检测推理
Args:
    model (HLTModel): 预训练的HLT模型
    code_samples (List[List[List[str]]]): 代码样本，形状为(batch_size, M, N)，其中M为语句数，N为语句中的token数
    vocab (dict): 词汇表，用于将token映射为索引
    device (str): 设备，默认为"cpu"
"""
def infer(
    model: HLTModel,
    code_samples: List[List[List[str]]],
    vocab: dict,
    device: str = "cpu"
) -> List[int]:
    token_ids, token_mask, statement_mask = preprocess_code(code_samples, vocab)
    token_ids = token_ids.to(device)
    token_mask = token_mask.to(device)
    statement_mask = statement_mask.to(device)

    with torch.no_grad():
        logits = model(token_ids, token_mask, statement_mask)
    
    preds = logits.argmax(dim=1).cpu().tolist()
    return preds

# 示例用法
if __name__ == "__main__":
    VOCAB_SIZE = 10000
    CHECKPOINT_PATH = "checkpoints/hlt_model.pth"
    SAMPLE_CODE = [
        [
            ["if", "(", "x", ">", "0", ")"],
            ["int", "y", "=", "x", "+", "1", ";"]  
        ]
    ]
    VOCAB = {
        "if": 1, "(": 2, "x": 3, ">": 4, "0": 5, "int": 6, "y": 7, "=": 8, "+": 9, "1": 10,
        ";": 11, "<pad>": 0, "<unk>": 12
    }
    model = load_pretrained_model(VOCAB_SIZE, checkpoint_path=CHECKPOINT_PATH)
    predictions = infer(model, SAMPLE_CODE, VOCAB)
    print(f"推理结果（0：非漏洞，1：漏洞）：{predictions}")