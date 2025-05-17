import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from models.hlt_model import HLTModel
from typing import List, Tuple
import numpy as np
from sklearn.metrics import f1_score

class CodeVulnerabilityDataset(Dataset):
    """
    自定义数据集类，用于加载与处理后的漏洞监测数据
    Args:
        token_ids (torch.Tensor): 输入语句的token索引，形状为(batch_size, M, N)，其中M为语句数，N为语句中的token数
        token_mask (torch.Tensor): 输入语句的token掩码，形状为(batch_size, M, N)，用于指示哪些token是有效的
        statement_masks (torch.Tensor): 输入语句的掩码，形状为(batch_size, M)，用于指示哪些语句是有效的
        labels (torch.Tensor): 标签，形状为(batch_size)，用于指示漏洞是否存在
    """
    def __init__(self, token_ids: torch.Tensor, token_mask: torch.Tensor, statement_masks: torch.Tensor, labels: torch.Tensor):
        self.token_ids = token_ids
        self.token_mask = token_mask
        self.statement_masks = statement_masks
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return (self.token_ids[idx], self.token_mask[idx], self.statement_masks[idx], self.labels[idx])

"""
训练函数
Args:
    train_dataset (Dataset): 训练数据集
    val_dataset (Dataset): 验证数据集
    vocab_size (int): 词汇表大小
    embedding_dim (int): 嵌入维度，默认为128
    num_layers (int): LSTM层数，默认为2
    batch_size (int): 批大小，默认为32
    epochs (int): 训练轮数，默认为50
    lr (float): 学习率，默认为2e-5
    device (str): 设备，默认为"cpu"
"""
def train_model(
    train_dataset: Dataset,
    val_dataset: Dataset,
    vocab_size: int,
    embedding_dim: int = 128,
    num_layers: int = 2,
    batch_size: int = 32,
    epochs: int = 50,
    lr: float = 2e-5,
    device: str = "cpu"
):
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model = HLTModel(
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        num_layers=num_layers
    )
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    best_val_f1 = 0.0

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        train_preds = []
        train_labels = []

        for batch in train_loader:
            token_ids, token_mask, statement_mask, labels = [x.to(device) for x in batch]
            labels = [x.to(device) for x in batch]

            logits = model(token_ids, token_mask, statement_mask)
            loss = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * len(labels)
            preds = logits.argmax(dim=1).cpu().tolist()
            train_preds.extend(preds)
            train_labels.extend(labels.cpu().tolist())

        train_loss /= len(train_dataset)
        train_f1 = f1_score(train_labels, train_preds)

        # 验证阶段
        model.eval()
        val_loss = 0.0
        val_preds = []
        val_labels = []

        with torch.no_grad():
            for batch in val_loader:
                token_ids, token_mask, statement_mask, labels = [x.to(device) for x in batch]
                logits = model(token_ids, token_mask, statement_mask)
                loss = criterion(logits, labels)

                val_loss += loss.item() * len(labels)
                preds = logits.argmax(dim=1).cpu().tolist()
                val_preds.extend(preds)
                val_labels.extend(labels.cpu().tolist())

        val_loss /= len(val_dataset)
        val_f1 = f1_score(val_labels, val_preds)

        print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss:.4f}, Train F1: {train_f1:.4f}, Val Loss: {val_loss:.4f}, Val F1: {val_f1:.4f}")

        if(val_f1) > best_val_f1:
            best_val_f1 = val_f1
            torch.save(model.state_dict(), "checkpoints/best_hlt_model.pth")
            print(f"模型已保存（验证F1提升至{best_val_f1:.4f}）")

    print(f"训练完成！最佳验证F1: {best_val_f1:.4f}")

# 训练示例
if __name__ == "__main__":
    VOCAB_SIZE = 10000 
    BATCH_SIZE = 16
    EPOCHS = 30
    LR = 2e-5

    # 应在实际数据集上划分，按照8:1:1的比例划分训练集、验证集和测试集
    train_dataset = CodeVulnerabilityDataset(
        token_ids=torch.randint(0, VOCAB_SIZE, (1000, 30*30)),
        token_masks=torch.randint(0, 2, (1000, 30*30), dtype=torch.bool),
        statement_masks=torch.randint(0, 2, (1000, 30), dtype=torch.bool),
        labels=torch.randint(0, 2, (1000,))
    )
    val_dataset = CodeVulnerabilityDataset(
        token_ids=torch.randint(0, VOCAB_SIZE, (200, 30*30)),
        token_masks=torch.randint(0, 2, (200, 30*30), dtype=torch.bool),
        statement_masks=torch.randint(0, 2, (200, 30), dtype=torch.bool),
        labels=torch.randint(0, 2, (200,))
    )

    train_model(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        vocab_size=VOCAB_SIZE,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        lr=LR
    )