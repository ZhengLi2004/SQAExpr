# HLT模型复现TODO List（Refined Version）

---

## 一、数据处理模块
### 1. 数据集准备
- [ ] 从 [https://osf.io/d45bw/](https://osf.io/d45bw/) 下载Russell等人的svd原始数据集  
- [ ] 生成svd-1（3000样本）和svd-2（2000样本），**确保正负样本严格1:1且无重叠**  
- [ ] 实现数据清洗函数：  
  - **删除C/C++注释**（包括`/* */`和`//`）  
  - **标准化代码格式**（统一缩进、换行符、去除多余空格）  
  - **处理预处理指令**（保留`#include`和`#define`，但删除条件编译指令如`#ifdef`）  

### 2. 分词与预处理
- [ ] 实现语句分割函数：  
  - **以`;`和`{`为分隔符**（需处理嵌套`{}`，例如函数体内部的大括号不分割）  
- [ ] 统计token频率，**构建词汇表**（保留频率≥2的token，最大词汇量限制为论文中报告的svd-1:8859，svd-2:6946）  
- [ ] 处理OOV：  
  - **将未登录词映射为`<unk>`**，初始化`<unk>`向量为随机正态分布（均值0，方差0.01）  

### 3. 填充与掩码生成
- [ ] **函数级填充**：  
  - 截断/填充语句数至`M=30`（根据论文中箱线图分析，上四分位数为30）  
- [ ] **语句级填充**：  
  - 截断/填充token数至`N=30`（根据论文中平均语句长度≈10，设置30以保留完整信息）  
- [ ] 生成`token_mask`（shape: `[batch_size, M, N]`）：  
  - **标记PAD位置为`True`，实际token为`False`**  
- [ ] 生成`statement_mask`（shape: `[batch_size, M]`）：  
  - **填充语句标记为`True`，真实语句为`False`**  

### 4. 词向量初始化
- [ ] **使用Word2vec（Skip-Gram模型）训练token嵌入**：  
  - 参数：`vector_size=128`, `window=5`, `min_count=2`, `workers=4`  
- [ ] 初始化Embedding层：  
  - **加载预训练的词向量，允许微调（trainable=True）**  

---

## 二、模型构建模块
### 1. BiLSTM层
- [ ] 实现多层BiLSTM类（支持1或2层）：  
  - **输入**：`token_ids`（`batch, M, N`）→ Embedding后转为（`batch, M, N, d`）  
  - **BiLSTM处理**：  
    - 每层双向LSTM的隐藏维度为`d//2`，最终拼接为`d`  
    - 使用`pack_padded_sequence`和`pad_packed_sequence`处理填充  
  - **输出**：  
    - 最后一层双向隐藏状态拼接 → 线性变换（`nn.Linear(2*d, d)`）→ `tanh`激活 → 语句表示（`batch, M, d`）  

### 2. Transformer层
- [ ] 实现单层Transformer Encoder（与基线模型区分）：  
  - **参数**：`nhead=6`, `dim_feedforward=512`  
  - **位置编码**：使用可学习的绝对位置编码（`nn.Embedding(M, d)`）  
  - **输入**：语句表示 + 位置编码 → LayerNorm  
  - **输出**：  
    - 对非填充语句的输出取平均 → 函数表示（`batch, d`）  
  - **注意力掩码**：在自注意力计算中应用`statement_mask`（`masked_fill`替换为`-inf`）  

### 3. 完整模型
- [ ] 组合BiLSTM和Transformer，实现HLT类：  
  - **输入流程**：  
    `token_ids` → Embedding → BiLSTM → Transformer → 分类头（`nn.Linear(d, 2)`）  
  - **维度验证**：  
    - BiLSTM输出：`[batch, M, d]`  
    - Transformer输出：`[batch, d]`  

---

## 三、训练配置模块
### 1. 训练参数
- [ ] 超参数配置：  
  - `d=128`, `M=30`, `N=30`, `num_layers=2`（HLT2）  
  - `batch_size=64`, `epochs=100`  
- [ ] **优化器**：  
  - 使用RAdam（`lr=0.001`, `weight_decay=1e-5`）  
- [ ] **早停机制**：  
  - 监控验证集准确率，若连续10个epoch无提升则停止  

### 2. 损失与指标
- [ ] **损失函数**：  
  - `nn.CrossEntropyLoss`（无需类别权重，数据集平衡）  
- [ ] **指标计算**：  
  - 使用`sklearn.metrics`计算准确率和F1-score（`average='binary'`）  

### 3. 数据加载器
- [ ] 实现层次化DataLoader：  
  - **输出四元组**：`(token_ids, labels, token_mask, statement_mask)`  
  - **动态生成mask**：在`collate_fn`中根据填充后的序列生成  

---

## 四、基线模型实现
### 1. Transformer基线（6层Encoder）
- [ ] 模型结构：  
  - **输入**：`token_ids`展平为`[batch, M*N]`  
  - **位置编码**：每个token的绝对位置（可学习，`Embedding(M*N, d)`）  
  - **6层Transformer Encoder**（`nhead=6`, `dim_feedforward=512`）  
  - **输出**：平均所有token表示 → 分类头  

### 2. Embedding-Transformer基线
- [ ] 模型结构：  
  - **语句向量生成**：对每个语句的token向量取平均（`[batch, M, d]`）  
  - **单层Transformer Encoder**（参数同上）  
  - **输出**：同HLT的Transformer部分  

---

## 五、实验脚本
### 1. 训练脚本
- [ ] 支持多GPU训练：  
  - 使用`torch.nn.DataParallel`封装模型  
- [ ] 日志记录：  
  - 保存训练/验证集的loss、acc、F1到TensorBoard  

### 2. 评估脚本
- [ ] 复现论文结果：  
  - **svd-1**：目标准确率≥67.85%，F1≥70.17%  
  - **svd-2**：目标准确率≥67.0%，F1≥72.03%  

### 3. 可视化
- [ ] 绘制语句长度分布图：  
  - 验证`M=30`覆盖约75%的函数（根据论文图4箱线图）  
- [ ] 保存模型参数和训练曲线（使用`torch.save`和`matplotlib`）  

---

## 六、验证与调试
### 1. 单元测试
- [ ] **数据预处理测试**：  
  - 随机抽样检查填充后维度是否为`[M=30, N=30]`  
  - 验证mask是否正确标记PAD  
- [ ] **模型前向测试**：  
  - 输入随机张量（`batch=2, M=30, N=30`），检查输出维度为`[2, 2]`  

### 2. 对比实验
- [ ] **基线模型验证**：  
  - Transformer基线在svd-1的准确率应≈64.66%（允许±0.5%浮动）  
- [ ] **HLT层数对比**：  
  - 确认`HLT_2`（2层BiLSTM）效果优于`HLT_1`  

### 3. 错误分析
- [ ] **误分类样本分析**：  
  - 检查长语句截断是否导致关键token丢失  
  - 分析复杂控制流（如嵌套循环）的处理能力  

---

## 依赖项
- [ ] **PyTorch 1.8.0+**（需支持Transformer和RAdam）  
- [ ] **NLTK**（用于分词辅助，如标点处理）  
- [ ] **scikit-learn 0.24+**（计算F1-score）  
- [ ] **tensorboard**（训练可视化）  

---

**备注**：  
- 所有实现需严格对齐论文中的实验设置（如维度、层数、位置编码类型）。  
- 基线模型的超参数（如Transformer层数）需与论文第IV-B节一致。  
- 词向量初始化时，若使用预训练Word2vec模型，需在训练前冻结Embedding层1个epoch后再微调。