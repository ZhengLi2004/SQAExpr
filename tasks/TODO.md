# HLT模型复现TODO List

## 一、数据处理模块
### 1. 数据集准备
- [] 从 https://osf.io/d45bw/ 下载svd原始数据集
- [] 生成svd-1（3000样本）和svd-2（2000样本），正负样本1:1
- [] 实现数据清洗函数：去除注释、空白符，标准化代码格式

### 2. 分词与预处理
- [ ] 实现语句分割函数（以";"和"{"为分隔符）
- [ ] 统计token频率，构建词汇表（保留freq≥2的token）
- [ ] 实现token到id的映射，处理OOV为<unk>

### 3. 填充与掩码生成
- [ ] 实现函数级填充：截断/填充语句数至M=30
- [ ] 实现语句级填充：截断/填充token数至N=30
- [ ] 生成token_mask（shape: [batch_size, M, N]），标记PAD位置为True
- [ ] 生成statement_mask（shape: [batch_size, M]），标记填充语句为True

### 4. 词向量初始化
- [ ] 使用Word2vec训练token嵌入（维度128），或加载预训练模型
- [ ] 初始化Embedding层，可训练或冻结

## 二、模型构建模块
### 1. BiLSTM层
- [ ] 实现多层BiLSTM类（支持1或2层）
  - 输入：token_ids (batch, M, N) → 经Embedding转为(batch, N, d)
  - 输出：语句表示 (batch, M, d)，拼接双向最后层隐藏状态并线性变换
- [ ] 在forward中应用token_mask，通过pack_padded_sequence优化计算

### 2. Transformer层
- [ ] 实现单层Transformer Encoder（含6头自注意力）
  - 输入：语句表示 + 位置编码（位置编码需可学习或固定）
  - 输出：函数表示 (batch, d)，平均非填充语句的输出
- [ ] 在注意力计算中应用statement_mask，掩盖填充语句

### 3. 完整模型
- [ ] 组合BiLSTM和Transformer，实现HLT类
  - 输入：token_ids (batch, M, N)
  - 输出：logits (batch, 2)
- [ ] 测试模型维度匹配：BiLSTM输出→Transformer输入→分类层

## 三、训练配置模块
### 1. 训练参数
- [ ] 定义超参数配置：d=128, M=30, N=30, num_layers=2（HLT2）
- [ ] 实现RAdam优化器（学习率0.001）
- [ ] 定义早停机制（基于验证集准确率，耐心值10）

### 2. 损失与指标
- [ ] 实现二分类交叉熵损失
- [ ] 计算准确率和F1-score（需处理不平衡样本）

### 3. 数据加载器
- [ ] 实现层次化DataLoader，输出(token_ids, labels, token_mask, statement_mask)
- [ ] 支持批量处理，动态生成mask

## 四、基线模型实现
### 1. Transformer基线（6层Encoder）
- [ ] 实现直接输入token序列的Transformer模型（无BiLSTM层）
  - 输入：token_ids (batch, M*N) → 位置编码到每个token
  - 输出：平均所有token的Encoder输出作为函数表示

### 2. Embedding-Transformer基线
- [ ] 实现语句向量平均+单层Transformer
  - 先对每个语句的token向量求平均得到语句向量
  - 输入Transformer同HLT的Transformer部分

## 五、实验脚本
### 1. 训练脚本
- [ ] 编写train.py，支持多GPU训练（DataParallel）
- [ ] 记录训练日志：loss、acc、F1在训练/验证集的变化

### 2. 评估脚本
- [ ] 编写evaluate.py，在测试集上计算最终指标
- [ ] 复现论文结果：svd-1 acc≥67.85%，svd-2 acc≥67.0%

### 3. 可视化
- [ ] 绘制语句长度和token长度分布（验证M=30, N=30的合理性）
- [ ] 保存模型参数和训练曲线

## 六、验证与调试
### 1. 单元测试
- [ ] 测试数据预处理：确保填充后维度正确，mask标记正确
- [ ] 测试模型前向传播：输入随机张量，检查输出维度

### 2. 对比实验
- [ ] 运行基线模型，验证Transformer基线acc≈64.66%（svd-1）
- [ ] 对比HLT1和HLT2，确认两层BiLSTM效果更好

### 3. 错误分析
- [ ] 分析误分类样本：检查是否因长语句截断或复杂逻辑处理不足


## 依赖项
- [ ] 安装PyTorch 1.8.0+
- [ ] 安装NLTK（分词辅助）
- [ ] 安装scikit-learn（计算F1-score）
- [ ] 安装tensorboard（训练可视化）