---
title: Annotated Transformer
date: 2018-12-06 14:27:46
tags:
mathjax: true
---

## Background

减少时序计算的目标构成了扩展神经GPU，ByteNet和ConvS2S的基础，它们都使用卷积神经网络作为基本构建块，并行计算所有输入和输出位置的隐藏表示。 在这些模型中，与来自两个任意输入或输出位置的信号相关的操作数量随着位置之间的距离而增加，对于ConvS2S呈线性增长，对于ByteNet呈对数增长，这使得学习远程位置之间的依赖性变得更加困难。 在Transformer中，操作被减少到恒定次数，尽管由于平均注意力加权位置而导致有效分辨率降低，但这种效果我们可以用Multi-Head Attention抵消。

Self-attention 是一种将单个序列的不同位置联系起来以便计算序列表示的注意机制。Self-attention 已经成功地用于各种任务，包括阅读理解，抽象概括，文本蕴涵和学习任务无关的句子表示。 端到端记忆网络基于循环注意机制而不是序列对齐重复，并且已经证明在简单语言问答和语言建模任务上表现良好。

据我们所知，Transformer 是第一个完全依靠自我注意的转换模型，用于计算其输入和输出的表示，而不是使用序列对齐的RNN或卷积。

## Model Architecture

大多数的神经序列转换模型都具有 encoder-decoder 结构。这里，编码器将符号表示的输入序列$(x_1, ..., x_n)$映射到连续表示序列$\mathbf{z} = (z_1, ..., z_n)$。给定$\mathbf{z}$，解码器然后一次一个元素地生成符号的输出序列 $(y_1,...,y_m)$。在每个步骤中，模型是自回归的，在生成下一个时将先前生成的符号作为附加输入使用。

```
class EncoderDecoder(nn.Module):
    """
    A standard Encoder-Decoder architecture. Base for this and many 
    other models.
    """
    def __init__(self, encoder, decoder, src_embed, tgt_embed, generator):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator
        
    def forward(self, src, tgt, src_mask, tgt_mask):
        "Take in and process masked src and target sequences."
        return self.decode(self.encode(src, src_mask), src_mask,
                            tgt, tgt_mask)
    
    def encode(self, src, src_mask):
        return self.encoder(self.src_embed(src), src_mask)
    
    def decode(self, memory, src_mask, tgt, tgt_mask):
        return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)
```

```
class Generator(nn.Module):
    "Define standard linear + softmax generation step."
    def __init__(self, d_model, vocab):
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, vocab)

    def forward(self, x):
        return F.log_softmax(self.proj(x), dim=-1)
```
Transformer遵循这种总体架构，使用针对编码器和解码器的堆叠式self-attention 和点式完全连接的层，分别如下图的左半部分和右半部分所示。
![](Annotated-Transformer/ModalNet-21.png)

## Encoder and Decoder Stacks
### Encoder
编码器由N = 6个相同的层堆叠组成。

```
def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])
```

```
class Encoder(nn.Module):
    "Core encoder is a stack of N layers"
    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)
        
    def forward(self, x, mask):
        "Pass the input (and mask) through each layer in turn."
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)
```
我们在两个子层中的使用残差连接，然后进行[规范化](https://arxiv.org/abs/1607.06450)。

```
class LayerNorm(nn.Module):
    "Construct a layernorm module (See citation for details)."
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2
```

使用残差连接后，每个子层的输出是$\mathrm{LayerNorm}(x + \mathrm{Sublayer}(x))$，其中$\mathrm{Sublayer}(x)$是由子层本身实现的功能。 我们将dropout应用于每个子层的输出，然后将其与子层输入相加并进行规范化。

为了使残差连接更容易，模型中的所有子层以及嵌入层的输出维度$d_{model}=512$。

```
class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        return x + self.dropout(sublayer(self.norm(x)))
```
每层有两个子层：第一个是multi-head self-attention，第二种是一个简单的全连接网络。
```
class EncoderLayer(nn.Module):
    "Encoder is made up of self-attn and feed forward (defined below)"
    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask):
        "Follow Figure 1 (left) for connections."
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)
```

### Decoder
解码器也由N = 6个相同的层堆叠组成。

```
class Decoder(nn.Module):
    "Generic N layer decoder with masking."
    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)
        
    def forward(self, x, memory, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)
```

除了编码器层中的两个子层之外，解码器在中间插入了一个子层，其对编码器的输出执行multi-head self-attention。 与编码器类似，我们在每个子层周围使用残差连接，然后进行规范化。

```
class DecoderLayer(nn.Module):
    "Decoder is made of self-attn, src-attn, and feed forward (defined below)"
    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 3)
 
    def forward(self, x, memory, src_mask, tgt_mask):
        "Follow Figure 1 (right) for connections."
        m = memory
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask))
        return self.sublayer[2](x, self.feed_forward)
```

我们还修改解码器堆栈中的自注意子层以防止某位置关注其后续位置。 这种掩蔽确保了位置$i$的预测仅依赖于小于$i$的位置的已知输出。
```
def subsequent_mask(size):
    "Mask out subsequent positions."
    attn_shape = (1, size, size)
    # 返回一个下三角矩阵， k = 0代表主对角线， k < 0 在其下，k > 0 在其上，保留对角线下的值，上的置为0 
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8') 
    return torch.from_numpy(subsequent_mask) == 0 # 为0的位置为true，允许查看的位置
```
```
plt.figure(figsize=(5,5))
plt.imshow(subsequent_mask(20)[0])
None
```
我们把掩模可视化，下图显示允许每个target word（行）查看的位置（列），黄色代表允许查看。
![](Annotated-Transformer/mask_example.png)

## Attention

