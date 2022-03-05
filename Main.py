#!/usr/bin/env python
# coding: utf-8

# 1.项目背景介绍
# ============

# <font size="3">情感识别的任务就是将一段文本分类，如分情感极性为正向和负向。  
#   
#   
#   
#  
#   
# </font>

# ![](https://ai-studio-static-online.cdn.bcebos.com/b9804199059b4cc3a45488808f955bc5957f3060ca3d4f53be99b6494108ced6)  
#   
#   <font size="3">正向： 表示正面积极的情感，如高兴，幸福，期待等，通常我们定义为1。  
# 负向： 表示负面消极的情感，如难过，伤心，愤怒等，通常我们定义为0。   
#   
#  
#   
# </font>
# 

#  <font size="3">本项目是一个情感陪伴聊天机器人的第一步，做的是对用户情感的极性检测，目的是用于检测用户输入的句子所包含的情感是正面的还是负面的，用以为后续机器人的实现提供基础的分析结果。  </font>

# 2.数据介绍
# ========

# 2.1 数据集的介绍
# -----------------------

# <font size="3">ChnSentiCorp数据集是一个中文情感二分类数据集。数据集的具体信息如下：</font>  

# 
# 
# 
# 
# | 任务类别 | 数据集名称 | 训练集大小 | 开发集大小 | 测试集大小 |
# | -------- | ---------- | ---------- | ---------- | ---------- |
# | 句子级情感分类 | ChnSentiCorp | 9600 | 1200 | 1200 |
# 
# 
# 
# 

# 2.2 数据集的准备
# -----------------------

# In[ ]:


#解压数据集 到work目录下
get_ipython().system('gzip -dfq /home/aistudio/data/data10320/chnsenticorp.tar.gz')
get_ipython().system('tar xvf data/data10320/chnsenticorp.tar -C /home/aistudio/work/')
# 查看数据集的目录结构
get_ipython().system('tree /home/aistudio/work/chnsenticorp')


# In[ ]:


#下载词向量并进行解压
get_ipython().system('wget https://paddlenlp.bj.bcebos.com/models/embeddings/w2v.wiki.target.word-char.dim300.tar.gz')
get_ipython().system('tar -xvf w2v.wiki.target.word-char.dim300.tar.gz')


# In[ ]:


import numpy as np

# 加载词向量文件
wiki = np.load('w2v.wiki.target.word-char.dim300.npz')
# 查看文件内容，vocab指的字典，也就是说有哪些字或者词对应着词向量
for val in wiki:
    print(val)       


# In[ ]:


# 打印一下前50的字典高频词， 看一下字典的基本信息
vocab = wiki['vocab']
print(vocab[:50].tolist())
print(len(vocab))


# In[ ]:


# embedding指的就是vocab中的字或词对应的向量
embedding = wiki['embedding']
# 查看embedding信息与 vocab应是一一对应的
print(embedding.shape)
# 查看第一个字 "的" 对应的词向量
print(embedding[1])


# 2.3 图像/文本数据的统计分析
# ---------------------------------------

# In[ ]:


import jieba

# 1. 创建字典
dictionary = {val: index for index, val in enumerate(vocab)}
count = 0
# 查看字典中的12个字跟对应的id
for k, v in dictionary.items():
    print(k, v)
    if count == 12:
        break
    count += 1


# In[ ]:


# 2. 分词
res = jieba.lcut('这样的酒店也配称为5星级？')
print('分词结果： ', res)


# In[ ]:


# 3. 在字典中查找token对应的索引
print(dictionary.get('这样'))
print(dictionary.get('的'))
print(dictionary.get('酒店'))
print(dictionary.get('也'))
print(dictionary.get('配'))
print(dictionary.get('称为'))
print(dictionary.get('5'))
print(dictionary.get('星级'))
print(dictionary.get('？'))


# In[ ]:


# 4. 根据索引查找对应的embedding
print('第十个字对应的embedding')
print(embedding[1784])


# In[ ]:


# jieba词频统计
from jieba import analyse
with open('./work/chnsenticorp/dev.tsv', mode='r', encoding='utf-8') as f:
    text = f.read()
extract_tags = analyse.extract_tags(text, withWeight=True)
for i, j in extract_tags:
    print(i, j)


# 2.4 数据集类的定义
# --------------------------

# In[ ]:


# 定义配置参数
class Config(object):
    def __init__(self):
        # 超参数定义
        self.epochs = 100
        self.lr = 0.001
        self.max_seq_len = 256
        self.batch_size = 256

        # 数据集定义
        self.train_path = './work/chnsenticorp/train.tsv'
        self.dev_path = './work/chnsenticorp/dev.tsv'
        self.test_path = './work/chnsenticorp/test.tsv'

        # 模型参数定义
        self.embedding_name = 'w2v.wiki.target.word-char.dim300'
        self.num_filters = 256
        self.dropout = 0.2
        self.num_class = 2


# In[ ]:


# 定义切词器，目的是将句子切词并转换为id，并进行最大句子截断
class Tokenizer(object):
    def __init__(self, vocab):
        self.vocab = vocab
        self.tokenizer = JiebaTokenizer(vocab)  #定义切词器
        self.UNK_TOKEN = '[UNK]'
        self.PAD_TOKEN = '[PAD]'
        self.pad_token_id = vocab.token_to_idx.get(self.PAD_TOKEN)

    # 将文本序列切词并转换为id，并设定句子最大长度，超出将被截断
    def text_to_ids(self, text, max_seq_len=512):
        input_ids = []
        unk_token_id = self.vocab[self.UNK_TOKEN]
        for token in self.tokenizer.cut(text):
            token_id = self.vocab.token_to_idx.get(token, unk_token_id)
            input_ids.append(token_id)
        return input_ids[:max_seq_len]
# 定义数据读取
# 这个函数的目的，是作为load_dataset的参数，用来创建Dataset
def read_func(file_path, is_train=True):
    df = pd.read_csv(file_path, sep='\t')
    for index, row in df.iterrows():
        if is_train:
            yield {'label': row['label'], 'text_a': row['text_a']}
        else:
            yield {'text_a': row['text_a']}


# 定义数据预处理函数
# 将输入句子转换为id
def convert_example(example, tokenizer, max_seq_len):
    text_a = example['text_a']
    text_a_ids = tokenizer.text_to_ids(text_a, max_seq_len)
    if 'label' in example:  # 如果有label表示是训练集或者验证集，否则是测试集
        return text_a_ids, example['label']
    else:
        return text_a_ids


# In[ ]:


# 定义数据读取
# 这个函数的目的，是作为load_dataset的参数，用来创建Dataset
def read_func(file_path, is_train=True):
    df = pd.read_csv(file_path, sep='\t')
    for index, row in df.iterrows():
        if is_train:
            yield {'label': row['label'], 'text_a': row['text_a']}
        else:
            yield {'text_a': row['text_a']}


# 定义数据预处理函数
# 将输入句子转换为id
def convert_example(example, tokenizer, max_seq_len):
    text_a = example['text_a']
    text_a_ids = tokenizer.text_to_ids(text_a, max_seq_len)
    if 'label' in example:  # 如果有label表示是训练集或者验证集，否则是测试集
        return text_a_ids, example['label']
    else:
        return text_a_ids


# In[ ]:


from paddlenlp.data import JiebaTokenizer, Stack, Pad, Tuple
from functools import partial
from paddlenlp.embeddings import TokenEmbedding  


# In[ ]:


# 创建配置参数对象
config = Config()
# 定义词向量Layer
embedding = TokenEmbedding(embedding_name=config.embedding_name,
                                        unknown_token='[UNK]', 
                                        unknown_token_vector=None, 
                                        extended_vocab_path=None, 
                                        trainable=True, 
                                        keep_extended_vocab_only=False)
# 根据字典定义切词器
tokenizer = Tokenizer(embedding.vocab)

trans_fn = partial(convert_example, tokenizer=tokenizer, max_seq_len=config.max_seq_len)


# In[ ]:


import pandas as pd
from paddlenlp.datasets import load_dataset, MapDataset

# 加载数据集
train_dataset = load_dataset(read_func, file_path=config.train_path, is_train=True, lazy=False)
dev_dataset = load_dataset(read_func, file_path=config.dev_path, is_train=True, lazy=False)
test_dataset = load_dataset(read_func, file_path=config.test_path, is_train=False, lazy=False)


# In[ ]:


# 定义数据预处理函数
train_dataset.map(trans_fn)
dev_dataset.map(trans_fn)
test_dataset.map(trans_fn)


# In[ ]:


# 这个函数用来对训练集和验证集进行处理，核心目的就是进行padding。将一个mini-batch的句子长度对齐
batchify_fn_1 = lambda samples, fn=Tuple(
    Pad(pad_val=tokenizer.pad_token_id, axis=0),  # text_a
    Stack(),   # label
): fn(samples)

# 这个函数用来对测试集进行处理
batchify_fn_2 = lambda samples, fn=Tuple(
    Pad(pad_val=tokenizer.pad_token_id, axis=0),  # text_a
): fn(samples)


# 2.5 数据集类的测试
# --------------------------

# In[ ]:


from paddle.io import DataLoader

train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=config.batch_size,
    return_list=True,
    shuffle=True,
    collate_fn=batchify_fn_1
)
print(len(train_loader))
dev_loader = DataLoader(
    dataset=dev_dataset,
    batch_size=config.batch_size,
    return_list=True,
    shuffle=False,
    collate_fn=batchify_fn_1
)
print(len(dev_loader))

test_loader = DataLoader(
    dataset=test_dataset,
    batch_size=config.batch_size,
    return_list=True,
    shuffle=False,
    collate_fn=batchify_fn_2
)
print(len(test_loader))
print(train_dataset[0])
print(dev_dataset[10])
print(test_dataset[0])
for comment,label in enumerate(train_loader):
    print(comment, label)


# 2.6 词云可视化
# --------------------

# In[ ]:


get_ipython().system('pip install wordcloud')


# In[ ]:


# 词云可视化
import matplotlib.pyplot as plt
from wordcloud import WordCloud

result = {}
for word in extract_tags:
    result[word[0]] = word[1]

wordcloud = WordCloud(
    background_color="orange",
    max_font_size=85,
    font_path='samples/simkai.ttf')
wordcloud.generate_from_frequencies(result)

plt.figure()
plt.axis('off')
plt.imshow(wordcloud)
plt.show()


# 3.模型介绍
# ========

# <font size="3">PaddleHub是基于PaddlePaddle生态下的预训练模型管理和迁移学习工具，可以结合预训练模型更便捷地开展迁移学习工作。本项目将采用PaddleHub一键加载ERNIE Tiny模型。  
# 
# 项目采用的模型Ernie Tiny 主要通过模型结构压缩和模型蒸馏的方法，将 ERNIE 2.0 Base模型进行压缩。Ernie Tiny模型采用 3 层 transformer 结构，利用模型蒸馏的方式在 Transformer 层和 Prediction 层学习 ERNIE 2.0 模型对应层的分布和输出，通过综合优化能带来4.3倍的预测提速。</font>

# ![](https://ai-studio-static-online.cdn.bcebos.com/3492bffa26c64f938fc4f8abbaafb53ed470e9a373d5464996239c6bfe62751e)
# 

# 4.模型训练
# ============

# 4.1 建立模型
# -----------------

# In[ ]:


# 安装paddlepaddle和paddlehub
get_ipython().system('pip3 install --upgrade paddlepaddle -i https://mirror.baidu.com/pypi/simple')
get_ipython().system('pip3 install --upgrade paddlehub -i https://mirror.baidu.com/pypi/simple')


# In[1]:


import paddlehub as hub
import paddle


# In[2]:


# 加载语义模型（需指定模型版本否则报错）
module = hub.Module(name='ernie_tiny', version='2.0.1', task='seq-cls', num_classes=2)


# In[3]:


#加载并定义3个数据集
train_dataset = hub.datasets.ChnSentiCorp(tokenizer = module.get_tokenizer() , max_seq_len=128 ,mode='train')
dev_dataset = hub.datasets.ChnSentiCorp(tokenizer = module.get_tokenizer() , max_seq_len=128 ,mode='dev')
test_dataset = hub.datasets.ChnSentiCorp(tokenizer = module.get_tokenizer() , max_seq_len=128 ,mode='test')


# 4.2 定义优化器并配置训练模型
# ----

# In[5]:


# 定义优化器和进行参数配置
optimizer = paddle.optimizer.Adam(learning_rate=5e-5, parameters=module.parameters())
trainer = hub.Trainer(module,optimizer,checkpoint_dir='test_ernie_test_cls', use_gpu= False)


# In[ ]:


# 配置训练参数，启动训练，并指定验证集(模型训练较慢，所以仅指定epochs=1，batch_size=32)
trainer.train(train_dataset, epochs=1, batch_size=32, eval_dataset=dev_dataset, save_interval=5) 


# 5.模型评估与预测
# ====

# In[ ]:


# 在测试集上评估当前训练模型
result = trainer.evaluate(dev_dataset, batch_size=32)  


# In[ ]:


#使用3条数据文本进行预测
import paddlehub as hub

data = [
    ['交通方便；环境很好；服务态度很好'],
    ['前台接待太差，酒店有A B楼之分，本人check－in后，前台未告诉B楼在何处，并且B楼无明显指示；房间太小，根本不像4星级。'],
    ['19天硬盘就罢工了~~~算上运来的一周都没用上15天~~~可就是不能换了~~~唉~~~~你说这算什么事呀~~~'],
]
label_map = {0: 'negative', 1: 'positive'}

module = hub.Module(
    name='ernie_tiny',
    version='2.0.1',
    task='seq-cls',
    label_map=label_map)
results = module.predict(data, max_seq_len=50, batch_size=1, use_gpu=False)
for idx, text in enumerate(data):
    print('Data: {} \t Lable: {}'.format(text[0], results[idx]))



