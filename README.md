# Sentiment-analysis
Sentiment analysis



1.项目背景介绍
============

<font size="3">情感识别的任务就是将一段文本分类，如分情感极性为正向和负向。  
  
  
  
 
  
</font>

![](https://ai-studio-static-online.cdn.bcebos.com/b9804199059b4cc3a45488808f955bc5957f3060ca3d4f53be99b6494108ced6)  
  
  <font size="3">正向： 表示正面积极的情感，如高兴，幸福，期待等，通常我们定义为1。  
负向： 表示负面消极的情感，如难过，伤心，愤怒等，通常我们定义为0。   
  
 
  
</font>


 <font size="3">本项目是一个情感陪伴聊天机器人的第一步，做的是对用户情感的极性检测，目的是用于检测用户输入的句子所包含的情感是正面的还是负面的，用以为后续机器人的实现提供基础的分析结果。  </font>

2.数据介绍
========

2.1 数据集的介绍
-----------------------

<font size="3">ChnSentiCorp数据集是一个中文情感二分类数据集。数据集的具体信息如下：</font>  





| 任务类别 | 数据集名称 | 训练集大小 | 开发集大小 | 测试集大小 |
| -------- | ---------- | ---------- | ---------- | ---------- |
| 句子级情感分类 | ChnSentiCorp | 9600 | 1200 | 1200 |





2.2 数据集的准备
-----------------------


```python
#解压数据集 到work目录下
!gzip -dfq /home/aistudio/data/data10320/chnsenticorp.tar.gz
!tar xvf data/data10320/chnsenticorp.tar -C /home/aistudio/work/
# 查看数据集的目录结构
!tree /home/aistudio/work/chnsenticorp
```

    gzip: /home/aistudio/data/data10320/chnsenticorp.tar.gz: No such file or directory
    chnsenticorp/
    chnsenticorp/test.tsv
    chnsenticorp/train.tsv
    chnsenticorp/dev.tsv
    /home/aistudio/work/chnsenticorp
    ├── dev.tsv
    ├── test.tsv
    └── train.tsv
    
    0 directories, 3 files



```python
#下载词向量并进行解压
!wget https://paddlenlp.bj.bcebos.com/models/embeddings/w2v.wiki.target.word-char.dim300.tar.gz
!tar -xvf w2v.wiki.target.word-char.dim300.tar.gz
```



```python
import numpy as np

# 加载词向量文件
wiki = np.load('w2v.wiki.target.word-char.dim300.npz')
# 查看文件内容，vocab指的字典，也就是说有哪些字或者词对应着词向量
for val in wiki:
    print(val)       
```

    vocab
    embedding



```python
# 打印一下前50的字典高频词， 看一下字典的基本信息
vocab = wiki['vocab']
print(vocab[:50].tolist())
print(len(vocab))
```

    ['，', '的', '。', '、', '平方公里', '和', '：', 'formula_', '在', '“', '一', '与', '了', '》', '一个', '”', '后', '中', '年', '中国', '有', '被', '地区', '及', '以', '人口密度', '人', '于', '他', '也', '而', '由', '《', '）', '10', '可以', '（', '位于', ')', '并', '为', '是', '等', '中华人民共和国', '成为', '12', '人口', '上', '美国', ',']
    352221



```python
# embedding指的就是vocab中的字或词对应的向量
embedding = wiki['embedding']
# 查看embedding信息与 vocab应是一一对应的
print(embedding.shape)
# 查看第一个字 "的" 对应的词向量
print(embedding[1])
```

    (352221, 300)
    [ 0.152521 -0.341647 -0.073913  0.038942  0.050971 -0.076602 -0.110351
     -0.113442  0.143589 -0.263466  0.177794 -0.013858  0.137263 -0.217374
     -0.018148 -0.317066 -0.214054 -0.220963 -0.078682 -0.207843  0.096573
     -0.097038 -0.268048 -0.229482  0.263559 -0.01643   0.154246 -0.072036
     -0.171584 -0.068479  0.115101 -0.215115 -0.162383  0.262748  0.058221
     -0.079936  0.10899  -0.14728  -0.082937 -0.027346  0.112706 -0.30309
      0.081799  0.039015 -0.137432 -0.090984 -0.00197  -0.336426  0.17729
      0.155571 -0.381625  0.126124 -0.083793 -0.23128   0.060697 -0.180996
      0.093496 -0.051971  0.234772 -0.075514  0.194272 -0.236215 -0.040863
     -0.047332 -0.089257 -0.065507 -0.252593  0.01125   0.145444  0.232675
     -0.088296 -0.136416  0.378186  0.046393  0.23225   0.065042  0.037886
      0.113326  0.243767  0.068571  0.013549  0.091744 -0.164455  0.174125
      0.23307   0.102729 -0.349806 -0.238806  0.071332  0.160546  0.055956
     -0.20342  -0.340732 -0.04415   0.318202 -0.007864 -0.437149 -0.043554
     -0.15565   0.021205 -0.038611 -0.095884 -0.337809 -0.126752  0.151647
      0.15783  -0.081212 -0.313803 -0.353301  0.082458 -0.008199  0.261839
     -0.004924  0.266484 -0.378672 -0.030143  0.1008    0.146761 -0.168808
      0.085657  0.026332 -0.201907  0.09296  -0.068673 -0.184357  0.096571
      0.026552  0.252368  0.011181 -0.107952 -0.000579 -0.027843  0.214281
     -0.188991 -0.165369 -0.046811  0.033118 -0.131094 -0.099295 -0.050579
      0.298792 -0.348701 -0.395699  0.208337  0.151253  0.050896 -0.112807
     -0.090449  0.02313  -0.066293  0.262178 -0.017196 -0.360578 -0.155941
      0.004876  0.444864  0.274192 -0.29544   0.111287  0.050504 -0.012197
     -0.028018 -0.184445  0.173637  0.214478 -0.320274 -0.343387  0.113768
     -0.146294 -0.19981  -0.346222 -0.156864  0.224232 -0.005361 -0.162875
     -0.085208 -0.016039 -0.089578  0.511295  0.056059  0.469392  0.018428
     -0.005743 -0.016258  0.31425  -0.065611 -0.055088 -0.117685  0.004397
     -0.227313  0.191211 -0.184306 -0.271876 -0.078257  0.051694 -0.0109
     -0.121614 -0.222053 -0.106867  0.122083 -0.045616 -0.282015  0.013526
      0.053414  0.126594  0.026822  0.162307 -0.257499  0.108193  0.243314
      0.068151 -0.06253   0.186153 -0.001156  0.093653 -0.525831  0.258715
      0.102637  0.04927   0.133203 -0.275549 -0.445494 -0.098353  0.004479
     -0.398171 -0.208897  0.021309 -0.147671  0.340889  0.048723  0.019949
     -0.052245  0.186812 -0.034853 -0.019484  0.074251  0.251589 -0.235023
     -0.279008  0.214144 -0.148907  0.096524  0.22265   0.150878  0.17993
      0.201503  0.162293  0.36894  -0.278514 -0.243338 -0.132454  0.092261
      0.301378  0.195642  0.225132 -0.186094 -0.157867  0.139048 -0.073375
      0.134048  0.260552  0.040206  0.272328  0.030982 -0.117896 -0.039553
     -0.305937 -0.02099   0.136647  0.183968  0.103705  0.177551  0.205922
     -0.200338  0.107317 -0.41112   0.264404  0.268886 -0.001548 -0.187708
      0.008518 -0.029049  0.132293  0.277376  0.367084  0.391804  0.098195
     -0.024639 -0.017596 -0.084271 -0.068892  0.188693  0.261833 -0.016907
      0.070412  0.012208  0.031999  0.042914  0.147195 -0.080608]


2.3 图像/文本数据的统计分析
---------------------------------------


```python
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
```

    ， 0
    的 1
    。 2
    、 3
    平方公里 4
    和 5
    ： 6
    formula_ 7
    在 8
    “ 9
    一 10
    与 11
    了 12



```python
# 2. 分词
res = jieba.lcut('这样的酒店也配称为5星级？')
print('分词结果： ', res)
```

    Building prefix dict from the default dictionary ...
    Loading model from cache /tmp/jieba.cache
    Loading model cost 0.743 seconds.
    Prefix dict has been built successfully.


    分词结果：  ['这样', '的', '酒店', '也', '配', '称为', '5', '星级', '？']



```python
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
```

    628
    1
    1965
    29
    7502
    165
    134
    30443
    1784



```python
# 4. 根据索引查找对应的embedding
print('第十个字对应的embedding')
print(embedding[1784])
```

    第十个字对应的embedding
    [-0.211015  0.371151 -0.429409  0.002337 -0.304096  0.364816  0.391649
     -0.107209  0.484592  0.070282 -0.130838  0.026703 -0.005343  0.193773
     -0.16938   0.04636  -0.147465  0.344871  0.261421 -0.311586 -0.002753
     -0.011945 -0.269062 -0.113537 -0.18929  -0.453769  0.015942 -0.203646
     -0.242445 -0.300856 -0.144519 -0.057521  0.136354  0.190224 -0.483966
     -0.287246  0.148725 -0.158994  0.139199  0.115553 -0.121526  0.098384
     -0.02939  -0.308522 -0.331937 -0.072781  0.077903  0.049448 -0.245022
      0.317022 -0.247106  0.132422 -0.215921 -0.072787 -0.319834  0.028511
      0.321383  0.203269  0.243769 -0.125953  0.017895 -0.135638 -0.009326
      0.153859  0.061309 -0.280682 -0.057736  0.042958 -0.055143  0.266406
      0.111043 -0.293399  0.308726  0.028142  0.313393 -0.041551 -0.068743
      0.252304 -0.076995  0.244487  0.260915 -0.085712 -0.131976 -0.056907
      0.163906  0.039709 -0.149906  0.072166  0.033228  0.029566  0.136244
      0.152013 -0.177268 -0.144486 -0.492589 -0.077376 -0.223001 -0.035099
     -0.258213  0.155688  0.084651  0.153581  0.130612 -0.267632 -0.016078
     -0.13363   0.125319  0.098862  0.101727  0.0191    0.228885  0.042171
     -0.286158  0.351486  0.197632  0.070174  0.427454  0.003688 -0.299642
     -0.078227 -0.015954 -0.128707 -0.280217 -0.065884  0.234943  0.117971
     -0.091489  0.229471  0.199448  0.1549   -0.353897 -0.029914  0.155569
      0.215389  0.111408 -0.140984 -0.152292 -0.091242  0.171295  0.039876
     -0.127674 -0.28354  -0.464219 -0.169367 -0.00767   0.170748 -0.185472
     -0.113901  0.043261 -0.033038  0.529559  0.01143  -0.11405  -0.085599
      0.226706  0.285898  0.26259  -0.283073  0.072636  0.218459 -0.388294
     -0.154858  0.311722 -0.122602  0.019105  0.013265 -0.061365  0.058175
     -0.33813  -0.46058   0.168625  0.008384  0.465485 -0.160303  0.197273
     -0.073542  0.360149 -0.269793 -0.491348  0.126259  0.086952 -0.151129
      0.031738  0.115667  0.04006   0.209519  0.377957 -0.380783 -0.361594
      0.014064  0.062987  0.006225 -0.023196 -0.248063 -0.067072  0.270233
     -0.126842 -0.124    -0.100489  0.194622 -0.302983  0.066341 -0.549485
      0.082223  0.140567 -0.088284 -0.23065  -0.374734  0.216398 -0.412349
      0.40849  -0.163017 -0.099312  0.28803   0.230204  0.009966  0.589768
      0.238586 -0.413148  0.40582  -0.43511  -0.106064 -0.276646 -0.093064
      0.081152  0.139773  0.112354 -0.372269 -0.092714  0.492993  0.102615
      0.033022 -0.219861 -0.293371 -0.09275  -0.125992  0.013952 -0.352017
     -0.155752  0.393159 -0.079494  0.226863  0.202679  0.288778  0.049139
      0.231684 -0.105203  0.266087 -0.525267  0.034382  0.026183  0.127595
      0.402341 -0.023204 -0.23394   0.240684 -0.166106  0.053237  0.294144
     -0.076908  0.179752  0.02642  -0.064243 -0.027949 -0.599744  0.501828
      0.134213 -0.09897   0.435778  0.483174 -0.24191   0.259917 -0.195344
      0.079238  0.40929  -0.177936  0.05875  -0.140121  0.256695 -0.408062
     -0.274435  0.128353 -0.162644 -0.036668  0.205592 -0.285858 -0.017897
      0.325328  0.278413  0.242088  0.261875  0.175845  0.107766  0.138225
     -0.058623 -0.192074 -0.487265  0.069674 -0.010093 -0.30133 ]



```python
# jieba词频统计
from jieba import analyse
with open('./work/chnsenticorp/dev.tsv', mode='r', encoding='utf-8') as f:
    text = f.read()
extract_tags = analyse.extract_tags(text, withWeight=True)
for i, j in extract_tags:
    print(i, j)
```

    Building prefix dict from the default dictionary ...
    Dumping model to file cache /tmp/jieba.cache
    Loading model cost 0.826 seconds.
    Prefix dict has been built successfully.


    酒店 0.09231671359341595
    房间 0.05837604393282028
    不错 0.04943463553883596
    本书 0.040797121996002036
    入住 0.031625533390716734
    没有 0.030658701879640517
    感觉 0.028551691252834178
    比较 0.025452250667869904
    非常 0.02518059894592587
    携程 0.024404009205929344
    可以 0.024087477246410638
    宾馆 0.023729181662867173
    还是 0.023625008645386525
    服务 0.023590920072403987
    我们 0.021370317644523667
    前台 0.020893233558447452
    早餐 0.020689321131295027
    喜欢 0.019665073633516425
    就是 0.018299759776885482
    一个 0.01824404097762057


2.4 数据集类的定义
--------------------------


```python
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
```


```python
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
```


```python
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
```


```python
from paddlenlp.data import JiebaTokenizer, Stack, Pad, Tuple
from functools import partial
from paddlenlp.embeddings import TokenEmbedding  

```


```python
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
```


```python
import pandas as pd
from paddlenlp.datasets import load_dataset, MapDataset

# 加载数据集
train_dataset = load_dataset(read_func, file_path=config.train_path, is_train=True, lazy=False)
dev_dataset = load_dataset(read_func, file_path=config.dev_path, is_train=True, lazy=False)
test_dataset = load_dataset(read_func, file_path=config.test_path, is_train=False, lazy=False)
```


```python
# 定义数据预处理函数
train_dataset.map(trans_fn)
dev_dataset.map(trans_fn)
test_dataset.map(trans_fn)
```






```python
# 这个函数用来对训练集和验证集进行处理，核心目的就是进行padding。将一个mini-batch的句子长度对齐
batchify_fn_1 = lambda samples, fn=Tuple(
    Pad(pad_val=tokenizer.pad_token_id, axis=0),  # text_a
    Stack(),   # label
): fn(samples)

# 这个函数用来对测试集进行处理
batchify_fn_2 = lambda samples, fn=Tuple(
    Pad(pad_val=tokenizer.pad_token_id, axis=0),  # text_a
): fn(samples)
```

2.5 数据集类的测试
--------------------------


```python
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
```

  

2.6 词云可视化
--------------------


```python
!pip install wordcloud
```


```python
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
```


![png](output_33_0.png)


3.模型介绍
========

<font size="3">PaddleHub是基于PaddlePaddle生态下的预训练模型管理和迁移学习工具，可以结合预训练模型更便捷地开展迁移学习工作。本项目将采用PaddleHub一键加载ERNIE Tiny模型。  

项目采用的模型Ernie Tiny 主要通过模型结构压缩和模型蒸馏的方法，将 ERNIE 2.0 Base模型进行压缩。Ernie Tiny模型采用 3 层 transformer 结构，利用模型蒸馏的方式在 Transformer 层和 Prediction 层学习 ERNIE 2.0 模型对应层的分布和输出，通过综合优化能带来4.3倍的预测提速。</font>

![](https://ai-studio-static-online.cdn.bcebos.com/3492bffa26c64f938fc4f8abbaafb53ed470e9a373d5464996239c6bfe62751e)


4.模型训练
============

4.1 建立模型
-----------------


```python
# 安装paddlepaddle和paddlehub
!pip3 install --upgrade paddlepaddle -i https://mirror.baidu.com/pypi/simple
!pip3 install --upgrade paddlehub -i https://mirror.baidu.com/pypi/simple
```

 

```python
import paddlehub as hub
import paddle
```


```python
# 加载语义模型（需指定模型版本否则报错）
module = hub.Module(name='ernie_tiny', version='2.0.1', task='seq-cls', num_classes=2)
```

  

```python
#加载并定义3个数据集
train_dataset = hub.datasets.ChnSentiCorp(tokenizer = module.get_tokenizer() , max_seq_len=128 ,mode='train')
dev_dataset = hub.datasets.ChnSentiCorp(tokenizer = module.get_tokenizer() , max_seq_len=128 ,mode='dev')
test_dataset = hub.datasets.ChnSentiCorp(tokenizer = module.get_tokenizer() , max_seq_len=128 ,mode='test')
```

4.2 定义优化器并配置训练模型
----


```python
# 定义优化器和进行参数配置
optimizer = paddle.optimizer.Adam(learning_rate=5e-5, parameters=module.parameters())
trainer = hub.Trainer(module,optimizer,checkpoint_dir='test_ernie_test_cls', use_gpu= False)
```



```python
# 配置训练参数，启动训练，并指定验证集(模型训练较慢，所以仅指定epochs=1，batch_size=32)
trainer.train(train_dataset, epochs=1, batch_size=32, eval_dataset=dev_dataset, save_interval=5) 
```

    [2022-03-05 16:35:36,544] [   TRAIN] - Epoch=1/1, Step=10/300 loss=0.6647 acc=0.5656 lr=0.000050 step/sec=0.09 | ETA 00:54:10
    [2022-03-05 16:37:22,241] [   TRAIN] - Epoch=1/1, Step=20/300 loss=0.4266 acc=0.8281 lr=0.000050 step/sec=0.09 | ETA 00:53:30
    [2022-03-05 16:39:08,361] [   TRAIN] - Epoch=1/1, Step=30/300 loss=0.3185 acc=0.8719 lr=0.000050 step/sec=0.09 | ETA 00:53:21
    [2022-03-05 16:40:57,983] [   TRAIN] - Epoch=1/1, Step=40/300 loss=0.2806 acc=0.8969 lr=0.000050 step/sec=0.09 | ETA 00:53:43
    [2022-03-05 16:42:47,490] [   TRAIN] - Epoch=1/1, Step=50/300 loss=0.2409 acc=0.9187 lr=0.000050 step/sec=0.09 | ETA 00:53:55
    [2022-03-05 16:44:36,751] [   TRAIN] - Epoch=1/1, Step=60/300 loss=0.2577 acc=0.9062 lr=0.000050 step/sec=0.09 | ETA 00:54:02
    [2022-03-05 16:46:31,165] [   TRAIN] - Epoch=1/1, Step=70/300 loss=0.2307 acc=0.9250 lr=0.000050 step/sec=0.09 | ETA 00:54:29
    [2022-03-05 16:48:25,165] [   TRAIN] - Epoch=1/1, Step=80/300 loss=0.3273 acc=0.8906 lr=0.000050 step/sec=0.09 | ETA 00:54:48
    [2022-03-05 16:50:14,009] [   TRAIN] - Epoch=1/1, Step=90/300 loss=0.2450 acc=0.8969 lr=0.000050 step/sec=0.09 | ETA 00:54:46
    [2022-03-05 16:52:02,293] [   TRAIN] - Epoch=1/1, Step=100/300 loss=0.2724 acc=0.9000 lr=0.000050 step/sec=0.09 | ETA 00:54:42
    [2022-03-05 16:53:51,381] [   TRAIN] - Epoch=1/1, Step=110/300 loss=0.2671 acc=0.9062 lr=0.000050 step/sec=0.09 | ETA 00:54:41
    [2022-03-05 16:55:42,732] [   TRAIN] - Epoch=1/1, Step=120/300 loss=0.2859 acc=0.8719 lr=0.000050 step/sec=0.09 | ETA 00:54:46
    [2022-03-05 16:57:33,049] [   TRAIN] - Epoch=1/1, Step=130/300 loss=0.2478 acc=0.8906 lr=0.000050 step/sec=0.09 | ETA 00:54:48
    [2022-03-05 16:59:23,152] [   TRAIN] - Epoch=1/1, Step=140/300 loss=0.2364 acc=0.8938 lr=0.000050 step/sec=0.09 | ETA 00:54:49
    [2022-03-05 17:01:15,123] [   TRAIN] - Epoch=1/1, Step=150/300 loss=0.2085 acc=0.9219 lr=0.000050 step/sec=0.09 | ETA 00:54:53
    [2022-03-05 17:03:11,033] [   TRAIN] - Epoch=1/1, Step=160/300 loss=0.2179 acc=0.9344 lr=0.000050 step/sec=0.09 | ETA 00:55:05
    [2022-03-05 17:05:05,471] [   TRAIN] - Epoch=1/1, Step=170/300 loss=0.2598 acc=0.9031 lr=0.000050 step/sec=0.09 | ETA 00:55:12
    [2022-03-05 17:06:58,313] [   TRAIN] - Epoch=1/1, Step=180/300 loss=0.2094 acc=0.9250 lr=0.000050 step/sec=0.09 | ETA 00:55:16
    [2022-03-05 17:08:50,404] [   TRAIN] - Epoch=1/1, Step=190/300 loss=0.2261 acc=0.9250 lr=0.000050 step/sec=0.09 | ETA 00:55:19
    [2022-03-05 17:10:40,642] [   TRAIN] - Epoch=1/1, Step=200/300 loss=0.2622 acc=0.9031 lr=0.000050 step/sec=0.09 | ETA 00:55:18
    [2022-03-05 17:12:35,529] [   TRAIN] - Epoch=1/1, Step=210/300 loss=0.2163 acc=0.9156 lr=0.000050 step/sec=0.09 | ETA 00:55:24
    [2022-03-05 17:14:25,208] [   TRAIN] - Epoch=1/1, Step=220/300 loss=0.2112 acc=0.9219 lr=0.000050 step/sec=0.09 | ETA 00:55:23
    [2022-03-05 17:16:20,477] [   TRAIN] - Epoch=1/1, Step=230/300 loss=0.1869 acc=0.9281 lr=0.000050 step/sec=0.09 | ETA 00:55:29
    [2022-03-05 17:18:13,412] [   TRAIN] - Epoch=1/1, Step=240/300 loss=0.2042 acc=0.9313 lr=0.000050 step/sec=0.09 | ETA 00:55:31
    [2022-03-05 17:20:08,661] [   TRAIN] - Epoch=1/1, Step=250/300 loss=0.2616 acc=0.9031 lr=0.000050 step/sec=0.09 | ETA 00:55:36
    [2022-03-05 17:22:00,198] [   TRAIN] - Epoch=1/1, Step=260/300 loss=0.2176 acc=0.9094 lr=0.000050 step/sec=0.09 | ETA 00:55:36
    [2022-03-05 17:23:47,189] [   TRAIN] - Epoch=1/1, Step=270/300 loss=0.1829 acc=0.9344 lr=0.000050 step/sec=0.09 | ETA 00:55:32
    [2022-03-05 17:25:41,731] [   TRAIN] - Epoch=1/1, Step=280/300 loss=0.1938 acc=0.9313 lr=0.000050 step/sec=0.09 | ETA 00:55:35
    [2022-03-05 17:27:38,419] [   TRAIN] - Epoch=1/1, Step=290/300 loss=0.1744 acc=0.9437 lr=0.000050 step/sec=0.09 | ETA 00:55:41
    [2022-03-05 17:29:31,250] [   TRAIN] - Epoch=1/1, Step=300/300 loss=0.2067 acc=0.9344 lr=0.000050 step/sec=0.09 | ETA 00:55:43


5.模型评估与预测
====


```python
# 在测试集上评估当前训练模型
result = trainer.evaluate(test_dataset, batch_size=32)  
```



```python
#使用3条数据文本进行预测
import paddlehub as hub

data = [
    ['交通方便；环境很好；服务态度很好，但房间太小了'],
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
```

    [2022-03-05 17:50:27,631] [    INFO] - Already cached /home/aistudio/.paddlenlp/models/ernie-tiny/ernie_tiny.pdparams
    [2022-03-05 17:50:39,360] [    INFO] - Already cached /home/aistudio/.paddlenlp/models/ernie-tiny/vocab.txt
    [2022-03-05 17:50:39,362] [    INFO] - Already cached /home/aistudio/.paddlenlp/models/ernie-tiny/spm_cased_simp_sampled.model
    [2022-03-05 17:50:39,364] [    INFO] - Already cached /home/aistudio/.paddlenlp/models/ernie-tiny/dict.wordseg.pickle


    Data: 交通方便；环境很好；服务态度很好，但房间太小了 	 Lable: negative
    Data: 前台接待太差，酒店有A B楼之分，本人check－in后，前台未告诉B楼在何处，并且B楼无明显指示；房间太小，根本不像4星级。 	 Lable: negative
    Data: 19天硬盘就罢工了~~~算上运来的一周都没用上15天~~~可就是不能换了~~~唉~~~~你说这算什么事呀~~~ 	 Lable: negative


6.总结与升华
===

<font size="3"> 最后我们通过模型训练所记录下来的Log，运用VisualDL，绘制了模型训练的Loss和Accuracy的趋势图。</font>

![](https://ai-studio-static-online.cdn.bcebos.com/40e6087cd95f4e8f974bfe1c870d48f1e079383d45c44090b1ebdd7febf90635)


![](https://ai-studio-static-online.cdn.bcebos.com/6984e57bd8514f94b9d5bb6baaa991984f36ea55f54644ff91612a64996a25dd)


<font size="3"> 从图中我们可以看到，整个模型的准确率随训练轮数的变化，整个模型训练在前期就已经收敛了，后期的准确率都是高于0.9的，模型收敛速度较快，可能是因为数据量不是特别大的缘故。</font>

<font size="3"> 这是我第一次使用paddlepaddle,也是我第一次接触深度学习，跟着课程走学到了很多新的知识，磕磕绊绊的把项目完成了。由于是新手，所以选择的数据集也是比较简单的数据集，希望以后能做的更好。  
  最后提一点不成熟的小建议，希望官方能完善Paddlehub和VisualDL的相关文档，现有的文档对新手来说可能没有那么友好，使用起来没有那么容易。
</font>

7.个人总结
==

<font size="3">作者：洛米。  
研二，一个深度学习的小白，希望能和大家共同学习，一起进步。  
  
[个人主页](https://aistudio.baidu.com/aistudio/personalcenter/thirdview/2012606)   
</font>
