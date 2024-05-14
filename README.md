中国科学院大学南京学院

**题目：**多系统算法对比-财经新闻情感分类

课程名称：     机器学习系统      

组  员：    郭志鹏、张磊、李传坤   

**指导教师：**          



 

# 第一章 问题背景

财经新闻情感分类是指使用自然语言处理技术来分析和识别财经新闻文本中的情绪倾向。这个领域的研究对金融市场的分析和预测尤为重要，因为新闻中传达的情绪可以显著影响股票市场和投资者的行为。

## 1.1 背景和重要性

在金融市场中，信息是极其重要的，而新闻报道是市场参与者获取信息的主要渠道之一。新闻中的语言表达和情绪倾向可以在很大程度上影响投资者的心理和决策，进而影响市场价格和波动。例如，一条报道银行业绩超预期的正面新闻可能会推动相关股票价格上涨，而报道金融危机的负面新闻可能会引起市场恐慌，导致股价下跌。

## 1.2应用场景

市场预测与分析： 情绪分析可以帮助分析师和投资者理解市场趋势，预测市场动向。

算法交易： 在高频交易和算法交易中，自动化工具可以利用情绪分析的结果快速做出交易决策。

风险管理： 通过监控新闻中的负面情绪，金融机构可以提前感知潜在的风险，采取相应措施。

## 1.3 处理过程

数据采集： 收集财经新闻文本，可能来源于新闻网站、社交媒体等。

文本预处理： 清洗数据，去除噪声，如HTML标签、特殊符号等。

特征提取： 从清洗后的文本中提取有用的特征，如TF-IDF权重、词频等。

情感分类： 使用机器学习模型或深度学习模型来分类新闻的情感倾向。

模型评估： 通过准确率，运行时间等指标评估模型的性能。

## 1.4 挑战

情绪的主观性和复杂性： 不同的读者可能对同一篇新闻的情绪有不同的理解。

语境和讽刺的理解： 财经新闻中常常包含复杂的经济术语和隐喻，这对情感分析的准确性是一个挑战。

数据不平衡： 在实际的数据集中，某些类别的新闻（如中性新闻）可能比其他类别（如正面或负面新闻）更为常见，这可能导致模型偏向于预测较多的类别。

# 第二章 数据集

数据集主要用于训练财经新闻情感分类的模型。数据来源于雪球网上万得资讯发布的正负面新闻标题。雪球网是一个综合性的投资社交网络平台，专注于提供全面的财经信息和市场数据。用户可以在平台上获取实时的股票市场动态，包括A股、港股和美股等。帮助用户评估公司的财务状况和业绩表现。此外，投资者分享和讨论个人的投资策略和市场见解，使其成为一个投资者交流和获取投资灵感的理想场所。通过实时新闻更新和教育资源，雪球旨在提升用户的投资决策质量和市场分析能力。

通过爬虫采集到7046条新闻标题作为种子数据集，其中正面新闻5147条，负面新闻1899条。对数据进行扩充，扩充的策略是通过搜索引擎搜索和筛选，得到最终的数据集。数据集中包含17149条新闻数据，包括日期、公司、代码、正/负面、标题、正文6个字段，其中正面新闻12514条，负面新闻4635条。

数据集链接https://github.com/wwwxmu/Dataset-of-financial-news-sentiment-classification数据集源[xueqiu.com](https://xueqiu.com/)

 



 

# 第三章 实验方法

​    实验使用了三种框架，分别是sklearn，torch，torch- DistributedDataParallel。分别实现了感知机对财经新闻情感分类。使用psutil对cpu，memory进行监控，使用python包time计算训练和测试时间，使用torch.cuda. memory_allocated计算gpu使用情况。

## 1.1 实验环境配置

单机环境

13th Gen Intel(R) Core(TM) i7-13620H

NVIDIA GeForce RTX 4060 Laptop GPU 8显存

Memory 16G

多卡分布式环境

4*A100(40G)

## 1.2 实验方法

​    分别实验sklearn，torch， DistributedDataParallel，实现感知机，对财经新闻情感分类。

### 1.2.1 sklearn

数据读取：

data = pd.read_csv('情感分类.csv')：从情感分类.csv文件中读取训练数据集。

特征提取：

vectorizer = TfidfVectorizer(max_features=1000)：初始化一个TfidfVectorizer对象，用于将文本数据转换为TF-IDF格式的数值特征。参数max_features=1000限制了特征的数量最多为1000，这有助于减少模型复杂性并可能提高训练速度。

X_train = vectorizer.fit_transform(data['正文'])：对训练数据集中的文本正文列进行拟合，并转换成TF-IDF特征矩阵。

X_test = vectorizer.transform(test_data['正文'])：使用之前拟合的向量化器转换测试数据集中的文本，保证训练集和测试集的特征空间一致。

模型model = LogisticRegression()

### 1.2.2 torch_cup

​    使用cup训练深度学习模型。

数据预处理:

使用 MaxAbsScaler 进行归一化处理。这种方法将每个特征缩放到 [-1, 1] 的范围内，有助于优化模型训练过程。

X_train_scaled 和 X_test_scaled 是将训练和测试数据转换成归一化后的数组形式。

转换为 PyTorch 张量:

torch.FloatTensor 和 torch.LongTensor 用于将数据转换为 PyTorch 理解的张量格式。FloatTensor 用于特征，LongTensor 用于标签。

定义数据加载器:

TensorDataset 封装了特征和标签张量，以便与 DataLoader 配合使用。

DataLoader 提供了批量加载数据的功能，这里批量大小设置为10，且设置 shuffle=True 以随机洗牌数据增加模型的泛化能力。

模型定义:

LogisticRegressionModel(X_train_torch.shape[1])

### 1.2.2 torch_GPU

​    算法同torch_cpu，需加上改行代码device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

1.2.3 torch- DistributedDataParallel

​    算法同torch_cpu，setup(rank, world_size)

  device = torch.device(f"cuda:{rank}")

# 第四章 实验结果

对比模型的内存使用，运行时间，准确率，参数，cpu使用率，GPU显存。

![img](D:\笔记\clip_image002.png)