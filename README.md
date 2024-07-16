

# University of Chinese Academy of Sciences

**Title:** Multi-System Algorithm Comparison - Sentiment Classification of Financial News

Course Name: Machine Learning Systems

Group Members: Guo Zhipeng, Zhang Lei, Li Chuankun

**Instructor: Liu Jie**  
[Personal Homepage](https://people.ucas.ac.cn/~liujie)

# Chapter 1: Background

Sentiment classification of financial news involves using natural language processing technologies to analyze and identify the emotional tendencies in the text of financial news. Research in this field is particularly important for the analysis and prediction of financial markets, as the sentiments conveyed in the news can significantly affect stock market behavior and investor actions.

## 1.1 Background and Importance

In financial markets, information is extremely crucial, and news reports are one of the primary sources of market information for participants. The language and emotional tendencies expressed in news can significantly influence investors' psychology and decisions, thereby affecting market prices and volatility. For example, a positive news report about a bank's performance exceeding expectations might push up the stock prices, while negative news about a financial crisis could trigger market panic and lead to a drop in stock prices.

## 1.2 Application Scenarios

Market Prediction and Analysis: Sentiment analysis can help analysts and investors understand market trends and predict market movements.

Algorithmic Trading: In high-frequency and algorithmic trading, automated tools can use the results of sentiment analysis to make quick trading decisions.

Risk Management: By monitoring negative sentiments in news, financial institutions can anticipate potential risks and take appropriate measures.

## 1.3 Process

Data Collection: Collect financial news text, possibly from news websites, social media, etc.

Text Preprocessing: Clean the data by removing noise such as HTML tags and special symbols.

Feature Extraction: Extract useful features from the cleaned text, such as TF-IDF weights and word frequencies.

Sentiment Classification: Use machine learning or deep learning models to classify the sentiment of the news.

Model Evaluation: Assess the performance of the model using metrics such as accuracy and runtime.

## 1.4 Challenges

Subjectivity and Complexity of Emotions: Different readers might interpret the emotions of the same news article differently.

Understanding Context and Irony: Financial news often contains complex economic terms and metaphors, posing a challenge to the accuracy of sentiment analysis.

Data Imbalance: In real datasets, certain categories of news (like neutral news) may be more common than others (like positive or negative news), which could cause the model to be biased towards predicting more common categories.

# Chapter 2: Dataset

The dataset is primarily used to train models for sentiment classification of financial news. The data is sourced from positive and negative news headlines published on Xueqiu, a comprehensive investment social network platform focused on providing comprehensive financial information and market data. Users can obtain real-time updates on the stock market, including A-shares, Hong Kong stocks, and U.S. stocks, helping them assess companies' financial conditions and performance. Moreover, investors share and discuss personal investment strategies and market insights, making it an ideal place for investor interaction and investment inspiration. Through real-time news updates and educational resources, Xueqiu aims to enhance users' decision-making quality and market analysis capabilities.

The seed dataset, collected through web scraping, includes 7,046 news headlines with 5,147 positive and 1,899 negative news items. The dataset was expanded through search engine searches and filtering, resulting in a final dataset containing 17,149 news items, including fields for date, company, code, sentiment (positive/negative), headline, and text, with 12,514 positive and 4,635 negative news items.

Dataset link: [https://github.com/wwwxmu/Dataset-of-financial-news-sentiment-classification](https://github.com/wwwxmu/Dataset-of-financial-news-sentiment-classification) Source: [xueqiu.com](https://xueqiu.com/)

# Chapter 3: Experimental Methods

The experiments employed three frameworks: sklearn, torch, and torch-DistributedDataParallel, each implementing a perceptron for sentiment classification of financial news. CPU and memory were monitored using psutil, training and testing times were calculated using the python package time, and GPU usage was calculated using torch.cuda.memory_allocated.

## 1.1 Experimental Environment Setup

Single Machine Environment:

- 13th Gen Intel(R) Core(TM) i7-13620H
- NVIDIA GeForce RTX 4060 Laptop GPU with 8GB VRAM
- Memory 16GB

Multi-GPU Distributed Environment:

- 4*A100 (40GB)

## 1.2 Experimental Methods

### 1.2.1 sklearn

Data Reading:

- `data = pd.read_csv('sentiment_classification.csv')`: Reads the training dataset from the `sentiment_classification.csv` file.

Feature Extraction:

- `vectorizer = TfidfVectorizer(max_features=1000)`: Initializes a TfidfVectorizer object to convert text data into numerical features in TF-IDF format. The parameter `max_features=1000` limits the number of features to a maximum of 1000, helping to reduce model complexity and potentially increase training speed.

- `X_train = vectorizer.fit_transform(data['text'])`: Fits the vectorizer to the text column in the training dataset and converts it into a TF-IDF feature matrix.

- `X_test = vectorizer.transform(test_data['text'])`: Transforms the text in the test dataset using the previously fitted vectorizer, ensuring that the feature space is consistent between training and testing datasets.

Model: `model = LogisticRegression()`

### 1.2.2 torch_cpu

Using CPU to train deep learning models.

Data Preprocessing:

- Uses `MaxAbsScaler` for normalization, scaling each feature to the range [-1, 1], which helps optimize the training process.

- `X_train_scaled` and `X_test_scaled` are the training and testing data transformed into normalized array formats.

Converting to PyTorch Tensors:

- `torch.FloatTensor` and `torch.LongTensor` are used to convert data into tensor formats understood by PyTorch. `FloatTensor` is used for features, `LongTensor` for labels.

Defining Data Loaders:

- `TensorDataset` encapsulates feature and label tensors for use with `DataLoader`.

- `DataLoader` provides functionality for batch loading data, with a batch size set to 10 and `shuffle=True` to randomize data for increased model generalization.

Model Definition:

- `LogisticRegressionModel(X_train_torch.shape[1])`

### 1.2.3 torch_GPU

The algorithm is the same as torch_cpu, with the addition of the line `device = torch.device("cuda" if torch.cuda.is_available() else "cpu")`.

1.2.3 torch- DistributedDataParallel

The algorithm is the same as torch_cpu, with `setup(rank, world_size)`

  `device = torch.device(f"cuda:{rank}")`

# Chapter 4: Experimental Results

Comparing model memory usage, runtime, accuracy, parameters, CPU usage, and GPU VRAM.
![image](https://github.com/leoz9/MLSystem_UCAS_2024Spring/assets/59195872/6ba7d844-c63b-4445-ac03-51dc3fcc8c27)

---
