import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import time
import psutil


def load_data():
    data = pd.read_csv('情感分类.csv')
    test_data = pd.read_csv('情感分类测试机.csv')
    vectorizer = TfidfVectorizer(max_features=1000)
    X_train = vectorizer.fit_transform(data['正文'])
    y_train = data['正负面']
    X_test = vectorizer.transform(test_data['正文'])
    y_test = test_data['正负面']
    return X_train, y_train, X_test, y_test


def train_sklearn(X_train, y_train):
    model = LogisticRegression()
    model.max_iter = 10000
    model.fit(X_train, y_train)
    return model


def test_sklearn(model, X_test, y_test):
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    return accuracy


def measure_performance(model_train_func, model_test_func, data_load_func):
    start_time = time.time()
    X_train, y_train, X_test, y_test = data_load_func()  # 修正错误：将括号添加到函数调用中
    model = model_train_func(X_train, y_train)
    memory = psutil.Process().memory_info().rss / (1024 * 1024)  # in MB
    duration = time.time() - start_time

    start_test_time = time.time()
    accuracy = model_test_func(model, X_test, y_test)
    test_duration = time.time() - start_test_time

    # 获取模型参数
    num_params = model.coef_.shape[0] * model.coef_.shape[1]

    # 获取CPU利用率
    cpu_percent = psutil.cpu_percent()

    return memory, duration + test_duration, accuracy, num_params, cpu_percent


# Measure performance
sklearn_memory, sklearn_duration, sklearn_accuracy, sklearn_params, sklearn_cpu = measure_performance(train_sklearn,
                                                                                                      test_sklearn,
                                                                                                      load_data)

# Save data to file
sklearn_data = {
    'Memory (MB)': sklearn_memory,
    'Duration (s)': sklearn_duration,
    'Accuracy': sklearn_accuracy,
    'Parameters': sklearn_params,
    'CPU Utilization (%)': sklearn_cpu
}
print(sklearn_data)
sklearn_df = pd.DataFrame([sklearn_data])
sklearn_df.to_csv('sklearn_performance.csv', index=False)
