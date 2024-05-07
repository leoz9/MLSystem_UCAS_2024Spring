import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MaxAbsScaler
import time
import psutil
import os
import matplotlib.pyplot as plt
import pandas as pd
# Define the model class for Torch
class LogisticRegressionModel(nn.Module):
    def __init__(self, num_features):
        super().__init__()
        self.linear = nn.Linear(num_features, 2)  # 2 for binary classification
    def forward(self, x):
        return self.linear(x)

def load_data():
    data = pd.read_csv('情感分类.csv')
    test_data = pd.read_csv('情感分类测试机.csv')
    vectorizer = TfidfVectorizer(max_features=1000)
    X_train = vectorizer.fit_transform(data['正文'])
    y_train = data['正负面']
    X_test = vectorizer.transform(test_data['正文'])
    y_test = test_data['正负面']
    return X_train, y_train, X_test, y_test

def train_torch(data):
    X_train, y_train, X_test, y_test = data
    scaler = MaxAbsScaler()
    X_train_scaled = scaler.fit_transform(X_train.toarray())
    X_test_scaled = scaler.transform(X_test.toarray())
    X_train_torch = torch.FloatTensor(X_train_scaled)
    y_train_torch = torch.LongTensor(y_train.values)
    X_test_torch = torch.FloatTensor(X_test_scaled)
    y_test_torch = torch.LongTensor(y_test.values)
    train_data = TensorDataset(X_train_torch, y_train_torch)
    train_loader = DataLoader(train_data, batch_size=10, shuffle=True)
    model = LogisticRegressionModel(X_train_torch.shape[1])
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    num_epochs = 100
    for epoch in range(num_epochs):
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
    return model, X_train_torch, y_train_torch, X_test_torch, y_test_torch

def test_torch(model, X_test, y_test):
    with torch.no_grad():
        outputs = model(X_test)
        _, predicted = torch.max(outputs.data, 1)
        accuracy = (predicted == y_test).sum().item() / y_test.size(0)
    return accuracy

def measure_performance(model_train_func, model_test_func, data_load_func):
    initial_memory = psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)
    initial_cpu = psutil.cpu_percent(interval=1)

    start_time = time.time()
    model, X_train, y_train, X_test, y_test = model_train_func(data_load_func())
    train_duration = time.time() - start_time
    final_cpu = psutil.cpu_percent(interval=1)

    start_test_time = time.time()
    accuracy = model_test_func(model, X_test, y_test)
    test_duration = time.time() - start_test_time

    final_memory = psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)
    memory_used = final_memory - initial_memory
    cpu_usage = (final_cpu + initial_cpu) / 2

    if isinstance(model, nn.Module):
        params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    else:
        params = model.coef_.size

    return memory_used, train_duration, test_duration, accuracy, params, cpu_usage

# Measure performance
torch_results = measure_performance(train_torch, test_torch, load_data)


# Printing results
print("Torch Results (Memory MB, Train Duration, Test Duration, Accuracy, Parameters, CPU %):", torch_results)
torch_data = {
    'Memory (MB)': torch_results[0],
    'Duration (s)': torch_results[1]+torch_results[2],
    'Accuracy': torch_results[3],
    'Parameters': torch_results[4],
    'CPU Utilization (%)': torch_results[5]
}

torch_df = pd.DataFrame([torch_data])
torch_df.to_csv('torch_performance.csv', index=False)
# Plotting the results separately
metrics = ['Memory (MB)', 'Train Duration (s)', 'Test Duration (s)', 'Accuracy', 'Parameters', 'CPU Usage (%)']
data_torch = torch_results