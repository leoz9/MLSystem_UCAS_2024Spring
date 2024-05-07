import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MaxAbsScaler
import time
import psutil
import os
import csv
import pandas as pd
# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

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

def train_torch(X_train, y_train):
    scaler = MaxAbsScaler()
    X_train_scaled = scaler.fit_transform(X_train.toarray())
    X_train_torch = torch.FloatTensor(X_train_scaled).to(device)
    y_train_torch = torch.LongTensor(y_train.values).to(device)
    model = LogisticRegressionModel(X_train_torch.shape[1]).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    num_epochs = 10000
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        outputs = model(X_train_torch)
        loss = criterion(outputs, y_train_torch)
        loss.backward()
        optimizer.step()
    return model

def test_torch(model, X_test, y_test):
    scaler = MaxAbsScaler()
    X_test_scaled = scaler.fit_transform(X_test.toarray())
    X_test_torch = torch.FloatTensor(X_test_scaled).to(device)
    y_test_torch = torch.LongTensor(y_test.values).to(device)
    with torch.no_grad():
        outputs = model(X_test_torch)
        _, predicted = torch.max(outputs, 1)
        accuracy = (predicted == y_test_torch).sum().item() / y_test_torch.size(0)
    return accuracy

def measure_performance_torch():
    X_train, y_train, X_test, y_test = load_data()
    initial_memory = psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)
    initial_gpu_memory = torch.cuda.memory_allocated() / (1024 * 1024)
    initial_cpu = psutil.cpu_percent(interval=1)

    start_time = time.time()
    model = train_torch(X_train, y_train)
    train_duration = time.time() - start_time
    final_cpu = psutil.cpu_percent(interval=1)

    start_test_time = time.time()
    accuracy = test_torch(model, X_test, y_test)
    test_duration = time.time() - start_test_time

    final_memory = psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)
    final_gpu_memory = torch.cuda.memory_allocated() / (1024 * 1024)
    memory_used = final_memory - initial_memory
    gpu_memory_used = final_gpu_memory - initial_gpu_memory
    cpu_usage = (final_cpu + initial_cpu) / 2

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    return memory_used, gpu_memory_used, train_duration, test_duration, accuracy, num_params, cpu_usage

# Measure performance for Torch
torch_results = measure_performance_torch()

# Save Torch performance data to file
torch_data = {
    'Memory (MB)': torch_results[0],
    'Duration (s)': torch_results[2] + torch_results[3],
    'Accuracy': torch_results[4],
    'Parameters': torch_results[5],
    'CPU Utilization (%)': torch_results[6],
    'GPU Memory (MB)': torch_results[1],
}

with open('gpu_performance.csv', 'w', newline='') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=torch_data.keys())
    writer.writeheader()
    writer.writerow(torch_data)

