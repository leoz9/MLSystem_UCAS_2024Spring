import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MaxAbsScaler
import pandas as pd
import time
import psutil
import os
import csv
from tqdm import tqdm

# Define the model class
class LogisticRegressionModel(nn.Module):
    def __init__(self, num_features):
        super().__init__()
        self.linear = nn.Linear(num_features, 2)  # 2 for binary classification

    def forward(self, x):
        return self.linear(x)

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12345'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup():
    dist.destroy_process_group()

# Load data function
def load_data():
    data = pd.read_csv('/home/jovyan/code/homework/data/情感分类.csv')
    test_data = pd.read_csv('/home/jovyan/code/homework/data/情感分类测试机.csv')
    vectorizer = TfidfVectorizer(max_features=1000)
    X_train = vectorizer.fit_transform(data['正文'])
    y_train = data['正负面']
    X_test = vectorizer.transform(test_data['正文'])
    y_test = test_data['正负面']
    return X_train, y_train, X_test, y_test

def train_torch(rank, world_size):
    setup(rank, world_size)
    device = torch.device(f"cuda:{rank}")
    X_train, y_train, X_test, y_test = load_data()

    scaler = MaxAbsScaler()
    X_train_scaled = scaler.fit_transform(X_train.toarray())
    X_train_torch = torch.FloatTensor(X_train_scaled).to(device)
    y_train_torch = torch.LongTensor(y_train.values).to(device)

    model = LogisticRegressionModel(X_train_torch.shape[1]).to(device)
    model = DDP(model, device_ids=[rank])

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    num_epochs = 100
    start_time = time.time()

    for epoch in tqdm(range(num_epochs), desc=f"Training on GPU {rank}"):
        optimizer.zero_grad()
        outputs = model(X_train_torch)
        loss = criterion(outputs, y_train_torch)
        loss.backward()
        optimizer.step()
    train_duration = time.time() - start_time
    # Testing phase
    accuracy = test_torch(model, X_test, y_test, device)

    cleanup()
    if rank == 0:  # Only print on the main process
        torch_data = {
            'Memory (MB)': psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024),
            'Duration (s)': train_duration,
            'Accuracy': accuracy,
            'Parameters': sum(p.numel() for p in model.parameters() if p.requires_grad),
            'CPU Utilization (%)': psutil.cpu_percent(),
            'GPU Memory (MB)': torch.cuda.memory_allocated() / (1024 * 1024),
        }
        print(torch_data)
        with open('gpu_performance.csv', 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=torch_data.keys())
            writer.writeheader()
            writer.writerow(torch_data)

def test_torch(model, X_test, y_test, device):
    scaler = MaxAbsScaler()
    X_test_scaled = scaler.fit_transform(X_test.toarray())
    X_test_torch = torch.FloatTensor(X_test_scaled).to(device)
    y_test_torch = torch.LongTensor(y_test.values).to(device)

    with torch.no_grad():
        outputs = model.module(X_test_torch)
        _, predicted = torch.max(outputs, 1)
        local_accuracy = (predicted == y_test_torch).sum().item() / y_test_torch.size(0)

    # Reduce accuracy across all processes
    accuracy_tensor = torch.tensor(local_accuracy).to(device)
    dist.all_reduce(accuracy_tensor, op=dist.ReduceOp.SUM)
    global_accuracy = accuracy_tensor.item() / dist.get_world_size()
    return global_accuracy

def main():
    world_size = torch.cuda.device_count()
    torch.multiprocessing.spawn(train_torch, args=(world_size,), nprocs=world_size, join=True)

if __name__ == "__main__":
    main()
