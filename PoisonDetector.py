import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

class SequnceDataset(Dataset):
    def __init__(self, root_dirs, split=True, is_train=True, is_binary=True, is_regression=False, test_ratio=0.2, random_seed=0):
        self.files = []
        self.file_labels = []

        if 'with_adv' in root_dirs:
            adv_files = [os.path.join(root_dirs['with_adv'], f) for f in os.listdir(root_dirs['with_adv'])]
            for f in adv_files:
                self.files.append(f)
                if is_binary:
                    self.file_labels.append(1)
                else:
                    node_id = int(os.path.basename(f).split('-')[0][1:]) + 1
                    self.file_labels.append(node_id)

        if 'without_adv' in root_dirs:
            clean_files = [os.path.join(root_dirs['without_adv'], f) for f in os.listdir(root_dirs['without_adv'])]
            self.files.extend(clean_files)
            self.file_labels.extend([0] * len(clean_files))
                                        
        train_files, test_files, train_labels, test_labels = train_test_split(
            self.files, self.file_labels, test_size=test_ratio, random_state=random_seed
        )
        print(f'Train size: {len(train_files)}, Test size: {len(test_files)}')

        if split:
            if is_train:
                self.files = train_files
                self.file_labels = train_labels
            else:
                self.files = test_files
                self.file_labels = test_labels
        
        self.is_regression = is_regression

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        file_path = self.files[index]
        label = self.file_labels[index]
        with open(file_path, 'r') as file:
            accuracies = [float(line.strip()) for line in file]
        
        if self.is_regression:
            return torch.tensor(accuracies, dtype=torch.float), torch.tensor(label, dtype=torch.float)
        return torch.tensor(accuracies, dtype=torch.float), torch.tensor(label, dtype=torch.long)
    

class PoisonDetector_LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(PoisonDetector_LSTM, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        out, (hn, cn) = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out

class PoisonDetector_Regressor(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers):
        super(PoisonDetector_Regressor, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)
    
    def forward(self, x):
        out, (hn, cn) = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out

def run(adv_root_dir='Sequences/adversarial', clean_root_dir='Sequences/clean', model_path='model.pth', is_binary=False, is_regression=False):
    # Hyperparameters
    input_dim = 1
    hidden_dim = 100
    num_layers = 1
    output_dim = 9
    num_epochs = 200
    learning_rate = 0.01

    # Load data
    root_dirs = {
        'with_adv': adv_root_dir,
        'without_adv': clean_root_dir
    }
    
    train_dataset = SequnceDataset(root_dirs, is_train=True, is_binary=False, is_regression=is_regression)
    test_dataset = SequnceDataset(root_dirs, is_train=False, is_binary=False, is_regression=is_regression)

    if is_binary:
        output_dim = 2
        train_dataset = SequnceDataset(root_dirs, is_train=True, is_binary=True, is_regression=is_regression)
        test_dataset = SequnceDataset(root_dirs, is_train=False, is_binary=True, is_regression=is_regression)

    train_loader = DataLoader(train_dataset, batch_size=10, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=10, shuffle=False)

    # Initialize model
    if is_regression:
        model = PoisonDetector_Regressor(input_dim, hidden_dim, num_layers)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    else:
        model = PoisonDetector_LSTM(input_dim, hidden_dim, num_layers, output_dim)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Train model
    model.train()
    for epoch in range(num_epochs):
        for i, (sequences, labels) in enumerate(train_loader):
            sequences = sequences.unsqueeze(-1)
            if is_regression:
                labels = labels.unsqueeze(-1)
            optimizer.zero_grad()
            outputs = model(sequences)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}')
    torch.save(model.state_dict(), model_path)

    # Evaluate model
    evaluate_model(model, criterion, test_loader, is_regression)

def evaluate_model(model, criterion, test_loader, is_regression=False):
    if is_regression:
        model.eval()
        total_loss = 0
        total_samples = 0
        with torch.no_grad():
            for sequences, labels in test_loader:
                sequences = sequences.unsqueeze(-1)
                labels = labels.unsqueeze(-1)
                outputs = model(sequences)
                loss = criterion(outputs, labels)
                total_loss += loss.item() * sequences.size(0)
                total_samples += sequences.size(0)
        
        mse = total_loss / total_samples
        print(f'MSE: {mse}')

    else:
        model.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            for sequences, labels in test_loader:
                sequences = sequences.unsqueeze(-1)
                outputs = model(sequences)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            
            print(f'Accuracy: {100 * correct / total}%')


def test(test_root_dir='Sequences/test', model_path='model.pth', is_binary=False):
    # Hyperparameters
    input_dim = 1
    hidden_dim = 100
    num_layers = 1
    output_dim = 9

    # Load test data
    root_dirs = {
        'with_adv': test_root_dir
    }

    test_dataset = SequnceDataset(root_dirs, split=False, is_binary=True)
    test_loader = DataLoader(test_dataset, batch_size=10, shuffle=False)

    if is_binary:
        output_dim = 2
        test_dataset = SequnceDataset(root_dirs, split=False, is_binary=True)
        test_loader = DataLoader(test_dataset, batch_size=10, shuffle=False)

    # Load model
    model = PoisonDetector_LSTM(input_dim, hidden_dim, num_layers, output_dim)
    model.load_state_dict(torch.load(model_path))

    # Test model
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for sequences, labels in test_loader:
            sequences = sequences.unsqueeze(-1)
            outputs = model(sequences)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        print(f'Accuracy: {100 * correct / total}%')

if __name__ == '__main__':
    run(adv_root_dir='Sequences/adversarial', clean_root_dir='Sequences/clean')