import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

class SequnceDataset(Dataset):
    def __init__(self, root_dirs, split=True, is_train=True, is_binary=True, test_ratio=0.2, random_seed=0):
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

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        file_path = self.files[index]
        label = self.file_labels[index]
        with open(file_path, 'r') as file:
            accuracies = [float(line.strip()) for line in file]
        
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

def run():
    # Hyperparameters
    input_dim = 1
    hidden_dim = 100
    num_layers = 1
    output_dim = 9
    num_epochs = 100
    learning_rate = 0.01

    # Load data
    root_dirs = {
        'with_adv': 'Sequences/adversarial',
        'without_adv': 'Sequences/clean'
    }

    train_dataset = SequnceDataset(root_dirs, is_train=True, is_binary=False)
    test_dataset = SequnceDataset(root_dirs, is_train=False, is_binary=False)

    train_loader = DataLoader(train_dataset, batch_size=10, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=10, shuffle=False)

    # Initialize model
    model = PoisonDetector_LSTM(input_dim, hidden_dim, num_layers, output_dim)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Train model
    model.train()
    for epoch in range(num_epochs):
        for i, (sequences, labels) in enumerate(train_loader):
            sequences = sequences.unsqueeze(-1)
            optimizer.zero_grad()
            outputs = model(sequences)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}')
    # torch.save(model.state_dict(), 'model.pth')

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

def test():
    # Hyperparameters
    input_dim = 1
    hidden_dim = 100
    num_layers = 1
    output_dim = 2

    # Load model
    model = PoisonDetector_LSTM(input_dim, hidden_dim, num_layers, output_dim)
    model.load_state_dict(torch.load('model.pth'))

    # Load test data
    root_dirs = {
        'with_adv': 'Sequences/test'
    }

    test_dataset = SequnceDataset(root_dirs, split=False, is_binary=True)
    test_loader = DataLoader(test_dataset, batch_size=10, shuffle=False)

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
    run()