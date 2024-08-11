import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import networkx as nx
from collections import deque
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

class GraphGenerator:
    def __init__(self, size, rank=0, comm=None):
        self.size = size
        self.rank = rank
        self.comm = comm

    def selectGraph(self, graph, p=None, num_c=None):
        if isinstance(graph, list):
            return graph
        else:
            g = []
            if graph == 'fully-connected':
                fc_graph = nx.complete_graph(self.size)
                g = list(fc_graph.edges)

            elif graph == 'ring':
                ring_graph = nx.cycle_graph(self.size)
                g = list(ring_graph.edges)
            
            elif graph == 'linear':
                for i in range(self.size - 1):
                    g.append((i, i + 1))

            elif graph == 'clique-ring':
                per_c = int(self.size / num_c)
                rem = self.size % num_c
                for i in range(num_c):
                    if i != num_c - 1:
                        fc_graph = nx.complete_graph(per_c)
                        fc_graph = nx.convert_node_labels_to_integers(fc_graph, i * per_c)
                        g += list(fc_graph.edges)
                        g.append((i * per_c + per_c - 1, i * per_c + per_c))
                    else:
                        fc_graph = nx.complete_graph(per_c + rem)
                        fc_graph = nx.convert_node_labels_to_integers(fc_graph, i * per_c)
                        g += list(fc_graph.edges)
                        if num_c > 2:
                            g.append((self.size - 1, 0))

            elif graph == 'erdos-renyi':
                if self.rank == 0:
                    while True:
                        erdos_graph = nx.erdos_renyi_graph(self.size, p)
                        if nx.is_connected(erdos_graph):
                            g = list(erdos_graph.edges)
                            num_edges = len(g) * np.ones(1, dtype=np.int)
                            print('Generated Erdos-Renyi Graph Edges:')
                            print(g)
                            break
                else:
                    num_edges = np.zeros(1, dtype=np.int)
                self.comm.Bcast(num_edges, root=0)
                num_edges = num_edges[0]
                if self.rank != 0:
                    data = np.empty((num_edges, 2), dtype=np.int)
                else:
                    data = np.array(g, dtype=np.int)
                self.comm.Bcast(data, root=0)
                if self.rank != 0:
                    for i in range(num_edges):
                        g.append((data[i][0], data[i][1]))
            return g
class BinarySequenceDataset(Dataset):
    def __init__(self, root_dirs, split=True, is_train=True, test_ratio=0.2, random_seed=0):
        self.files = []
        self.file_labels = []

        if 'with_adv' in root_dirs:
            adv_files = [os.path.join(root_dirs['with_adv'], f) for f in os.listdir(root_dirs['with_adv'])]
            self.files.extend(adv_files)
            self.file_labels.extend([1] * len(adv_files))
        
        if 'without_adv' in root_dirs:
            clean_files = [os.path.join(root_dirs['without_adv'], f) for f in os.listdir(root_dirs['without_adv'])]
            self.files.extend(clean_files)
            self.file_labels.extend([0] * len(clean_files))
        
        if split:
            train_files, test_files, train_labels, test_labels = train_test_split(
                self.files, self.file_labels, test_size=test_ratio, random_state=random_seed
            )
            print(f'Train size: {len(train_files)}, Test size: {len(test_files)}')

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


class SequnceDataset(Dataset):
    def __init__(self, root_dirs, distances, split=True, is_train=True, test_ratio=0.2, is_regression=True, random_seed=0):
        self.files = []
        self.file_labels = []
        self.is_regression = is_regression

        adv_files = [os.path.join(root_dirs['with_adv'], f) for f in os.listdir(root_dirs['with_adv'])]
        for f in adv_files:
            self.files.append(f)
            node_id = int(os.path.basename(f).split('-')[0][1:])
            self.file_labels.append(distances[node_id])

        if split:                           
            train_files, test_files, train_labels, test_labels = train_test_split(
                self.files, self.file_labels, test_size=test_ratio, random_state=random_seed
            )
            print(f'Train size: {len(train_files)}, Test size: {len(test_files)}')

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

def calculate_distances(g, start_node=0):
    G = nx.Graph()
    G.add_edges_from(g)
    
    distances = {node: float('inf') for node in G.nodes}
    distances[start_node] = 0
    
    queue = deque([start_node])
    
    while queue:
        current_node = queue.popleft()
        current_distance = distances[current_node]
        
        for neighbor in G.neighbors(current_node):
            if distances[neighbor] == float('inf'):
                distances[neighbor] = current_distance + 1
                queue.append(neighbor)
    
    max_distance = max(distances.values())
    return distances, max_distance

def normalize_distances(distances, max_distance):
    if max_distance == 0:
        return distances
    return {node: distance / max_distance for node, distance in distances.items()}

def denormalize_results(results, graph, graph_size, start_node=0):
    GP = GraphGenerator(size=graph_size)
    g = GP.selectGraph(graph)
    distances, max_distance = calculate_distances(g, start_node)
    denormalized_results = []
    for prediction, groud_truth in results:
        denormalized_prediction = prediction * max_distance
        denormalized_groud_truth = groud_truth * max_distance
        denormalized_results.append((denormalized_prediction, denormalized_groud_truth))
    return denormalized_results

def run(adv_root_dir='Sequences/adversarial', clean_root_dir='Sequences/clean', model_path='model.pth', is_binary=False, is_regression=False, graph='linear', graph_size=8):
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

    # Calculate distances
    GP = GraphGenerator(size=graph_size)
    g = GP.selectGraph(graph)
    distances, max_distance= calculate_distances(g)
    distances = normalize_distances(distances, max_distance)

    if is_binary:
        output_dim = 2
        train_dataset = BinarySequenceDataset(root_dirs, is_train=True)
        test_dataset = BinarySequenceDataset(root_dirs, is_train=False)
    else:
        train_dataset = SequnceDataset(root_dirs, distances, is_train=True, is_regression=is_regression)
        test_dataset = SequnceDataset(root_dirs, distances, is_train=False, is_regression=is_regression)

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


def test(test_root_dir='Sequences/test', model_path='model.pth', is_binary=False, is_regression=False, graph = 'linear', graph_size = 8):
    # Hyperparameters
    input_dim = 1
    hidden_dim = 100
    num_layers = 1
    output_dim = 9

    # Load test data
    root_dirs = {
        'with_adv': test_root_dir
    }

    # Calculate distances
    GP = GraphGenerator(size=graph_size)
    g = GP.selectGraph(graph)
    distances = calculate_distances(g)

    test_dataset = SequnceDataset(root_dirs, distances, split=False, is_train=False, is_regression=is_regression)
    test_loader = DataLoader(test_dataset, batch_size=10, shuffle=False)

    # if is_binary:
    #     output_dim = 2
    #     test_dataset = SequnceDataset(root_dirs, split=False, is_binary=True)
    #     test_loader = DataLoader(test_dataset, batch_size=10, shuffle=False)

    # Load model

    # Initialize model
    if is_regression:
        model = PoisonDetector_Regressor(input_dim, hidden_dim, num_layers)
        criterion = nn.MSELoss()
    else:
        model = PoisonDetector_LSTM(input_dim, hidden_dim, num_layers, output_dim)

    model.load_state_dict(torch.load(model_path))

    # Test model
    model.eval()
    if is_regression:
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

def predict(sequence_file, model_path):
    model = PoisonDetector_Regressor(1, 100, 1)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    with open(sequence_file, 'r') as file:
        accuracies = [float(line.strip()) for line in file]
    
    sequences = torch.tensor(accuracies, dtype=torch.float).unsqueeze(-1).unsqueeze(0)
    outputs = model(sequences)
    _, predicted = torch.max(outputs.data, 1)
    print(predicted.item())
    m
if __name__ == '__main__':
    run(adv_root_dir='Sequences/adversarial', clean_root_dir='Sequences/clean')