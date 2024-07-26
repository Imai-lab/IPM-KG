import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data, DataLoader
from sklearn.model_selection import KFold
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
from sklearn.model_selection import KFold
from torch_geometric.nn import SAGEConv
import torch.nn as nn
import random
import networkx as nx
from gensim.models import Word2Vec
from torch_geometric.data import Data
from torch_geometric.data import Batch
from tqdm import tqdm
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
import os
from sklearn.metrics import precision_recall_curve, roc_curve, auc
import matplotlib.pyplot as plt
import pickle
from sklearn.model_selection import train_test_split, KFold
from sklearn.model_selection import StratifiedKFold
from node2vec import Node2Vec
from sklearn.metrics import precision_recall_curve, roc_curve, auc
import numpy as np
from Edgeclassify_module import SampledSAGEConv, EdgeClassifier, load_edges, load_edge_labels, node2vec_embedding, create_data_object

# Load edges and edge labels from text files
edges = load_edges('Graph_data/graph_edge.txt')
edge_labels = load_edge_labels('Graph_data/graph_label.txt')
all_nodes = set()
for edge in edges:
    all_nodes.update(edge)
sorted_nodes = sorted(all_nodes)
# NumPy create
sorted_nodes_np = np.array(sorted_nodes)
node_to_idx = {node: idx for idx, node in enumerate(sorted_nodes_np)}
edges_idx = [(node_to_idx[src], node_to_idx[tgt]) for src, tgt in edges]

# Create a networkx graph from the edges
G = nx.Graph()
G.add_edges_from(edges_idx)
node2vec_model = node2vec_embedding(G)
with open('model_saved/graph_embedding.pkl', 'wb') as fout:
   pickle.dump(node2vec_model, fout)

with open('model_saved/graph_embedding.pkl', 'rb') as file:
   vec_model = pickle.load(file)

num_nodes = len(node_to_idx)
x = torch.tensor(numpy.array([vec_model.wv[str(node)] for node in range(num_nodes)]), dtype=torch.float)
previous_tensor_shape = x.size()

# Create data object
edge_labels_tensor = torch.tensor(edge_labels, dtype=torch.long)  # Convert edge_labels to tensor
data = create_data_object(x, edges_idx, edge_labels_tensor)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

edge_labels_np = data.edge_labels.cpu().numpy()
zero_one_indices = np.where((edge_labels_np == 0) | (edge_labels_np == 1))[0]

# Extract indices with edge_label other than 0, 1
non_zero_one_indices = np.where((edge_labels_np != 0) & (edge_labels_np != 1))[0]
kf = KFold(n_splits=5, shuffle=True, random_state=42)


if not os.path.exists("data"):
    os.makedirs("data")

best_macro_f1 = 0
best_hyperparameters = {}
batch_size=8

dropout_probs = [0.3,0.5,0.7]
hidden_channels_list = [128]
learning_rates = [0.001,0.002,0.003]
num_layers_list = [1,2,3]
sampling_ratio_list=[0.6, 0.8, 1]

numbers=0
k_loss=0
from sklearn.model_selection import StratifiedKFold
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
for train_valid_indices, test_indices in skf.split(zero_one_indices, edge_labels_np[zero_one_indices]):
    kf = StratifiedKFold(n_splits=4, shuffle=True, random_state=42)
    train_indices, valid_indices = next(kf.split(zero_one_indices[train_valid_indices], edge_labels_np[zero_one_indices[train_valid_indices]]))
    best_valid_f1 = 0
    best_hyperparameters = {}
    for dropout_prob in dropout_probs:
        for hidden_channels in hidden_channels_list:
            for lr in learning_rates:
                for num_layers in num_layers_list:
                    for sampling_ratio in sampling_ratio_list:
                        train_zero_one_indices = zero_one_indices[train_valid_indices[train_indices]]
                        valid_zero_one_indices = zero_one_indices[train_valid_indices[valid_indices]]
                        train_edge_index = data.edge_index[:, train_zero_one_indices]
                        train_edge_labels = data.edge_labels[train_zero_one_indices]
                        valid_edge_index = data.edge_index[:, valid_zero_one_indices]
                        valid_edge_labels = data.edge_labels[valid_zero_one_indices]
                        train_data = Data(x=data.x, edge_index=train_edge_index, edge_labels=train_edge_labels, edge_attr=data.edge_attr[train_zero_one_indices])
                        valid_data = Data(x=data.x, edge_index=valid_edge_index, edge_labels=valid_edge_labels, edge_attr=data.edge_attr[valid_zero_one_indices])
                        train_data = train_data.to(device)
                        valid_data = valid_data.to(device)
                        train_dataset = [Data(x=data.x, edge_index=train_data.edge_index[:, i:i+1], edge_label=train_data.edge_labels[i:i+1], edge_attr=train_data.edge_attr[i:i+1]) for i in range(train_data.edge_index.shape[1])]
                        valid_dataset = [Data(x=data.x, edge_index=valid_data.edge_index[:, i:i+1], edge_label=valid_data.edge_labels[i:i+1], edge_attr=valid_data.edge_attr[i:i+1]) for i in range(valid_data.edge_index.shape[1])]
                        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
                        valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
                        model = EdgeClassifier(num_features=128, hidden_channels=hidden_channels, num_classes=2, num_layers=num_layers,dropout_prob=dropout_prob).to(device)
                        criterion = torch.nn.CrossEntropyLoss()
                        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
                        for epoch in tqdm(range(25), desc="Epoch", leave=False):
                            model.train()
                            loss_add=0
                            loss_n=0
                            for batch in tqdm(train_loader, desc="Train", leave=False):
                                batch = batch.to(device)
                                optimizer.zero_grad()
                                train_logits = model(batch)
                                loss = criterion(train_logits, batch.edge_label)
                                loss.backward()
                                optimizer.step()
                                loss_add+=loss
                                loss_n+=1
                            loss_avg=loss_add/loss_n
                            print(loss_avg)
                            with open("log/log_graph.txt", "a") as log_file:
                                log_file.write(f"epoch={epoch}, loss={loss_avg}\n")
                            # Evaluating on valid set after each epoch
                            model.eval()
                            valid_probs_list = []
                            valid_preds_list = []
                            valid_labels_list = []
                            with torch.no_grad():
                                for batch in tqdm(valid_loader, desc="Valid", leave=False):
                                    batch = batch.to(device)
                                    valid_logits = model(batch)
                                    valid_logits_binary = valid_logits[:, :2]
                                    valid_preds_binary = valid_logits_binary.argmax(dim=-1)
                                    valid_probs = F.softmax(valid_logits_binary, dim=-1)[:, 1]
                                    valid_probs_list.extend(valid_probs.cpu().tolist())
                                    valid_preds_list.extend(valid_preds_binary.cpu().tolist())
                                    valid_labels_list.extend(batch.edge_label.cpu().tolist())
                            precision, recall, f1_score, _ = precision_recall_fscore_support(valid_labels_list, valid_preds_list, average='macro')
                            cm = confusion_matrix(valid_labels_list, valid_preds_list)
                            print(f"Confusion Matrix:\n{cm}\n")
                            if f1_score > best_valid_f1:
                                best_valid_f1 = f1_score
                                best_hyperparameters = {
                                    "dropout_prob": dropout_prob,
                                    "hidden_channels": hidden_channels,
                                    "lr": lr,
                                    "num_layers": num_layers,
                                    "epoch": epoch,
                                    "sampling_ratio": sampling_ratio
                                    }
    #Using the best hyperparameters to train on combined train+valid and evaluate on test
    dropout_prob = best_hyperparameters['dropout_prob']
    hidden_channels = best_hyperparameters['hidden_channels']
    lr = best_hyperparameters['lr']
    num_layers = best_hyperparameters['num_layers']
    sampling_ratio=best_hyperparameters['sampling_ratio']
    train_indices = zero_one_indices[train_valid_indices[train_indices]]
    valid_indices = zero_one_indices[train_valid_indices[valid_indices]]
    test_indices = zero_one_indices[test_indices]
    train_edge_index = data.edge_index[:, train_indices]
    train_edge_labels = data.edge_labels[train_indices]
    valid_edge_index = data.edge_index[:, valid_indices]
    valid_edge_labels = data.edge_labels[valid_indices]
    test_edge_index = data.edge_index[:, test_indices]
    test_edge_labels = data.edge_labels[test_indices]
    train_data = Data(x=data.x, edge_index=train_edge_index, edge_labels=train_edge_labels, edge_attr=data.edge_attr[train_indices])
    valid_data = Data(x=data.x, edge_index=valid_edge_index, edge_labels=valid_edge_labels, edge_attr=data.edge_attr[valid_indices])
    test_data = Data(x=data.x, edge_index=test_edge_index, edge_labels=test_edge_labels, edge_attr=data.edge_attr[test_indices])
    train_data = train_data.to(device)
    valid_data = valid_data.to(device)
    test_data = test_data.to(device)
    train_dataset = [Data(x=data.x, edge_index=train_data.edge_index[:, i:i+1], edge_label=train_data.edge_labels[i:i+1], edge_attr=train_data.edge_attr[i:i+1]) for i in range(train_data.edge_index.shape[1])]
    valid_dataset = [Data(x=data.x, edge_index=valid_data.edge_index[:, i:i+1], edge_label=valid_data.edge_labels[i:i+1], edge_attr=valid_data.edge_attr[i:i+1]) for i in range(valid_data.edge_index.shape[1])]
    test_dataset = [Data(x=data.x, edge_index=test_data.edge_index[:, i:i+1], edge_label=test_data.edge_labels[i:i+1], edge_attr=test_data.edge_attr[i:i+1]) for i in range(test_data.edge_index.shape[1])]
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    # Training the model using best hyperparameters
    model = EdgeClassifier(num_features=128, hidden_channels=hidden_channels, num_classes=2, num_layers=num_layers,dropout_prob=dropout_prob).to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_record=[]
    best_f1 = 0.0
    best_model_path = 'data/best_model_folder/best_model'
    for epoch in tqdm(range(25), desc="Epoch", leave=False):
        model.train()
        loss_add=0
        loss_n=0
        for batch in tqdm(train_loader, desc="Train", leave=False):
            batch = batch.to(device)
            optimizer.zero_grad()
            train_logits = model(batch)
            loss = criterion(train_logits, batch.edge_label)
            loss.backward()
            optimizer.step()
            loss_add+=loss
            loss_n+=1
        loss_avg=loss_add/loss_n
        loss_record.append(loss_avg)
        print(f'loss_avg:{loss_avg}')
        with open("log/log_graph.txt", "a") as log_file:
            log_file.write(f"epoch={epoch}, loss={loss_avg}\n")
        # Evaluating on the test set
        model.eval()
        valid_probs_list = []
        valid_preds_list = []
        valid_labels_list = []
        with torch.no_grad():
            for batch in tqdm(valid_loader, desc="Valid", leave=False):
                batch = batch.to(device)
                valid_logits = model(batch)
                valid_logits_binary = valid_logits[:, :2]
                valid_preds_binary = valid_logits_binary.argmax(dim=-1)
                valid_probs = F.softmax(valid_logits_binary, dim=-1)[:, 1]
                valid_probs_list.extend(valid_probs.cpu().tolist())
                valid_preds_list.extend(valid_preds_binary.cpu().tolist())
                valid_labels_list.extend(batch.edge_label.cpu().tolist())
        precision, recall, f1_score, _ = precision_recall_fscore_support(valid_labels_list, valid_preds_list, average='macro')
        cm = confusion_matrix(valid_labels_list, valid_preds_list)
        print(f"Confusion Matrix:\n{cm}\n")
        if f1_score > best_f1:
            best_f1 = f1_score
            torch.save(model.state_dict(), best_model_path)
    model.load_state_dict(torch.load(best_model_path))
    model.eval()
    test_probs_list = []
    test_preds_list = []
    test_labels_list = []
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Test", leave=False):
            batch = batch.to(device)
            test_logits = model(batch)
            test_logits_binary = test_logits[:, :2]
            test_preds_binary = test_logits_binary.argmax(dim=-1)
            test_probs = F.softmax(test_logits_binary, dim=-1)[:, 1]
            test_probs_list.extend(test_probs.cpu().tolist())
            test_preds_list.extend(test_preds_binary.cpu().tolist())
            test_labels_list.extend(batch.edge_label.cpu().tolist())
    numbers+=1
    precision, recall, thresholds_pr = precision_recall_curve(test_labels_list, test_probs_list)
    fpr, tpr, thresholds_roc = roc_curve(test_labels_list, test_probs_list)
    auc_roc = auc(fpr, tpr)
    auc_pr = auc(recall, precision)
    plt.figure()
    plt.plot(fpr, tpr, label=f"ROC curve (area = {auc_roc:.2f})")
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic Curve')
    plt.legend(loc="lower right")
    plt.savefig(f"data/figure/k{numbers}_ROC_curve_new.png")
    plt.close()
    plt.figure()
    plt.plot(recall, precision, label=f"PR curve (area = {auc_pr:.2f})")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc="lower right")
    plt.savefig(f"data/figure/k{numbers}_data_PR_curve_new.png")
    plt.close()
    precision, recall, f1_score, _ = precision_recall_fscore_support(test_labels_list, test_preds_list, average='macro')
    cm = confusion_matrix(test_labels_list, test_preds_list)
    # Logging the results
    print(f"Test Results - Precision: {precision:.2f}, Recall: {recall:.2f}, F1 Score: {f1_score:.2f}")
    print(f"Confusion Matrix:\n{cm}\n")
    with open("log/log_graph.txt", "a") as log_file:
        log_file.write(f"k={numbers}Using best hyperparameters: dropout_prob:{dropout_prob},hidden_channels:{hidden_channels},lr:{lr},num_layers:{num_layers}\n")
        log_file.write(f"Test Results - Precision: {precision:.2f}, Recall: {recall:.2f}, F1 Score: {f1_score:.2f}\n")
    k_loss+=1
    x = list(range(1, len(loss_record)+1))
    loss_record_np = [l.cpu().detach().numpy() for l in loss_record]
    plt.plot(x, loss_record_np, marker='o', linestyle='-')
    plt.title('Loss over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss Value')
    plt.savefig(f'data/loss_log_0906_graph_isa_{k_loss}_new.png', dpi=300)
