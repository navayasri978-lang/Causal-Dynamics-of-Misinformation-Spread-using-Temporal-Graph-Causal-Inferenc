TGCI: Temporal Graph Causal Inference

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from torch_geometric_temporal.signal import DynamicGraphTemporalSignal
from torch_geometric_temporal.nn.recurrent import TGNMemory, TGNMessagePassing
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors
from econml.dml import LinearDML
from sklearn.metrics import mean_absolute_error, r2_score

# ==========================================
# 1. Temporal Graph Encoder (TGE)
# ==========================================
class TemporalGraphEncoder(nn.Module):
    def __init__(self, node_features, memory_dim=64, time_dim=16, embedding_dim=128):
        super().__init__()
        self.memory_dim = memory_dim
        self.time_dim = time_dim
        self.embedding_dim = embedding_dim
        self.memory = TGNMemory(
            num_nodes=node_features,
            raw_message_dim=embedding_dim,
            memory_dimension=memory_dim,
            time_dimension=time_dim
        )
        self.gnn = TGNMessagePassing(
            in_channels=memory_dim,
            out_channels=embedding_dim
        )

    def forward(self, x, edge_index, edge_attr, t):
        memory = self.memory(x, t)
        out = self.gnn(memory, edge_index, edge_attr)
        return out

# ==========================================
# 2. Causal Identification Layer (CIL)
# ==========================================
def causal_matching(z, treatment, covariates):
    """
    Estimate propensity scores and perform nearest-neighbor matching.
    """
    logit = LogisticRegression(max_iter=1000)
    logit.fit(covariates, treatment)
    propensity = logit.predict_proba(covariates)[:, 1]

    # Nearest neighbor matching
    treated_idx = np.where(treatment == 1)[0]
    control_idx = np.where(treatment == 0)[0]
    nbrs = NearestNeighbors(n_neighbors=1).fit(propensity[control_idx].reshape(-1, 1))
    _, indices = nbrs.kneighbors(propensity[treated_idx].reshape(-1, 1))
    matched_pairs = list(zip(treated_idx, control_idx[indices[:, 0]]))
    return matched_pairs, propensity

# ==========================================
# 3. Counterfactual Simulation Engine (CSE)
# ==========================================
def simulate_counterfactual(model, data, intervention_mask):
    """
    Predict outcomes under no-intervention condition.
    """
    y_true, y_pred_cf = [], []
    for t in range(len(data.features)):
        x_t = data.features[t]
        edge_index_t = data.edge_indices[t]
        edge_attr_t = data.edge_weights[t]
        y = data.targets[t]
        pred = model(x_t, edge_index_t, edge_attr_t, t)
        y_true.append(y.mean().item())
        y_pred_cf.append(pred.mean().item() * (1 - intervention_mask[t]))  # simulate no intervention
    return np.array(y_true), np.array(y_pred_cf)

# ==========================================
# 4. TGCI Training + Causal Estimation
# ==========================================
def train_tgci(data):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = TemporalGraphEncoder(node_features=data.features[0].shape[1]).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    print("Training TGCI model...")
    for epoch in range(5):  # Simplified training loop
        total_loss = 0
        for t in range(len(data.features)):
            x_t = data.features[t].to(device)
            edge_index_t = data.edge_indices[t].to(device)
            edge_attr_t = data.edge_weights[t].to(device)
            y_t = data.targets[t].to(device)

            optimizer.zero_grad()
            y_pred = model(x_t, edge_index_t, edge_attr_t, t)
            loss = nn.MSELoss()(y_pred.mean(), y_t.mean())
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}: Loss={total_loss:.4f}")
    return model

# ==========================================
# 5. Example: Running TGCI on Dummy Data
# ==========================================
def generate_synthetic_temporal_graph(num_nodes=100, timesteps=10):
    edge_indices, edge_weights, features, targets = [], [], [], []
    for t in range(timesteps):
        edges = torch.randint(0, num_nodes, (2, num_nodes))
        weights = torch.rand(num_nodes)
        x = torch.rand(num_nodes, 8)
        y = torch.rand(num_nodes, 1)
        edge_indices.append(edges)
        edge_weights.append(weights)
        features.append(x)
        targets.append(y)
    return DynamicGraphTemporalSignal(edge_indices, edge_weights, features, targets)

if __name__ == "__main__":
    data = generate_synthetic_temporal_graph()
    tgci_model = train_tgci(data)

    intervention_mask = np.random.randint(0, 2, size=len(data.features))
    y_true, y_cf = simulate_counterfactual(tgci_model, data, intervention_mask)

    # Causal Effect Estimation (ATE)
    att = np.mean(y_true - y_cf)
    print(f"\nEstimated ATT (Causal Effect): {att:.4f}")
