import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_recall_fscore_support, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import os
import copy

# ============================================================
# CONFIGURATION
# ============================================================
NUM_CLIENTS = 2          # Number of federated clients
ALPHA = 0.5              # Dirichlet concentration parameter
                         #   alpha -> 0   : highly heterogeneous (each client gets 1 class)
                         #   alpha -> inf : homogeneous (IID)
                         #   Typical values: 0.1 (very hetero), 0.5 (moderate), 100 (near-IID)
NUM_ROUNDS = 5          # Number of federated rounds
LOCAL_EPOCHS = 40        # Local epochs per round per client (best for LSTM)
BATCH_SIZE = 32
LR = 0.001
WEIGHT_DECAY = 1e-5
HIDDEN_DIM = 128         # LSTM hidden dimension (from centralized setting)
NUM_LAYERS = 2
DROPOUT = 0.1
THRESHOLD = 0.3          # Sigmoid threshold for binary classification (best for LSTM)
SEED = 42

np.random.seed(SEED)
torch.manual_seed(SEED)

# ============================================================
# DATA DISTRIBUTION FUNCTIONS
# ============================================================

def dirichlet_split(y, num_clients, alpha, seed=42, min_samples=10):
    rng = np.random.default_rng(seed)
    classes = np.unique(y)
    client_indices = [[] for _ in range(num_clients)]

    for cls in classes:
        cls_idx = np.where(y == cls)[0]
        rng.shuffle(cls_idx)

        # Retry until all clients get at least min_samples (or cls has enough)
        for attempt in range(100):
            proportions = rng.dirichlet(alpha=np.repeat(alpha, num_clients))
            split_points = (np.cumsum(proportions) * len(cls_idx)).astype(int)[:-1]
            splits = np.split(cls_idx, split_points)
            if all(len(s) >= min_samples or len(cls_idx) < num_clients * min_samples for s in splits):
                break

        for client_id, split in enumerate(splits):
            client_indices[client_id].extend(split.tolist())

    for client_id in range(num_clients):
        rng.shuffle(client_indices[client_id])
        client_indices[client_id] = np.array(client_indices[client_id], dtype=np.int64)

    return client_indices


def equal_split(y, num_clients, seed=42):
    """
    Homogeneous (IID) split: shuffles all samples and divides them
    into num_clients equal (or near-equal) chunks.
    Every client ends up with roughly the same class distribution
    as the overall dataset — no Dirichlet involved.

    Parameters
    ----------
    y           : array-like of labels
    num_clients : int
    seed        : int

    Returns
    -------
    client_indices : list of np.ndarray  (one per client)
    """
    rng = np.random.default_rng(seed)
    all_idx = np.arange(len(y))
    rng.shuffle(all_idx)
    # np.array_split handles uneven sizes gracefully
    splits = np.array_split(all_idx, num_clients)
    return [s.astype(np.int64) for s in splits]


def print_client_distribution(client_indices, y, num_clients):
    """Prints class distribution per client."""
    print(f"\n{'─'*60}")
    print(f"{'Client':<10} {'Total':>8} {'Non-Mal (0)':>12} {'Malicious (1)':>14} {'Mal%':>8}")
    print(f"{'─'*60}")
    for i in range(num_clients):
        idx = client_indices[i]
        labels = y[idx]
        n_total = len(labels)
        n_mal = np.sum(labels == 1)
        n_nonmal = np.sum(labels == 0)
        pct = 100 * n_mal / n_total if n_total > 0 else 0
        print(f"Client {i:<4} {n_total:>8} {n_nonmal:>12} {n_mal:>14} {pct:>7.1f}%")
    print(f"{'─'*60}\n")


# ============================================================
# LSTM MODEL
# ============================================================

class LSTMClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, num_layers=2, dropout=0.1):
        super(LSTMClassifier, self).__init__()
        
        # Input projection
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Classification head
        self.fc1 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim // 2, 1)  # Binary classification with sigmoid
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # x shape: (batch_size, input_dim)
        # Add sequence dimension: (batch_size, 1, input_dim)
        x = x.unsqueeze(1)
        
        # Project to hidden dimension
        x = self.input_projection(x)
        
        # Apply LSTM
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        # Take the last output from LSTM
        x = lstm_out[:, -1, :]
        
        # Classification head
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x


# ============================================================
# FEDERATED AVERAGING (FedAvg)
# ============================================================

def fedavg(global_model, client_models, client_sizes):
    """
    Aggregates client models using weighted FedAvg.
    Weights are proportional to each client's dataset size.
    """
    total_samples = sum(client_sizes)
    global_state = global_model.state_dict()

    for key in global_state:
        global_state[key] = sum(
            client_models[i].state_dict()[key] * (client_sizes[i] / total_samples)
            for i in range(len(client_models))
        )

    global_model.load_state_dict(global_state)
    return global_model


# ============================================================
# LOCAL TRAINING
# ============================================================

def local_train(model, train_loader, local_epochs, device, lr=LR, weight_decay=WEIGHT_DECAY):
    model.train()
    criterion = nn.BCEWithLogitsLoss()  # Sigmoid + BCE for binary classification
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    for epoch in range(local_epochs):
        epoch_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device).float(), y_batch.to(device).float().unsqueeze(1)
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        # Apply scheduler based on average epoch loss
        avg_epoch_loss = epoch_loss / len(train_loader)
        scheduler.step(avg_epoch_loss)

    return model


# ============================================================
# EVALUATION
# ============================================================

def evaluate(model, data_loader, device):
    model.eval()
    criterion = nn.BCEWithLogitsLoss()
    total_loss = 0.0
    all_preds, all_labels, all_probs = [], [], []

    with torch.no_grad():
        for X_batch, y_batch in data_loader:
            X_batch, y_batch = X_batch.to(device).float(), y_batch.to(device).float().unsqueeze(1)
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            total_loss += loss.item()
            # Apply sigmoid and threshold 0.3
            probs = torch.sigmoid(outputs)
            preds = (probs > THRESHOLD).long().squeeze()
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y_batch.squeeze().cpu().numpy().astype(int))
            all_probs.extend(probs.squeeze().cpu().numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    avg_loss = total_loss / len(data_loader)
    return avg_loss, accuracy, all_preds, all_labels, all_probs


# ============================================================
# MAIN
# ============================================================

def run_federated(setting_label, X_train, X_test, y_train, y_test,
                  feature_cols, device, num_rounds=NUM_ROUNDS,
                  local_epochs=LOCAL_EPOCHS, num_clients=NUM_CLIENTS,
                  mode='heterogeneous', alpha=0.5):
    """
    Runs one full federated training experiment with LSTM.

    Parameters
    ----------
    setting_label : string label for logging/saving
    mode          : 'homogeneous'  -> equal split (true IID, no Dirichlet)
                    'heterogeneous' -> Dirichlet split with given alpha
    alpha         : Dirichlet concentration (only used when mode='heterogeneous')
                    e.g. 0.1=very skewed, 0.5=moderate
    """
    print(f"\n{'='*60}")
    if mode == 'homogeneous':
        print(f"  FEDERATED SETTING: {setting_label.upper()}  (Equal / IID split)")
    else:
        print(f"  FEDERATED SETTING: {setting_label.upper()}  (Dirichlet alpha={alpha})")
    print(f"  Clients={num_clients}, Rounds={num_rounds}, Local Epochs={local_epochs}")
    print(f"{'='*60}")

    # Distribute training data
    if mode == 'homogeneous':
        client_indices = equal_split(y_train, num_clients, seed=SEED)
    else:
        client_indices = dirichlet_split(y_train, num_clients, alpha, seed=SEED)
    print_client_distribution(client_indices, y_train, num_clients)

    # Build per-client DataLoaders
    client_loaders = []
    client_sizes = []
    for i in range(num_clients):
        idx = client_indices[i]
        if len(idx) == 0:
            print(f"  WARNING: Client {i} has no data — skipping.")
            client_loaders.append(None)
            client_sizes.append(0)
            continue

        X_c = torch.FloatTensor(X_train[idx])
        y_c = torch.LongTensor(y_train[idx])
        ds = TensorDataset(X_c, y_c)
        loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True)
        client_loaders.append(loader)
        client_sizes.append(len(idx))

    # Global test loader
    X_test_tensor = torch.FloatTensor(X_test)
    y_test_tensor = torch.LongTensor(y_test)
    test_loader = DataLoader(TensorDataset(X_test_tensor, y_test_tensor),
                             batch_size=BATCH_SIZE, shuffle=False)

    # Initialize global model (LSTM)
    global_model = LSTMClassifier(
        input_dim=len(feature_cols),
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT
    ).to(device)

    # Round-level tracking
    round_accuracies = []
    round_losses = []

    for rnd in range(1, num_rounds + 1):
        client_models = []
        active_clients = []

        for i in range(num_clients):
            if client_loaders[i] is None:
                continue

            # Give each client a fresh copy of the global model
            local_model = copy.deepcopy(global_model).to(device)
            local_model = local_train(local_model, client_loaders[i], local_epochs, device)
            client_models.append(local_model)
            active_clients.append(i)

        if not client_models:
            print(f"  Round {rnd}: No active clients.")
            continue

        # Aggregate
        active_sizes = [client_sizes[i] for i in active_clients]
        global_model = fedavg(global_model, client_models, active_sizes)

        # Evaluate global model
        loss, acc, preds, labels, probs = evaluate(global_model, test_loader, device)
        round_accuracies.append(acc)
        round_losses.append(loss)

        print(f"  Round {rnd:>3}/{num_rounds} | Loss: {loss:.4f} | Accuracy: {acc:.4f}")

    # Final evaluation
    final_loss, final_acc, final_preds, final_labels, final_probs = evaluate(global_model, test_loader, device)
    # Calculate all metrics
    final_probs = np.array(final_probs)
    final_preds_arr = np.array(final_preds)
    final_labels_arr = np.array(final_labels)
    
    # Binary metrics
    precision, recall, f1, _ = precision_recall_fscore_support(final_labels_arr, final_preds_arr, average='binary')
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(final_labels_arr, final_preds_arr, average='macro')
    precision_w, recall_w, f1_w, _ = precision_recall_fscore_support(final_labels_arr, final_preds_arr, average='weighted')
    
    # ROC-AUC and FPR/TPR
    roc_auc = roc_auc_score(final_labels_arr, final_probs)
    fpr, tpr, _ = roc_curve(final_labels_arr, final_probs)
    
    # Get per-class metrics
    tn = np.sum((final_preds_arr == 0) & (final_labels_arr == 0))
    fp = np.sum((final_preds_arr == 1) & (final_labels_arr == 0))
    fn = np.sum((final_preds_arr == 0) & (final_labels_arr == 1))
    tp = np.sum((final_preds_arr == 1) & (final_labels_arr == 1))
    
    fpr_calc = fp / (fp + tn) if (fp + tn) > 0 else 0
    tpr_calc = tp / (tp + fn) if (tp + fn) > 0 else 0

    alpha_label = 'equal/IID' if mode == 'homogeneous' else f'alpha={alpha}'
    print(f"\n{'='*120}")
    print(f"FINAL RESULTS — {setting_label.upper()} ({alpha_label})")
    print(f"Decision Threshold: {THRESHOLD}")
    print(f"{'='*120}")
    
    # Print detailed metrics table
    print(f"\n{'─'*120}")
    print(f"{'Metric':<25} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'FPR':<12} {'TPR':<12} {'ROC-AUC':<12}")
    print(f"{'─'*120}")
    print(f"{'Binary':<25} {final_acc:<12.4f} {precision:<12.4f} {recall:<12.4f} {f1:<12.4f} {fpr_calc:<12.4f} {tpr_calc:<12.4f} {roc_auc:<12.4f}")
    print(f"{'Macro':<25} {final_acc:<12.4f} {precision_macro:<12.4f} {recall_macro:<12.4f} {f1_macro:<12.4f} {fpr_calc:<12.4f} {tpr_calc:<12.4f} {roc_auc:<12.4f}")
    print(f"{'Weighted':<25} {final_acc:<12.4f} {precision_w:<12.4f} {recall_w:<12.4f} {f1_w:<12.4f} {fpr_calc:<12.4f} {tpr_calc:<12.4f} {roc_auc:<12.4f}")
    print(f"{'─'*120}")
    
    print(f"\n  Test Loss: {final_loss:.4f}")
    print(f"\n  Classification Report:")
    print(classification_report(final_labels_arr, final_preds_arr, target_names=['Non-Malicious', 'Malicious']))
    print(f"\n  Confusion Matrix:")
    print(confusion_matrix(final_labels_arr, final_preds_arr))

    metrics = {
        'setting': setting_label,
        'mode': mode,
        'alpha': alpha if mode == 'heterogeneous' else 'N/A',
        'accuracy': final_acc,
        'binary_precision': precision,
        'binary_recall': recall,
        'binary_f1': f1,
        'macro_precision': precision_macro,
        'macro_recall': recall_macro,
        'macro_f1': f1_macro,
        'weighted_precision': precision_w,
        'weighted_recall': recall_w,
        'weighted_f1': f1_w,
        'roc_auc': roc_auc,
        'fpr': fpr_calc,
        'tpr': tpr_calc,
        'round_accuracies': round_accuracies,
        'round_losses': round_losses,
    }
    return metrics


# ============================================================
# LOAD DATA
# ============================================================

file_path = "../../dataset/r4.2/ExtractedData/weekr4.2.csv"
data = pd.read_csv(file_path)

print("Data shape:", data.shape)

# Binary labels
data['insider_binary'] = (data['insider'] != 0).astype(int)
print(f"\nLabel distribution:\n{data['insider_binary'].value_counts()}")

# Features
exclude_cols = ['user', 'week', 'insider', 'insider_binary', 'starttime', 'endtime',
                'role', 'b_unit', 'f_unit', 'dept', 'team', 'ITAdmin']
feature_cols = [col for col in data.columns if col not in exclude_cols]
print(f"Number of features: {len(feature_cols)}")

X = data[feature_cols].fillna(0).values
y = data['insider_binary'].values

# Normalize
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=SEED, stratify=y
)

print(f"\n--- Train/Test Split ---")
print(f"Train: {len(y_train)} | Malicious: {np.sum(y_train==1)} | Non-Malicious: {np.sum(y_train==0)}")
print(f"Test : {len(y_test)}  | Malicious: {np.sum(y_test==1)} | Non-Malicious: {np.sum(y_test==0)}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\nUsing device: {device}")

os.makedirs(f'results/lstm/{NUM_CLIENTS}/round{NUM_ROUNDS}/epochs_{LOCAL_EPOCHS}/threshold_{THRESHOLD}', exist_ok=True)

# ============================================================
# RUN EXPERIMENTS
# ============================================================
# mode='homogeneous'   -> true equal split, NO Dirichlet at all
# mode='heterogeneous' -> Dirichlet split; alpha controls skew:
#   alpha=0.1 -> very heterogeneous (each client mostly one class)
#   alpha=0.5 -> moderately heterogeneous
#   alpha=1.0 -> somewhat more uniform

experiments = [
    {'mode': 'heterogeneous', 'alpha': 0.1,  'label': 'heterogeneous_strong'},
    {'mode': 'heterogeneous', 'alpha': 0.5,  'label': 'heterogeneous_moderate'},
    {'mode': 'heterogeneous', 'alpha': 1.0,  'label': 'heterogeneous_weak'},
    {'mode': 'homogeneous',   'alpha': None, 'label': 'homogeneous_iid'},
]

all_results = []
for exp in experiments:
    result = run_federated(
        setting_label=exp['label'],
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        feature_cols=feature_cols,
        device=device,
        num_rounds=NUM_ROUNDS,
        local_epochs=LOCAL_EPOCHS,
        num_clients=NUM_CLIENTS,
        mode=exp['mode'],
        alpha=exp['alpha'] if exp['alpha'] is not None else 0.5
    )
    all_results.append(result)

# ============================================================
# SAVE SUMMARY CSV
# ============================================================

summary_rows = []
for r in all_results:
    summary_rows.append({
        'Setting':            r['setting'],
        'Mode':               r['mode'],
        'Alpha':              r['alpha'],
        'Accuracy':           r['accuracy'],
        'Binary_Precision':   r['binary_precision'],
        'Binary_Recall':      r['binary_recall'],
        'Binary_F1':          r['binary_f1'],
        'Macro_Precision':    r['macro_precision'],
        'Macro_Recall':       r['macro_recall'],
        'Macro_F1':           r['macro_f1'],
        'Weighted_Precision': r['weighted_precision'],
        'Weighted_Recall':    r['weighted_recall'],
        'Weighted_F1':        r['weighted_f1'],
        'ROC_AUC':            r['roc_auc'],
        'FPR':                r['fpr'],
        'TPR':                r['tpr'],
    })

summary_df = pd.DataFrame(summary_rows)
print(f"\n{'='*140}")
print("SUMMARY ACROSS ALL SETTINGS")
print(f"{'='*140}")
print(summary_df.to_string(index=False))
summary_df.to_csv(f'results/lstm/{NUM_CLIENTS}/round{NUM_ROUNDS}/epochs_{LOCAL_EPOCHS}/threshold_{THRESHOLD}/federated_lstm_summary.csv', index=False)
print(f"\nSummary saved to 'results/lstm/{NUM_CLIENTS}/round{NUM_ROUNDS}/epochs_{LOCAL_EPOCHS}/threshold_{THRESHOLD}/federated_lstm_summary.csv'")

# ============================================================
# PLOT: Accuracy per round for each setting
# ============================================================

fig, axes = plt.subplots(1, len(experiments), figsize=(18, 5))
fig.suptitle(f'Federated LSTM Training — Accuracy per Round (N={NUM_CLIENTS} clients, Epochs={LOCAL_EPOCHS}, Threshold={THRESHOLD})', fontsize=14)

colors = ['tab:red', 'tab:orange', 'tab:green', 'tab:blue']
for ax, result, color in zip(axes, all_results, colors):
    rounds = list(range(1, len(result['round_accuracies']) + 1))
    ax.plot(rounds, result['round_accuracies'], marker='o', color=color, label='Accuracy', linewidth=2)
    subtitle = 'Equal/IID' if result['mode'] == 'homogeneous' else f"α={result['alpha']}"
    ax.set_title(f"{result['setting']}\n({subtitle})", fontsize=11)
    ax.set_xlabel('Round')
    ax.set_ylabel('Test Accuracy')
    ax.set_ylim(0.98, 1.0)
    ax.grid(True, alpha=0.3)
    ax.legend()

plt.tight_layout()
plt.savefig(f'results/lstm/{NUM_CLIENTS}/round{NUM_ROUNDS}/epochs_{LOCAL_EPOCHS}/threshold_{THRESHOLD}/federated_accuracy_per_round.png', dpi=300)
print(f"Plot saved to 'results/lstm/{NUM_CLIENTS}/round{NUM_ROUNDS}/epochs_{LOCAL_EPOCHS}/threshold_{THRESHOLD}/federated_accuracy_per_round.png'")

# ============================================================
# PLOT: Final metrics comparison across settings
# ============================================================

metrics_to_plot = ['Binary_F1', 'Macro_F1', 'Weighted_F1', 'Accuracy']
fig, axes = plt.subplots(1, len(metrics_to_plot), figsize=(16, 5))
fig.suptitle('Final Metrics Comparison Across Federated LSTM Settings', fontsize=14)

x = np.arange(len(all_results))
labels_x = [r['setting'].replace('_', '\n') for r in all_results]

for ax, metric in zip(axes, metrics_to_plot):
    values = summary_df[metric].values
    bars = ax.bar(x, values, color=['tab:red', 'tab:orange', 'tab:green', 'tab:blue'], edgecolor='black', width=0.6)
    ax.set_title(metric.replace('_', ' '))
    ax.set_xticks(x)
    ax.set_xticklabels(labels_x, fontsize=9)
    ax.set_ylim(0, 1)
    ax.grid(axis='y', alpha=0.3)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f'{val:.3f}', ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig(f'results/lstm/{NUM_CLIENTS}/round{NUM_ROUNDS}/epochs_{LOCAL_EPOCHS}/threshold_{THRESHOLD}/federated_metrics_comparison.png', dpi=300)
print(f"Comparison plot saved to 'results/lstm/{NUM_CLIENTS}/round{NUM_ROUNDS}/epochs_{LOCAL_EPOCHS}/threshold_{THRESHOLD}/federated_metrics_comparison.png'")

# ============================================================
# PLOT: Client data distribution heatmap (for each setting)
# ============================================================

fig, axes = plt.subplots(1, len(experiments), figsize=(6 * len(experiments), 4))
fig.suptitle('Client Class Distribution per Federated LSTM Setting', fontsize=14)

for ax, exp in zip(axes, experiments):
    if exp['mode'] == 'homogeneous':
        client_indices = equal_split(y_train, NUM_CLIENTS, seed=SEED)
        title_str = f"{exp['label']}\n(Equal / IID)"
    else:
        client_indices = dirichlet_split(y_train, NUM_CLIENTS, exp['alpha'], seed=SEED)
        title_str = f"{exp['label']}\n(α={exp['alpha']})"

    dist_matrix = np.zeros((NUM_CLIENTS, 2))
    for i, idx in enumerate(client_indices):
        if len(idx) == 0:
            continue
        labels = y_train[idx]
        dist_matrix[i, 0] = np.sum(labels == 0) / len(labels)
        dist_matrix[i, 1] = np.sum(labels == 1) / len(labels)

    im = ax.imshow(dist_matrix, aspect='auto', cmap='YlOrRd', vmin=0, vmax=1)
    ax.set_title(title_str, fontsize=10)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['Non-Malicious', 'Malicious'])
    ax.set_yticks(range(NUM_CLIENTS))
    ax.set_yticklabels([f'Client {i}' for i in range(NUM_CLIENTS)])
    plt.colorbar(im, ax=ax, label='Class proportion')

plt.tight_layout()
plt.savefig(f'results/lstm/{NUM_CLIENTS}/round{NUM_ROUNDS}/epochs_{LOCAL_EPOCHS}/threshold_{THRESHOLD}/client_distribution_heatmap.png', dpi=300)
print(f"Distribution heatmap saved to 'results/lstm/{NUM_CLIENTS}/round{NUM_ROUNDS}/epochs_{LOCAL_EPOCHS}/threshold_{THRESHOLD}/client_distribution_heatmap.png'")

print(f"\n{'='*120}")
print("FEDERATED LSTM TRAINING COMPLETE")
print(f"{'='*120}\n")
