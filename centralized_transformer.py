import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_recall_fscore_support, roc_auc_score
import matplotlib.pyplot as plt
import os

# Load data
file_path = "../dataset/r4.2/ExtractedData/weekr4.2.csv"
data = pd.read_csv(file_path)

print("Data shape:", data.shape)
print("Columns:", data.columns.tolist())

# Convert multi-class labels (0, 1, 2, 3) to binary (0, 1)
# Labels 1, 2, 3 are malicious -> converted to 1
# Label 0 remains non-malicious -> remains 0
data['insider_binary'] = (data['insider'] != 0).astype(int)

print(f"Label distribution:")
print(data['insider_binary'].value_counts())

# Select feature columns (exclude user, week, and label columns)
exclude_cols = ['user', 'week', 'insider', 'insider_binary', 'starttime', 'endtime', 'role', 'b_unit', 'f_unit', 'dept', 'team', 'ITAdmin']
feature_cols = [col for col in data.columns if col not in exclude_cols]

print(f"Number of features: {len(feature_cols)}")

# Prepare features and labels
X = data[feature_cols].fillna(0).values
y = data['insider_binary'].values

# Normalize features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Train/test split (80/20)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Print train/test split statistics
print(f"\n--- Train/Test Split Statistics ---")
train_malicious = np.sum(y_train == 1)
train_non_malicious = np.sum(y_train == 0)
test_malicious = np.sum(y_test == 1)
test_non_malicious = np.sum(y_test == 0)

print(f"Training Set:")
print(f"  Malicious Instances: {train_malicious}")
print(f"  Non-Malicious Instances: {train_non_malicious}")
print(f"  Total Instances: {len(y_train)}")

print(f"Test Set:")
print(f"  Malicious Instances: {test_malicious}")
print(f"  Non-Malicious Instances: {test_non_malicious}")
print(f"  Total Instances: {len(y_test)}")
print()

# Convert to PyTorch tensors
X_train_tensor = torch.FloatTensor(X_train)
X_test_tensor = torch.FloatTensor(X_test)
y_train_tensor = torch.FloatTensor(y_train).unsqueeze(1)  # For sigmoid: shape (batch, 1)
y_test_tensor = torch.FloatTensor(y_test).unsqueeze(1)   # For sigmoid: shape (batch, 1)

# Create DataLoaders
batch_size = 32
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Transformer Model
class TransformerClassifier(nn.Module):
    def __init__(self, input_dim, num_heads=8, num_layers=2, hidden_dim=256, dropout=0.1):
        super(TransformerClassifier, self).__init__()
        
        # Input projection
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        
        # Positional encoding
        self.positional_encoding = nn.Parameter(torch.randn(1, 1, hidden_dim))
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
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
        
        # Add positional encoding
        x = x + self.positional_encoding
        
        # Apply transformer
        x = self.transformer_encoder(x)
        
        # Take the mean of the sequence
        x = x.mean(dim=1)
        
        # Classification head
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x

# Initialize device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Create results directory if it doesn't exist
os.makedirs('results', exist_ok=True)

# Training function
def train_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(train_loader)

# Evaluation function
def evaluate(model, data_loader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for X_batch, y_batch in data_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            total_loss += loss.item()
            
            # Get probabilities using sigmoid
            probs = torch.sigmoid(outputs).squeeze()
            all_probs.extend(probs.cpu().numpy())
            
            # Default prediction with threshold 0.5
            preds = (probs >= 0.5).long()
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y_batch.squeeze().cpu().numpy())
    
    all_probs = np.array(all_probs)
    accuracy = accuracy_score(all_labels, all_preds)
    return total_loss / len(data_loader), accuracy, all_preds, all_labels, all_probs

# Function to evaluate different decision thresholds
def evaluate_with_confidence_thresholds(probs, labels, thresholds):
    """
    Evaluate predictions with different decision thresholds.
    For each threshold, reclassify ALL samples: if prob > threshold → malicious (1), else → non-malicious (0)
    """
    results = {}
    
    for threshold in thresholds:
        # Reclassify ALL samples based on threshold
        reclassified_preds = (probs > threshold).astype(int)
        
        # Calculate confusion matrix values
        tn, fp, fn, tp = confusion_matrix(labels, reclassified_preds).ravel()
        
        # Calculate FPR and TPR
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
        
        # Calculate ROC-AUC (using probabilities, not discrete predictions)
        try:
            roc_auc = roc_auc_score(labels, probs)
        except:
            roc_auc = 0.0
        
        precision, recall, f1, _ = precision_recall_fscore_support(labels, reclassified_preds, average='binary')
        precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(labels, reclassified_preds, average='macro')
        precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(labels, reclassified_preds, average='weighted')
        accuracy = accuracy_score(labels, reclassified_preds)
        
        # Count predictions in each class
        num_predicted_malicious = np.sum(reclassified_preds == 1)
        
        results[threshold] = {
            'num_samples': len(labels),
            'predicted_malicious': num_predicted_malicious,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'precision_macro': precision_macro,
            'recall_macro': recall_macro,
            'f1_macro': f1_macro,
            'precision_weighted': precision_weighted,
            'recall_weighted': recall_weighted,
            'f1_weighted': f1_weighted,
            'fpr': fpr,
            'tpr': tpr,
            'roc_auc': roc_auc
        }
    
    return results

# Define confidence thresholds
confidence_thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

print(f"\n{'='*60}")
print(f"Training separate models for different epoch counts...")
print(f"{'='*60}")

# Train for different epoch counts
epoch_counts = list(range(10, 110, 10))  # [10, 20, 30, ..., 100]

for num_epochs in epoch_counts:
    print(f"\n{'='*60}")
    print(f"Training NEW model for {num_epochs} epochs...")
    print(f"{'='*60}")
    
    # Initialize fresh model for each epoch count
    model = TransformerClassifier(
        input_dim=len(feature_cols),
        num_heads=8,
        num_layers=2,
        hidden_dim=256,
        dropout=0.1
    ).to(device)
    
    # Loss and optimizer
    criterion = nn.BCEWithLogitsLoss()  # For binary classification with sigmoid
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    # Training loop
    train_losses = []
    val_losses = []
    
    for epoch in range(num_epochs):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_accuracy, _, _, _ = evaluate(model, test_loader, criterion, device)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        scheduler.step(val_loss)

        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")
    
    # Evaluate final model on test set
    test_loss, test_accuracy, test_preds, test_labels, test_probs = evaluate(model, test_loader, criterion, device)
    
    # Evaluate with different decision thresholds
    threshold_results = evaluate_with_confidence_thresholds(test_probs, test_labels, confidence_thresholds)
    
    print(f"\n{'='*80}")
    print(f"RESULTS FOR {num_epochs} EPOCHS - Decision Threshold Analysis")
    print(f"{'='*80}")
    print(f"(Each threshold reclassifies ALL {len(test_labels)} samples: if sigmoid > threshold → Malicious, else → Non-Malicious)\n")
    
    # Print header - split into two lines for readability
    header1 = f"{'Threshold':<12} {'Predicted':<12} {'Accuracy':<10} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'FPR':<10} {'TPR':<10} {'ROC-AUC':<10}"
    header2 = f"{'M-Prec':<10} {'M-Recall':<10} {'M-F1':<10} {'W-Prec':<10} {'W-Recall':<10} {'W-F1':<10}"
    print(header1)
    print(header2)
    print("-" * 156)
    
    # Print results for each threshold
    for threshold in confidence_thresholds:
        res = threshold_results[threshold]
        line1 = f"{threshold:<12.1f} {res['predicted_malicious']:<12} {res['accuracy']:<10.4f} {res['precision']:<12.4f} {res['recall']:<12.4f} {res['f1']:<12.4f} {res['fpr']:<10.4f} {res['tpr']:<10.4f} {res['roc_auc']:<10.4f}"
        line2 = f"{res['precision_macro']:<10.4f} {res['recall_macro']:<10.4f} {res['f1_macro']:<10.4f} {res['precision_weighted']:<10.4f} {res['recall_weighted']:<10.4f} {res['f1_weighted']:<10.4f}"
        print(line1 + " " + line2)
    
    print()

print(f"\n{'='*80}")
print("TRAINING COMPLETE - All epoch counts evaluated")
print(f"{'='*80}\n")