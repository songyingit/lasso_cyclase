# **Example notebook for predicting cyclase and peptide substrate pairs with 5-fold CV**
# **This notebook demonstrates 5-fold cross-validation with different random seeds for train/val/test splits**
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import math
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score, balanced_accuracy_score, roc_auc_score, precision_score
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoModelForMaskedLM

# ### **1. Model Architecture**
# **Cross-Attention Mechanism**


class CrossAttention(nn.Module):
    # Implements a Cross-Attention mechanism to learn interactions between cyclase and peptide substrate embeddings.
    def __init__(self):
        super(CrossAttention, self).__init__()
        self.W_query = nn.Parameter(torch.rand(1280, 1280))  # Weight matrix for queries
        self.W_key = nn.Parameter(torch.rand(1280, 1280))    # Weight matrix for keys
        self.W_value = nn.Parameter(torch.rand(1280, 1280))  # Weight matrix for values

    def forward(self, x_1, x_2, attn_mask=None):
        # Compute queries, keys, and values
        """
        query: Tensor of shape [batch_size, len_peptide, esm_dim]
        value: Tensor of shape [batch_size, len_cyclase, esm_dim]
        attn_mask: Attention mask to ignore padding residues
        """
        query = torch.matmul(x_1, self.W_query)
        key = torch.matmul(x_2, self.W_key)
        value = torch.matmul(x_2, self.W_value)

        # Compute attention scores
        attn_scores = torch.matmul(query, key.transpose(-2, -1))
        scaled_attn_scores = attn_scores / math.sqrt(query.size(-1))
        
        if attn_mask is not None:
            attn_scores = attn_scores.masked_fill(attn_mask == 0, float('-inf'))  # mask the padding residues
        
        attn_weights = F.softmax(scaled_attn_scores, dim=-1)
        
        # Apply attention weights to values
        output = torch.matmul(attn_weights, value)
        
        return output, attn_weights


# **MLP Model with Cross-Attention**


# Define the MLP model with CrossAttention
class MLPWithAttention(nn.Module):
    def __init__(self, input_size):
        super(MLPWithAttention, self).__init__()
        self.cross_attention = CrossAttention()
        self.mlp = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.ReLU(),
            nn.Linear(512, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
    
    def forward(self, cyclase, substrate, cyclase_mask, substrate_mask):
        # Apply cross-attention mechanism
        attn_mask = torch.matmul(substrate_mask.unsqueeze(-1).float(), cyclase_mask.unsqueeze(1).float())
        x_1, _ = self.cross_attention(substrate, cyclase, attn_mask)   # reweighted cyclase embeddings

        # Average embeddings along the sequence length dimension
        x_1_avg = torch.mean(x_1, dim=1)
        substrate_avg = torch.mean(substrate, dim=1)
        
        # Concatenate averaged embeddings and pass through MLP layers
        x = torch.cat((x_1_avg, substrate_avg), dim=1)
        return self.mlp(x)


# ### **2. Embedding Extraction**


# Function to get representation from Vanilla ESM model
def get_rep_from_VanillaESM(sequence):
    token_ids = esm_tokenizer(sequence, return_tensors='pt').to(device)
    with torch.no_grad():
        results = esm_model(token_ids.input_ids, output_hidden_states=True)
    representations = results.hidden_states[33][0]
    return representations.cpu().numpy()

# Function to get representation from LassoESM model
def get_rep_from_LassoESM(sequence):
    token_ids = LassoESM_tokenizer(sequence, return_tensors='pt').to(device)
    with torch.no_grad():
        results = LassoESM_model(token_ids.input_ids, output_hidden_states=True)
    representations = results.hidden_states[33][0]
    return representations.cpu().numpy()

# Function to pad the ESM embeddings
def pad_esm_embedding(embedding, max_length):
    embedding_tensor = torch.tensor(embedding, dtype=torch.float32)
    pad_length = max_length - embedding.shape[0]
    padding = torch.zeros((pad_length, embedding.shape[1]), dtype=torch.float32)
    embedding_tensor = torch.cat((embedding_tensor, padding), dim=0)
    
    # Create attention mask
    attn_mask = torch.ones(max_length, dtype=torch.float32)
    attn_mask[embedding.shape[0]:] = 0
    attn_mask[0] = 0  # BOS token
    attn_mask[embedding.shape[0] - 1] = 0  # EOS token
    
    return embedding_tensor, attn_mask


# ### **3. Custom Dataset**


class CustomDataset(Dataset):
    # Custom PyTorch dataset for cyclase and peptide substrate pairs.
    def __init__(self, cyclase_sequences, substrate_sequences, labels, max_cyclase_length, max_substrate_length):
        self.cyclase_sequences = cyclase_sequences
        self.substrate_sequences = substrate_sequences
        self.labels = labels
        self.max_cyclase_length = max_cyclase_length
        self.max_substrate_length = max_substrate_length
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        cyclase_seq = self.cyclase_sequences[idx]
        substrate_seq = self.substrate_sequences[idx]
        
        # Pad cyclase sequence and create mask
        cyclase_embedding, cyclase_mask = pad_esm_embedding(get_rep_from_VanillaESM(cyclase_seq), self.max_cyclase_length)
        
        # Pad substrate sequence and create mask
        substrate_embedding, substrate_mask = pad_esm_embedding(get_rep_from_LassoESM(substrate_seq), self.max_substrate_length)
        
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        return cyclase_embedding, substrate_embedding, cyclase_mask, substrate_mask, label


# ### **4. Model Training**

def train_model(model, train_loader, val_loader, criterion, optimizer, epochs=50, fold=1):
    """Fine-tunes the model and saves the best version based on validation loss"""
    min_val_loss = np.inf
    patience = 5
    patience_counter = 0
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for cyclase, substrate, cyclase_mask, substrate_mask, labels in train_loader:
            cyclase, substrate, cyclase_mask, substrate_mask, labels = cyclase.to(device), substrate.to(device), cyclase_mask.to(device), substrate_mask.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(cyclase, substrate, cyclase_mask, substrate_mask)
            loss = criterion(outputs, labels.unsqueeze(1))
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        train_loss = running_loss / len(train_loader)
        
        # Evaluate on validation data
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for cyclase, substrate, cyclase_mask, substrate_mask, labels in val_loader:
                cyclase, substrate, cyclase_mask, substrate_mask, labels = cyclase.to(device), substrate.to(device), cyclase_mask.to(device), substrate_mask.to(device), labels.to(device)
                outputs = model(cyclase, substrate, cyclase_mask, substrate_mask)
                loss = criterion(outputs, labels.unsqueeze(1))
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        print(f"Fold {fold} - Epoch [{epoch+1}/{epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        # Save best model
        if min_val_loss > val_loss:
            print(f'Fold {fold} - Val Loss Decreased({min_val_loss:.4f} to {val_loss:.4f}) Saving The Model')
            min_val_loss = val_loss
            torch.save(model.state_dict(), f'finetuned_FusC_Fus_model_fold{fold}.pth')
            patience_counter = 0
        else:
            patience_counter += 1
            
        # Early stopping
        if patience_counter >= patience:
            print(f"Fold {fold} - Early stopping at epoch {epoch+1} (no improvement for {patience} epochs)")
            break
    
    return min_val_loss


# ### **5. Model evaluation**

def evaluate_model(model, dataloader):
    """Evaluates the model using balanced accuracy, recall, AUC, and precision."""
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for cyclase, substrate, cyclase_mask, substrate_mask, labels in dataloader:
            cyclase, substrate, cyclase_mask, substrate_mask, labels = cyclase.to(device), substrate.to(device), cyclase_mask.to(device), substrate_mask.to(device), labels.to(device)
            outputs = model(cyclase, substrate, cyclase_mask, substrate_mask)
            preds = (outputs > 0.5).float()
            
            all_preds.extend(preds.cpu().numpy().flatten().tolist())
            all_labels.extend(labels.cpu().numpy().flatten().tolist())
            all_probs.extend(outputs.cpu().numpy().flatten().tolist())
    
    balanced_accuracy = balanced_accuracy_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds)
    auc = roc_auc_score(all_labels, all_probs)
    precision = precision_score(all_labels, all_preds)
    
    return balanced_accuracy, recall, auc, precision, all_preds, all_probs


if __name__ == "__main__":

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load ESM models
    print("\nLoading ESM models...")
    esm_model = AutoModelForMaskedLM.from_pretrained("facebook/esm2_t33_650M_UR50D").to(device)
    esm_tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D")
    esm_model.eval()
    
    LassoESM_model = AutoModelForMaskedLM.from_pretrained("ShuklaGroupIllinois/LassoESM").to(device)
    LassoESM_tokenizer = AutoTokenizer.from_pretrained("ShuklaGroupIllinois/LassoESM")
    LassoESM_model.eval()

    # Load FusC-Fus dataset
    print("\nLoading dataset...")
    data = pd.read_excel('../../../data/process_data/FusA_FusC_variants_processed.xlsx')
    Cyclase_seq = data.iloc[:, 1].tolist()
    substrate_seq = data.iloc[:, 0].tolist()
    labels = data.iloc[:, 2].tolist()

    # Calculate max lengths for padding
    max_cyclase_length = max(len(seq) for seq in Cyclase_seq) + 2
    max_substrate_length = max(len(seq) for seq in substrate_seq) + 2
    
    print(f"Dataset size: {len(labels)}")
    print(f"Max cyclase length: {max_cyclase_length}")
    print(f"Max substrate length: {max_substrate_length}")

    # Define 5 different random seeds for 5-fold CV
    random_seeds = [42, 123, 456, 789, 1024]
    
    # Store all results
    all_fold_results = []
    
    # Loop through each fold with different random seed
    for fold, seed in enumerate(random_seeds, start=1):
        print(f"\n{'='*80}")
        print(f"STARTING FOLD {fold}/5 (Random Seed: {seed})")
        print(f"{'='*80}")
        
        # Split data: 70% train, 15% val, 15% test with current seed
        Cyclase_train, Cyclase_temp, Substrate_train, Substrate_temp, ys_train, ys_temp = train_test_split(
            Cyclase_seq, substrate_seq, labels, test_size=0.3, stratify=labels, random_state=seed)
        Cyclase_val, Cyclase_test, Substrate_val, Substrate_test, ys_val, ys_test = train_test_split(
            Cyclase_temp, Substrate_temp, ys_temp, test_size=0.5, stratify=ys_temp, random_state=seed)

        print(f"Fold {fold} - Dataset splits - Train: {len(ys_train)}, Val: {len(ys_val)}, Test: {len(ys_test)}")

        # Create datasets and dataloaders
        train_dataset = CustomDataset(Cyclase_train, Substrate_train, ys_train, max_cyclase_length, max_substrate_length)
        val_dataset = CustomDataset(Cyclase_val, Substrate_val, ys_val, max_cyclase_length, max_substrate_length)
        test_dataset = CustomDataset(Cyclase_test, Substrate_test, ys_test, max_cyclase_length, max_substrate_length)

        train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

        # Initialize model
        input_size = 1280 * 2
        model = MLPWithAttention(input_size).to(device)
        
        # Load pre-trained weights
        print(f"Fold {fold} - Loading pre-trained model weights...")
        model.load_state_dict(torch.load('../../train_on_general_ds/saved_best_model_from_xnm.pth', map_location=device))
        
        # Freeze cross-attention layers, fine-tune only MLP
        criterion = nn.BCELoss()
        for param in model.cross_attention.parameters():
            param.requires_grad = False
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.0001)
        
        print(f"\nFold {fold} - Starting Fine-tuning...")
        # Fine-tune the model
        best_val_loss = train_model(model, train_loader, val_loader, criterion, optimizer, epochs=100, fold=fold)
        
        # Load best model for evaluation
        print(f"\nFold {fold} - Loading Best Model for Evaluation...")
        model.load_state_dict(torch.load(f'finetuned_FusC_Fus_model_fold{fold}.pth', map_location=device))
        
        # Evaluate on test set
        print(f"\nFold {fold} - Evaluating on Test Set...")
        balanced_accuracy, recall, auc, precision, test_preds_list, test_probs_list = evaluate_model(model, test_loader)
        
        print(f"\nFold {fold} - Test Set Results:")
        print(f"  Balanced Accuracy: {balanced_accuracy:.4f}")
        print(f"  Recall (TPR): {recall:.4f}")
        print(f"  AUC: {auc:.4f}")
        print(f"  Precision: {precision:.4f}")
        
        # Save predictions for this fold
        fold_results_df = pd.DataFrame({
            'Fold': fold,
            'Random_Seed': seed,
            'Cyclase_sequence': Cyclase_test,
            'Substrate_sequence': Substrate_test,
            'True_Label': ys_test,
            'Predicted_Probability': test_probs_list,
            'Predicted_Label': [int(pred) for pred in test_preds_list]
        })
        
        all_fold_results.append(fold_results_df)
        
        # Save individual fold results
        fold_results_df.to_csv(f'Fus_FusC_pred_results_fold{fold}.csv', index=False)
        print(f"Fold {fold} - Predictions saved to 'Fus_FusC_pred_results_fold{fold}.csv'")
    
    # Combine all fold results into a single CSV
    print(f"\n{'='*80}")
    print("COMBINING ALL FOLD RESULTS")
    print(f"{'='*80}")
    
    combined_results_df = pd.concat(all_fold_results, ignore_index=True)
    combined_results_df.to_csv('Fus_FusC_pred_results_all_folds.csv', index=False)
    print(f"\nAll predictions saved to 'Fus_FusC_pred_results_all_folds.csv'")
    print(f"Total predictions across all folds: {len(combined_results_df)}")
    
    # Calculate and display summary statistics across all folds
    print(f"\n{'='*80}")
    print("SUMMARY STATISTICS ACROSS ALL FOLDS")
    print(f"{'='*80}")
    
    summary_stats = []
    for fold in range(1, 6):
        fold_data = combined_results_df[combined_results_df['Fold'] == fold]
        fold_accuracy = balanced_accuracy_score(fold_data['True_Label'], fold_data['Predicted_Label'])
        fold_recall = recall_score(fold_data['True_Label'], fold_data['Predicted_Label'])
        fold_auc = roc_auc_score(fold_data['True_Label'], fold_data['Predicted_Probability'])
        fold_precision = precision_score(fold_data['True_Label'], fold_data['Predicted_Label'])
        
        summary_stats.append({
            'Fold': fold,
            'Random_Seed': random_seeds[fold-1],
            'Balanced_Accuracy': fold_accuracy,
            'Recall': fold_recall,
            'AUC': fold_auc,
            'Precision': fold_precision
        })
    
    summary_df = pd.DataFrame(summary_stats)
    summary_df.to_csv('Fus_FusC_summary_statistics.csv', index=False)
    
    print("\nPer-Fold Performance:")
    print(summary_df.to_string(index=False))
    
    print("\nAverage Performance Across Folds:")
    print(f"  Balanced Accuracy: {summary_df['Balanced_Accuracy'].mean():.4f} ± {summary_df['Balanced_Accuracy'].std():.4f}")
    print(f"  Recall: {summary_df['Recall'].mean():.4f} ± {summary_df['Recall'].std():.4f}")
    print(f"  AUC: {summary_df['AUC'].mean():.4f} ± {summary_df['AUC'].std():.4f}")
    print(f"  Precision: {summary_df['Precision'].mean():.4f} ± {summary_df['Precision'].std():.4f}")
    
    print(f"\nSummary statistics saved to 'Fus_FusC_summary_statistics.csv'")
    print("\n=== EXPERIMENT COMPLETE ===")