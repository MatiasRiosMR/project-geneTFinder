import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns  # agregado para estilos profesionales
import json
import os
from datetime import datetime

from model import ProteinClassifier, SimpleProteinClassifier

# Mapeo de aminoácidos a índices
AA_TO_IDX = {
    'A': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5,
    'G': 6, 'H': 7, 'I': 8, 'K': 9, 'L': 10,
    'M': 11, 'N': 12, 'P': 13, 'Q': 14, 'R': 15,
    'S': 16, 'T': 17, 'V': 18, 'W': 19, 'Y': 20,
    'X': 0, 'U': 0, 'B': 0, 'Z': 0, 'O': 0  # Unknown
}

class ProteinDataset(Dataset):
    def __init__(self, csv_file, max_length=1000):
        self.data = pd.read_csv(csv_file)
        self.max_length = max_length
        
    def __len__(self):
        return len(self.data)
    
    def encode_sequence(self, sequence):
        """Codifica secuencia de aminoácidos a índices"""
        encoded = [AA_TO_IDX.get(aa, 0) for aa in sequence.upper()]
        
        # Padding o truncado
        if len(encoded) > self.max_length:
            encoded = encoded[:self.max_length]
        else:
            encoded += [0] * (self.max_length - len(encoded))
        
        return torch.tensor(encoded, dtype=torch.long)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        sequence = self.encode_sequence(row['sequence'])
        label = torch.tensor(row['label'], dtype=torch.float32)
        return sequence, label

def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    predictions = []
    true_labels = []
    
    for sequences, labels in tqdm(dataloader, desc="Entrenamiento"):  # traducido
        sequences = sequences.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(sequences)
        loss = criterion(outputs, labels)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        predictions.extend((outputs > 0.5).cpu().numpy())
        true_labels.extend(labels.cpu().numpy())
    
    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(true_labels, predictions)
    
    return avg_loss, accuracy

def validate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    predictions = []
    true_labels = []
    probs = []
    
    with torch.no_grad():
        for sequences, labels in tqdm(dataloader, desc="Validación"):  # traducido
            sequences = sequences.to(device)
            labels = labels.to(device)
            
            outputs = model(sequences)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            probs.extend(outputs.cpu().numpy())
            predictions.extend((outputs > 0.5).cpu().numpy())
            true_labels.extend(labels.cpu().numpy())
    
    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(true_labels, predictions)
    precision = precision_score(true_labels, predictions)
    recall = recall_score(true_labels, predictions)
    f1 = f1_score(true_labels, predictions)
    auc = roc_auc_score(true_labels, probs)
    
    return avg_loss, accuracy, precision, recall, f1, auc

def plot_training_history(train_losses, val_losses, train_accs, val_accs):
    # Estilo profesional
    sns.set_theme(style="whitegrid")
    plt.rcParams.update({
        "figure.dpi": 120,
        "savefig.dpi": 300,
        "font.size": 12,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "legend.fontsize": 11
    })

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    epochs = list(range(1, len(train_losses) + 1))
    palette = sns.color_palette("tab10")

    # Pérdida
    ax1.plot(epochs, train_losses, label='Entrenamiento - Loss', marker='o', markersize=5, linewidth=2, color=palette[0])
    ax1.plot(epochs, val_losses, label='Validación - Loss', marker='s', markersize=5, linewidth=2, color=palette[1])
    ax1.set_xlabel('Época')
    ax1.set_ylabel('Pérdida')
    ax1.set_title('Pérdida de Entrenamiento y Validación')
    ax1.grid(alpha=0.3)
    ax1.legend(frameon=False)

    # Accuracy (mantener la palabra "Accuracy" en inglés)
    ax2.plot(epochs, train_accs, label='Entrenamiento', marker='o', markersize=5, linewidth=2, color=palette[2])
    ax2.plot(epochs, val_accs, label='Validación', marker='s', markersize=5, linewidth=2, color=palette[3])
    ax2.set_xlabel('Época')
    ax2.set_ylabel('Accuracy')  # mantenido en inglés según solicitud
    ax2.set_title('Training and Validation Accuracy')
    ax2.grid(alpha=0.3)
    ax2.legend(frameon=False)

    # Ajustes finales y guardado en alta resolución
    plt.tight_layout()
    plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
    plt.savefig('training_history.svg', bbox_inches='tight')
    print("✓ Gráficos profesionales guardados: training_history.png, training_history.svg")

def save_metadata(output_dir, metadata):
    """Guarda metadata en JSON y, si está disponible, en YAML."""
    os.makedirs(output_dir, exist_ok=True)
    json_path = Path(output_dir) / "metadata.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=4)

    try:
        import yaml
        yaml_path = Path(output_dir) / "metadata.yaml"
        with open(yaml_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(metadata, f, sort_keys=False, allow_unicode=True)
        print(f"✓ Metadata guardada: {json_path}, {yaml_path}")
    except Exception:
        print(f"✓ Metadata JSON guardada: {json_path} (PyYAML no disponible, solo JSON)")

def _encode_sequence_tensor(sequence, max_length=1000):
    """Codifica una secuencia de aminoácidos a tensor (batch size = 1)."""
    encoded = [AA_TO_IDX.get(aa, 0) for aa in sequence.upper()]
    if len(encoded) > max_length:
        encoded = encoded[:max_length]
    else:
        encoded += [0] * (max_length - len(encoded))
    return torch.tensor(encoded, dtype=torch.long).unsqueeze(0)  # shape (1, max_length)

def load_model_for_inference(model_path="best_model.pth", device=None,
                             vocab_size=21, embedding_dim=32, hidden_dim=64, dropout=0.5):
    """Carga el modelo guardado y lo deja en modo eval.
    Asegúrate de pasar los mismos hiperparámetros usados en entrenamiento si tu clase los requiere.
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ProteinClassifier(
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        dropout=dropout
    )
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model

def predict_sequence(model, sequence, max_length=1000, device=None):
    """Recibe un modelo en modo eval y una secuencia (string). Devuelve probabilidad y etiqueta."""
    if device is None:
        device = next(model.parameters()).device
    seq_tensor = _encode_sequence_tensor(sequence, max_length=max_length).to(device)
    with torch.no_grad():
        out = model(seq_tensor)  # se espera salida escalar/batch de probabilidades
    # Convertir a float
    try:
        prob = float(out.squeeze().cpu().item())
    except Exception:
        prob = float(np.asarray(out.cpu())[0])
    label = int(prob > 0.5)
    return {"probability": prob, "label": label}

def predict_from_checkpoint(sequence, model_path="best_model.pth", max_length=1000,
                            vocab_size=21, embedding_dim=32, hidden_dim=64, dropout=0.5):
    """Conveniencia: carga checkpoint y predice una secuencia (útil para la ventana de predecir)."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = load_model_for_inference(model_path=model_path, device=device,
                                     vocab_size=vocab_size, embedding_dim=embedding_dim,
                                     hidden_dim=hidden_dim, dropout=dropout)
    return predict_sequence(model, sequence, max_length=max_length, device=device)

def generate_dummy_dataset(output_path, n_samples=1000):  # aumentado a 1000 por defecto
    """Genera un dataset sintético pequeño para pruebas rápidas"""
    print("Generando dataset sintético de prueba...")
    
    # Aminoácidos comunes
    amino_acids = list('ACDEFGHIKLMNPQRSTVWY')
    
    sequences = []
    labels = []
    
    # Generar secuencias TF (con patrones simulados)
    for _ in range(n_samples // 2):
        # Secuencias más ricas en ciertos aminoácidos (simulando TF)
        seq_len = np.random.randint(50, 200)
        seq = ''.join(np.random.choice(list('CKRHGP'), seq_len))
        sequences.append(seq)
        labels.append(1)
    
    # Generar secuencias No-TF
    for _ in range(n_samples // 2):
        seq_len = np.random.randint(50, 200)
        seq = ''.join(np.random.choice(amino_acids, seq_len))
        sequences.append(seq)
        labels.append(0)
    
    # Crear DataFrame
    df = pd.DataFrame({
        'sequence': sequences,
        'label': labels
    })
    
    # Shuffle
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Guardar
    df.to_csv(output_path, index=False)
    print(f"✓ Dataset sintético creado: {output_path}")
    print(f"  Total: {len(df)} muestras (TF: {sum(labels)}, No-TF: {len(labels) - sum(labels)})")
    
    return output_path

def main():
    # Configuración para PRUEBA RÁPIDA
    BATCH_SIZE = 16
    EPOCHS = 10            # aumentado a 10 épocas
    LEARNING_RATE = 0.001
    MAX_LENGTH = 1000      # aumentado a 1000
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"🚀 MODO PRUEBA RÁPIDA - 1000 MUESTRAS 🚀")
    print(f"Usando dispositivo: {DEVICE}")
    
    # Cargar o generar dataset
    dataset_path = Path("data/protein_dataset.csv")
    
    if not dataset_path.exists():
        print("\n⚠️ Dataset no encontrado. Generando dataset sintético de prueba...")
        Path("data").mkdir(exist_ok=True)
        generate_dummy_dataset(dataset_path, n_samples=1000)  # generar 1000 muestras
    
    print(f"\nCargando dataset desde {dataset_path}...")
    dataset = ProteinDataset(dataset_path, max_length=MAX_LENGTH)
    
    if len(dataset) == 0:
        print("❌ Error: Dataset vacío")
        return
    
    # Split train/val/test
    train_size = int(0.7 * len(dataset))
    val_size = int(0.15 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    print(f"📊 Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
    
    # DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
    print("\n🧠 Usando modelo FINAL (ProteinClassifier)...")
    # Instanciar el modelo final. Ajusta los hiperparámetros si tu clase requiere otros nombres/valores.
    model = ProteinClassifier(
        vocab_size=21,
        embedding_dim=32,  # versión final
        hidden_dim=64,     # versión final
        dropout=0.5
    ).to(DEVICE)
    
    print(f"Parámetros del modelo: {sum(p.numel() for p in model.parameters()):,}")
    
    # Loss y optimizer
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Training loop
    best_val_loss = float('inf')
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    val_f1s, val_aucs = [], []
    epoch_history = []  # lista de dicts con métricas por época
    
    print(f"\n{'='*60}")
    print(f"🔥 INICIANDO ENTRENAMIENTO - {EPOCHS} ÉPOCAS")
    print(f"{'='*60}\n")
    
    for epoch in range(EPOCHS):
        print(f"📍 Epoch {epoch+1}/{EPOCHS}")
        
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, DEVICE)
        
        # Validate
        val_loss, val_acc, val_prec, val_rec, val_f1, val_auc = validate(
            model, val_loader, criterion, DEVICE
        )
        
        # Guardar historia
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        val_f1s.append(val_f1)
        val_aucs.append(val_auc)
        epoch_history.append({
            "epoch": epoch + 1,
            "train_loss": float(train_loss),
            "val_loss": float(val_loss),
            "train_acc": float(train_acc),
            "val_acc": float(val_acc),
            "val_precision": float(val_prec),
            "val_recall": float(val_rec),
            "val_f1": float(val_f1),
            "val_auc": float(val_auc)
        })
        
        print(f"   Train → Loss: {train_loss:.4f}, Acc: {train_acc:.4f}")
        print(f"   Val   → Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, F1: {val_f1:.4f}, AUC: {val_auc:.4f}")
        
        # Guardar mejor modelo
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_model.pth')
            print(f"   ✅ Mejor modelo guardado")
        print()
    
    # Test final
    print(f"\n{'='*60}")
    print("📊 EVALUACIÓN FINAL EN TEST SET")
    print(f"{'='*60}\n")
    
    model.load_state_dict(torch.load('best_model.pth'))
    test_loss, test_acc, test_prec, test_rec, test_f1, test_auc = validate(
        model, test_loader, criterion, DEVICE
    )
    
    print(f"Test Loss:      {test_loss:.4f}")
    print(f"Test Accuracy:  {test_acc:.4f}")
    print(f"Test Precision: {test_prec:.4f}")
    print(f"Test Recall:    {test_rec:.4f}")
    print(f"Test F1:        {test_f1:.4f}")
    print(f"Test AUC:       {test_auc:.4f}")
    
    # Construir metadata y guardarla
    metadata = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "device": str(DEVICE),
        "hyperparameters": {
            "batch_size": BATCH_SIZE,
            "epochs": EPOCHS,
            "learning_rate": LEARNING_RATE,
            "max_length": MAX_LENGTH,
            "model": {
                "type": model.__class__.__name__,
                "num_parameters": sum(p.numel() for p in model.parameters())
            }
        },
        "dataset": {
            "path": str(dataset_path),
            "total_samples": len(dataset),
            "split": {"train": len(train_dataset), "val": len(val_dataset), "test": len(test_dataset)},
            "label_counts": dataset.data['label'].value_counts().to_dict()
        },
        "training_history": epoch_history,
        "best_model_path": str(Path("best_model.pth").resolve()) if Path("best_model.pth").exists() else None,
        "test_metrics": {
            "loss": float(test_loss),
            "accuracy": float(test_acc),
            "precision": float(test_prec),
            "recall": float(test_rec),
            "f1": float(test_f1),
            "auc": float(test_auc)
        }
    }
    save_metadata("artifacts", metadata)
    
    # Plot
    plot_training_history(train_losses, val_losses, train_accs, val_accs)
    
if __name__ == "__main__":
    main()
