import torch
from model import ProteinClassifier
from train import AA_TO_IDX
from pathlib import Path

# NUEVO: umbral de decisión para clasificar TF
THRESHOLD = 0.6

# Cambios: ajustar max_length por defecto a 300 y hacer predict/load más robustos

def encode_sequence(sequence, max_length=300):
    """Codifica una secuencia de proteína (max_length por defecto 300 para concordar con web_app)."""
    encoded = [AA_TO_IDX.get(aa, 0) for aa in (sequence or "").upper()]
    if len(encoded) > max_length:
        encoded = encoded[:max_length]
    else:
        encoded += [0] * (max_length - len(encoded))
    return torch.tensor(encoded, dtype=torch.long).unsqueeze(0)

def predict(model, sequence, device, max_length=300):
    """Realiza predicción para una secuencia.
    Retorna (label_str, prob_tf) donde prob_tf es probabilidad de clase TF (0..1).
    Maneja logits/vec/probabilidades y modelos que devuelven softmax/sigmoid.
    """
    model.eval()
    with torch.no_grad():
        encoded = encode_sequence(sequence, max_length=max_length).to(device)
        output = model(encoded)

        # normalizar a probabilidad TF
        prob_tf = None
        try:
            if isinstance(output, torch.Tensor):
                out = output.detach().cpu()
                # caso: [1, C] o [B, C]
                if out.dim() >= 2 and out.shape[-1] > 1:
                    probs = torch.softmax(out, dim=-1)
                    prob_tf = float(probs[0, -1]) if probs.shape[-1] > 1 else float(probs[0, 0])
                else:
                    # escalar o vector de tamaño 1 -> aplicar sigmoid
                    val = out.squeeze()
                    prob_tf = float(torch.sigmoid(val).item())
            else:
                # fallback si model devuelve scalar/np array
                import numpy as _np
                if isinstance(output, (list, tuple)):
                    # tratar (label, prob) o (prob,)
                    maybe = output[-1]
                    if isinstance(maybe, (float, int)):
                        prob_tf = float(maybe)
                    elif isinstance(maybe, (list, _np.ndarray)):
                        arr = _np.array(maybe, dtype=float)
                        prob_tf = float(arr[-1]) if arr.size > 1 else float(arr[0])
                elif isinstance(output, (float, int)):
                    prob_tf = float(output)
                elif hasattr(output, "probs"):
                    arr = _np.array(output.probs, dtype=float)
                    prob_tf = float(arr[-1]) if arr.size > 1 else float(arr[0])
        except Exception:
            prob_tf = None

        if prob_tf is None:
            # último recurso: heurístico simple
            prob_tf = 0.5

        prediction = "TF" if prob_tf > THRESHOLD else "No-TF"
        return prediction, float(prob_tf)

def _extract_state_for_load(state):
    """Extrae state_dict si torch.load devuelve un artefacto empaquetado."""
    if isinstance(state, dict):
        for k in ("state_dict", "model_state_dict", "model_state", "net", "params"):
            if k in state and isinstance(state[k], dict):
                return state[k]
        # heurística: si los values parecen tensores -> asumir es state_dict
        vals = list(state.values())[:6]
        if any(hasattr(v, "shape") for v in vals):
            return state
    return state

def load_model_for_predict(model_path='best_model.pth', device=None,
                           vocab_size=21, embedding_dim=32, hidden_dim=64, dropout=0.5):
    """Carga modelo compatible con ProteinClassifier; maneja state_dict empaquetados."""
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ProteinClassifier(
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        dropout=dropout
    ).to(device)
    p = Path(model_path)
    if not p.exists():
        raise FileNotFoundError(f"No se encontró el checkpoint: {model_path}")
    state = torch.load(str(p), map_location=device)
    sd = _extract_state_for_load(state)
    try:
        model.load_state_dict(sd)
    except Exception:
        # intentar limpiar prefijos 'module.' si existen
        clean = {}
        if isinstance(sd, dict):
            for k, v in sd.items():
                nk = k.replace("module.", "") if k.startswith("module.") else k
                clean[nk] = v
        try:
            model.load_state_dict(clean)
        except Exception as e:
            raise RuntimeError(f"No se pudo cargar state_dict en el modelo: {e}")
    model.to(device)
    model.eval()
    return model

def predict_from_fasta(fasta_file, model_path='best_model.pth'):
    """Predice para todas las secuencias en un archivo FASTA"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Cargar modelo
    model = load_model_for_predict(model_path=model_path, device=device)
    print(f"Modelo cargado desde {model_path}\n")
    
    # Leer FASTA
    sequences = []
    headers = []
    current_seq = ""
    current_header = ""
    
    with open(fasta_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('>'):
                if current_seq:
                    sequences.append(current_seq)
                    headers.append(current_header)
                current_header = line[1:]
                current_seq = ""
            else:
                current_seq += line
        
        if current_seq:
            sequences.append(current_seq)
            headers.append(current_header)
    
    # Predecir
    print(f"Predicciones para {len(sequences)} secuencias:\n")
    print("-" * 80)
    
    for header, sequence in zip(headers, sequences):
        prediction, probability = predict(model, sequence, device)
        print(f"Secuencia: {header[:60]}...")
        print(f"Predicción: {prediction} (Probabilidad: {probability:.4f})")
        print(f"Longitud: {len(sequence)} aminoácidos")
        print("-" * 80)

# NUEVA: demo que prueba una secuencia TF-like y una No-TF-like
def demo_predict_examples(model_path='best_model.pth', max_length=1000):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if not Path(model_path).exists():
        print(f"⚠️  {model_path} no existe. Ejecutá entrenamiento o poné el checkpoint en ese path.")
        return

    model = load_model_for_predict(model_path=model_path, device=device)
    # TF-like (patrón similar a generate_dummy_dataset)
    tf_seq = "CKRHGPCKRHGPCKRHGPCKRHGPCKRHGPCKRHGP"
    # No-TF-like (secuencia variada)
    notf_seq = "ACDEFGHIKLMNPQRSTVWYACDEFGHIKLMNPQRSTV"

    print("\n--- DEMO: Predicción de ejemplo ---")
    pred_tf, prob_tf = predict(model, tf_seq, device)
    pred_notf, prob_notf = predict(model, notf_seq, device)

    print(f"TF-like (len {len(tf_seq)}): Probabilidad={prob_tf:.4f}, Predicción={pred_tf}")
    print(f"NoTF-like (len {len(notf_seq)}): Probabilidad={prob_notf:.4f}, Predicción={pred_notf}")
    print("-----------------------------------\n")

def run_tests_on_data(model_path='best_model.pth'):
    """
    Busca en ../data/ archivos FASTA y ejecuta predict_from_fasta sobre
    un archivo TF-like y uno No-TF-like si existen.
    """
    base = Path(__file__).resolve().parent.parent
    data_dir = base / "data"
    if not data_dir.exists() or not data_dir.is_dir():
        print("No existe el directorio data/. Ejecutá demo_predict_examples o colocá archivos en data/")
        return

    fasta_exts = {'.fa', '.fasta', '.txt'}
    files = [p for p in data_dir.rglob("*") if p.is_file() and p.suffix.lower() in fasta_exts]

    tf_file = None
    non_tf_file = None

    # Intentar heurísticas por nombre de archivo
    for p in files:
        s = p.stem.lower()
        if 'tf' in s and not any(x in s for x in ('non','not','no','neg','notf')):
            tf_file = p
            break
    for p in files:
        s = p.stem.lower()
        if any(x in s for x in ('non','not','no','neg','notf','no_tf','non_tf','no-tf','non-tf')):
            non_tf_file = p
            break

    # Fallback: tomar cualquier archivo con 'tf' y otro cualquiera
    if tf_file is None:
        for p in files:
            if 'tf' in p.stem.lower():
                tf_file = p
                break
    if non_tf_file is None:
        for p in files:
            if p != tf_file:
                non_tf_file = p
                break

    if tf_file is None and non_tf_file is None:
        print("No se encontraron archivos FASTA en data/ para probar.")
        return

    # Verificar existencia del checkpoint
    model_path_p = Path(model_path)
    if not model_path_p.exists():
        print(f"No se encontró el checkpoint '{model_path}'. Colocá el .pth en la raíz o ajustá model_path.")
        return

    print("=== Ejecutando pruebas en data/ ===")
    if tf_file:
        print(f"\n[TF candidate] {tf_file}")
        try:
            predict_from_fasta(str(tf_file), model_path=model_path)
        except Exception as e:
            print(f"Error al predecir {tf_file}: {e}")
    if non_tf_file:
        print(f"\n[Non-TF candidate] {non_tf_file}")
        try:
            predict_from_fasta(str(non_tf_file), model_path=model_path)
        except Exception as e:
            print(f"Error al predecir {non_tf_file}: {e}")
    print("\n=== Fin de pruebas ===")

# Ajustar entrypoint para ejecutar pruebas automáticas si hay data/
if __name__ == "__main__":
    # Preferir pruebas con archivos en data/ si existen, sino la demo de ejemplos
    base = Path(__file__).resolve().parent.parent
    data_dir = base / "data"
    has_fasta = False
    if data_dir.exists() and data_dir.is_dir():
        fasta_exts = {'.fa', '.fasta', '.txt'}
        for p in data_dir.rglob("*"):
            if p.is_file() and p.suffix.lower() in fasta_exts:
                has_fasta = True
                break

    if has_fasta:
        run_tests_on_data(model_path='best_model.pth')
    else:
        demo_predict_examples(model_path='best_model.pth')
