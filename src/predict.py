import torch
from model import ProteinClassifier
from train import AA_TO_IDX
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve, auc, confusion_matrix
import seaborn as sns

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

# NUEVA: demo que prueba MÚLTIPLES secuencias TF-like y No-TF-like
def demo_predict_examples(model_path='best_model.pth', max_length=1000, show_plot=True):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if not Path(model_path).exists():
        print(f"⚠️  {model_path} no existe. Ejecutá entrenamiento o poné el checkpoint en ese path.")
        return

    model = load_model_for_predict(model_path=model_path, device=device)
    
    # Secuencias de prueba reducidas: 6 TF + 6 No-TF
    test_sequences = [
        # TF-like - Patrones repetitivos característicos (6 secuencias)
        ("TF-01", "CKRHGPCKRHGPCKRHGPCKRHGPCKRHGPCKRHGP", "TF"),
        ("TF-02", "CKRHGP" * 10, "TF"),
        ("TF-03", "CKRHGPCKRHGPCKRHGPCKRHGPCKRHGPCKRHGPCKRHGP", "TF"),
        ("TF-04", "CKRHGP" * 8, "TF"),
        ("TF-05", "CKRHGPCKRHGPCKRHGP" * 3, "TF"),
        ("TF-06", "CKRHGP" * 12, "TF"),
        
        # No-TF-like - Secuencias variadas y diversas (6 secuencias)
        ("NoTF-01", "ACDEFGHIKLMNPQRSTVWYACDEFGHIKLMNPQRSTV", "No-TF"),
        ("NoTF-02", "MKVLWAALLVTFLAGCQAKVEQAVETEPEPELRQQTEWQSGQRWELALGRFWDYLRWVQT", "No-TF"),
        ("NoTF-03", "ADEFGHIKLMNPQRSTVWY" * 3, "No-TF"),
        ("NoTF-04", "MKTIIALSYIFCLVFA" * 2, "No-TF"),
        ("NoTF-05", "MKWVTFISLLLLFSSAYSRGVFRRDTHKSEIAHRFKDLGE", "No-TF"),
        ("NoTF-06", "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDG", "No-TF"),
    ]

    print("\n" + "=" * 80)
    print("PREDICCIONES MÚLTIPLES - VERIFICACIÓN DEL MODELO")
    print("=" * 80)
    
    results = []
    
    for name, sequence, expected in test_sequences:
        prediction, probability = predict(model, sequence, device)
        
        # Guardar resultado
        results.append({
            'name': name,
            'sequence': sequence,
            'expected': expected,
            'prediction': prediction,
            'probability': probability,
            'correct': prediction == expected
        })
        
        # Mostrar resultado individual
        status = "✓" if prediction == expected else "✗"
        print(f"\n{status} {name} (esperado: {expected})")
        print(f"  Secuencia: {sequence[:50]}{'...' if len(sequence) > 50 else ''}")
        print(f"  Longitud: {len(sequence)} aminoácidos")
        print(f"  Predicción: {prediction}")
        print(f"  Probabilidad TF: {probability:.4f}")
        print(f"  Confianza: {'Alta' if abs(probability - 0.5) > 0.3 else 'Media' if abs(probability - 0.5) > 0.1 else 'Baja'}")
        print("-" * 80)
    
    # Resumen estadístico
    print("\n" + "=" * 80)
    print("RESUMEN DE RESULTADOS")
    print("=" * 80)
    
    total = len(results)
    tf_count = sum(1 for r in results if r['prediction'] == 'TF')
    notf_count = total - tf_count
    correct = sum(1 for r in results if r['correct'])
    
    print(f"Total de secuencias analizadas: {total}")
    print(f"Predichas como TF: {tf_count}")
    print(f"Predichas como No-TF: {notf_count}")
    print(f"Predicciones correctas: {correct}/{total} ({100*correct/total:.1f}%)")
    
    # Promedios de probabilidad
    tf_probs = [r['probability'] for r in results if r['expected'] == 'TF']
    notf_probs = [r['probability'] for r in results if r['expected'] == 'No-TF']
    
    if tf_probs:
        print(f"\nProbabilidad promedio para TF reales: {sum(tf_probs)/len(tf_probs):.4f}")
    if notf_probs:
        print(f"Probabilidad promedio para No-TF reales: {sum(notf_probs)/len(notf_probs):.4f}")
    
    print("=" * 80 + "\n")
    
    # Generar gráficos
    if show_plot:
        plot_predictions(results)

def plot_predictions(results):
    """Genera gráficos visuales profesionales de las predicciones en archivos separados."""
    # Configurar estilo profesional
    plt.style.use('seaborn-v0_8-darkgrid')
    sns.set_palette("husl")
    
    # Colores profesionales
    COLOR_TF = '#2ecc71'      # Verde
    COLOR_NOTF = '#e74c3c'    # Rojo
    COLOR_CORRECT = '#3498db' # Azul
    COLOR_WRONG = '#e67e22'   # Naranja
    
    # Preparar datos
    y_true = [1 if r['expected'] == 'TF' else 0 for r in results]
    y_pred = [1 if r['prediction'] == 'TF' else 0 for r in results]
    y_scores = [r['probability'] for r in results]
    
    tp = sum(1 for r in results if r['expected'] == 'TF' and r['prediction'] == 'TF')
    tn = sum(1 for r in results if r['expected'] == 'No-TF' and r['prediction'] == 'No-TF')
    fp = sum(1 for r in results if r['expected'] == 'No-TF' and r['prediction'] == 'TF')
    fn = sum(1 for r in results if r['expected'] == 'TF' and r['prediction'] == 'No-TF')
    
    tf_probs = [r['probability'] for r in results if r['expected'] == 'TF']
    notf_probs = [r['probability'] for r in results if r['expected'] == 'No-TF']
    
    saved_files = []
    
    # ========== GRÁFICO 1: Probabilidades por secuencia ==========
    plt.figure(figsize=(12, 10))
    names = [r['name'] for r in results]
    probabilities = [r['probability'] for r in results]
    colors = [COLOR_CORRECT if r['correct'] else COLOR_WRONG for r in results]
    
    y_pos = np.arange(len(names))
    bars = plt.barh(y_pos, probabilities, color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
    plt.axvline(x=THRESHOLD, color='black', linestyle='--', linewidth=2.5, 
                label=f'Umbral de decisión ({THRESHOLD})', zorder=10)
    
    plt.yticks(y_pos, names, fontsize=9)
    plt.xlabel('Probabilidad de ser TF', fontsize=13, fontweight='bold')
    plt.ylabel('Secuencias', fontsize=13, fontweight='bold')
    plt.title('Probabilidades de Predicción por Secuencia', fontsize=16, fontweight='bold', pad=15)
    plt.xlim(0, 1.05)
    plt.legend(loc='lower right', fontsize=11)
    plt.grid(axis='x', alpha=0.4, linestyle='--')
    
    # Agregar valores en las barras
    for i, (bar, prob) in enumerate(zip(bars, probabilities)):
        plt.text(prob + 0.01, i, f'{prob:.3f}', va='center', fontsize=8, fontweight='bold')
    
    plt.tight_layout()
    filename1 = 'grafico_1_probabilidades_secuencias.png'
    plt.savefig(filename1, dpi=300, bbox_inches='tight', facecolor='white')
    saved_files.append(filename1)
    plt.close()
    
    # ========== GRÁFICO 2: Matriz de confusión ==========
    plt.figure(figsize=(10, 8))
    cm = confusion_matrix(y_true, y_pred)
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True, 
                square=True, linewidths=3, linecolor='black',
                xticklabels=['No-TF', 'TF'], yticklabels=['No-TF', 'TF'],
                annot_kws={'size': 20, 'weight': 'bold'})
    
    plt.xlabel('Predicción', fontsize=13, fontweight='bold')
    plt.ylabel('Valor Real', fontsize=13, fontweight='bold')
    plt.title('Matriz de Confusión', fontsize=16, fontweight='bold', pad=15)
    
    plt.tight_layout()
    filename2 = 'grafico_2_matriz_confusion.png'
    plt.savefig(filename2, dpi=300, bbox_inches='tight', facecolor='white')
    saved_files.append(filename2)
    plt.close()
    
    # ========== GRÁFICO 3: Curva ROC ==========
    plt.figure(figsize=(10, 8))
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    
    plt.plot(fpr, tpr, color='#2c3e50', linewidth=4, label=f'ROC (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], 'k--', linewidth=2.5, label='Clasificador aleatorio', alpha=0.6)
    plt.fill_between(fpr, tpr, alpha=0.3, color='#3498db')
    
    plt.xlabel('Tasa de Falsos Positivos (FPR)', fontsize=13, fontweight='bold')
    plt.ylabel('Tasa de Verdaderos Positivos (TPR)', fontsize=13, fontweight='bold')
    plt.title('Curva ROC (Receiver Operating Characteristic)', fontsize=16, fontweight='bold', pad=15)
    plt.legend(loc='lower right', fontsize=12)
    plt.grid(alpha=0.4, linestyle='--')
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    
    plt.tight_layout()
    filename3 = 'grafico_3_curva_roc.png'
    plt.savefig(filename3, dpi=300, bbox_inches='tight', facecolor='white')
    saved_files.append(filename3)
    plt.close()
    
    # ========== GRÁFICO 4: Distribución de probabilidades (Histograma) ==========
    plt.figure(figsize=(12, 8))
    
    bins = np.linspace(0, 1, 21)
    plt.hist(tf_probs, bins=bins, alpha=0.7, label=f'TF reales (n={len(tf_probs)})', 
             color=COLOR_TF, edgecolor='black', linewidth=1.5)
    plt.hist(notf_probs, bins=bins, alpha=0.7, label=f'No-TF reales (n={len(notf_probs)})', 
             color=COLOR_NOTF, edgecolor='black', linewidth=1.5)
    plt.axvline(x=THRESHOLD, color='black', linestyle='--', linewidth=3, 
                label=f'Umbral ({THRESHOLD})', zorder=10)
    
    plt.xlabel('Probabilidad de ser TF', fontsize=13, fontweight='bold')
    plt.ylabel('Frecuencia', fontsize=13, fontweight='bold')
    plt.title('Distribución de Probabilidades por Clase', fontsize=16, fontweight='bold', pad=15)
    plt.legend(loc='upper center', fontsize=12)
    plt.grid(alpha=0.4, linestyle='--')
    
    plt.tight_layout()
    filename4 = 'grafico_4_distribucion_probabilidades.png'
    plt.savefig(filename4, dpi=300, bbox_inches='tight', facecolor='white')
    saved_files.append(filename4)
    plt.close()
    
    # ========== GRÁFICO 5: Box plot de probabilidades ==========
    plt.figure(figsize=(10, 8))
    
    data_boxplot = [tf_probs, notf_probs]
    bp = plt.boxplot(data_boxplot, labels=['TF', 'No-TF'], patch_artist=True,
                     notch=True, showmeans=True, meanline=True,
                     boxprops=dict(linewidth=2.5, edgecolor='black'),
                     whiskerprops=dict(linewidth=2.5, color='black'),
                     capprops=dict(linewidth=2.5, color='black'),
                     medianprops=dict(linewidth=3, color='red'),
                     meanprops=dict(linewidth=3, color='blue', linestyle='--'))
    
    bp['boxes'][0].set_facecolor(COLOR_TF)
    bp['boxes'][1].set_facecolor(COLOR_NOTF)
    
    plt.axhline(y=THRESHOLD, color='black', linestyle='--', linewidth=2.5, 
                label=f'Umbral ({THRESHOLD})', alpha=0.7)
    plt.ylabel('Probabilidad de ser TF', fontsize=13, fontweight='bold')
    plt.xlabel('Clase Real', fontsize=13, fontweight='bold')
    plt.title('Distribución Estadística de Probabilidades (Box Plot)', fontsize=16, fontweight='bold', pad=15)
    plt.legend(fontsize=11)
    plt.grid(axis='y', alpha=0.4, linestyle='--')
    plt.ylim([-0.05, 1.05])
    
    plt.tight_layout()
    filename5 = 'grafico_5_boxplot_probabilidades.png'
    plt.savefig(filename5, dpi=300, bbox_inches='tight', facecolor='white')
    saved_files.append(filename5)
    plt.close()
    
    # ========== GRÁFICO 6: Panel de métricas ==========
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111)
    ax.axis('off')
    
    total = len(results)
    correct = sum(1 for r in results if r['correct'])
    accuracy = 100 * correct / total if total > 0 else 0
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    avg_tf_prob = sum(tf_probs) / len(tf_probs) if tf_probs else 0
    avg_notf_prob = sum(notf_probs) / len(notf_probs) if notf_probs else 0
    
    metrics_text = f"""
╔═════════════════════════════════════════════════════════╗
║     MÉTRICAS DE RENDIMIENTO DEL MODELO DE PREDICCIÓN   ║
║              DE FACTORES DE TRANSCRIPCIÓN               ║
╠═════════════════════════════════════════════════════════╣
║                                                         ║
║  📊 MÉTRICAS PRINCIPALES DE CLASIFICACIÓN               ║
║  ─────────────────────────────────────────────────      ║
║                                                         ║
║  Exactitud (Accuracy):           {accuracy:6.2f}%             ║
║  Precisión (Precision):          {precision:6.4f}             ║
║  Sensibilidad (Recall/TPR):      {recall:6.4f}             ║
║  Especificidad (TNR):            {specificity:6.4f}             ║
║  F1-Score:                       {f1:6.4f}             ║
║  AUC-ROC:                        {roc_auc:6.4f}             ║
║                                                         ║
║  📈 MATRIZ DE CONFUSIÓN DETALLADA                       ║
║  ─────────────────────────────────────────────────      ║
║                                                         ║
║  Verdaderos Positivos (TP):      {tp:6d}                ║
║  Verdaderos Negativos (TN):      {tn:6d}                ║
║  Falsos Positivos (FP):          {fp:6d}                ║
║  Falsos Negativos (FN):          {fn:6d}                ║
║                                                         ║
║  🎯 ESTADÍSTICAS GENERALES DE PREDICCIÓN                ║
║  ─────────────────────────────────────────────────      ║
║                                                         ║
║  Total de secuencias analizadas: {total:6d}                ║
║  Predicciones correctas:         {correct:6d}                ║
║  Predicciones incorrectas:       {total-correct:6d}                ║
║  Tasa de error:                  {100*(total-correct)/total:6.2f}%             ║
║                                                         ║
║  📊 ANÁLISIS DE PROBABILIDADES                          ║
║  ─────────────────────────────────────────────────      ║
║                                                         ║
║  Probabilidad promedio TF:       {avg_tf_prob:6.4f}             ║
║  Probabilidad promedio No-TF:    {avg_notf_prob:6.4f}             ║
║  Separación entre clases:        {abs(avg_tf_prob-avg_notf_prob):6.4f}             ║
║  Umbral de decisión:             {THRESHOLD:6.4f}             ║
║                                                         ║
║  ✅ CALIDAD DE SEPARACIÓN                               ║
║  ─────────────────────────────────────────────────      ║
║                                                         ║
║  Separación: {'EXCELENTE' if abs(avg_tf_prob-avg_notf_prob) > 0.5 else 'BUENA' if abs(avg_tf_prob-avg_notf_prob) > 0.3 else 'MODERADA':^40} ║
║                                                         ║
╚═════════════════════════════════════════════════════════╝
"""
    
    ax.text(0.5, 0.5, metrics_text, fontsize=11, family='monospace',
            verticalalignment='center', horizontalalignment='center',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5, pad=1.5))
    
    fig.suptitle('Resumen Completo de Métricas del Modelo', fontsize=18, fontweight='bold', y=0.98)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    filename6 = 'grafico_6_metricas_resumen.png'
    plt.savefig(filename6, dpi=300, bbox_inches='tight', facecolor='white')
    saved_files.append(filename6)
    plt.close()
    
    # Resumen de archivos guardados
    print("\n" + "=" * 70)
    print("📊 GRÁFICOS GENERADOS EXITOSAMENTE")
    print("=" * 70)
    for i, filename in enumerate(saved_files, 1):
        print(f"  {i}. {filename}")
    print("=" * 70)
    print(f"\n✅ Total de gráficos generados: {len(saved_files)}")
    print("📁 Todos los archivos están en alta resolución (300 DPI)\n")

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
        run_tests_on_data(model_path='./best_model.pth')
    else:
        demo_predict_examples(model_path='./best_model.pth')
