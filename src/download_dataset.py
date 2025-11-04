import requests
import time
from pathlib import Path
import pandas as pd

def download_uniprot_sequences(query, filename, limit=5000):
    """
    Descarga secuencias desde UniProt API
    """
    print(f"Descargando {filename}...")
    
    url = "https://rest.uniprot.org/uniprotkb/stream"
    params = {
        'query': query,
        'format': 'fasta',
        'size': limit
    }
    
    try:
        response = requests.get(url, params=params, timeout=30)
        
        if response.status_code == 200 and response.text.strip():
            with open(filename, 'w') as f:
                f.write(response.text)
            print(f"✓ Descargado: {filename}")
            return True
        else:
            print(f"✗ Error descargando {filename}: {response.status_code}")
            if response.text:
                print(f"  Respuesta: {response.text[:200]}")
            return False
    except Exception as e:
        print(f"✗ Error de conexión: {e}")
        return False

def parse_fasta(filename):
    """
    Parsea archivo FASTA y retorna lista de secuencias
    """
    sequences = []
    with open(filename, 'r') as f:
        sequence = ""
        for line in f:
            line = line.strip()
            if line.startswith('>'):
                if sequence:
                    sequences.append(sequence)
                sequence = ""
            else:
                sequence += line
        if sequence:
            sequences.append(sequence)
    return sequences

def create_balanced_dataset():
    """
    Crea un dataset balanceado con TFs y No-TFs
    """
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    
    # Descargar factores de transcripción (TF)
    # Zinc Finger
    download_uniprot_sequences(
        query="(keyword:KW-0862) AND (reviewed:true) AND (length:[50 TO 1000])",
        filename=data_dir / "tf_zf.fasta",
        limit=2000
    )
    time.sleep(2)
    
    # bZIP
    download_uniprot_sequences(
        query="(keyword:KW-0963) AND (reviewed:true) AND (length:[50 TO 1000])",
        filename=data_dir / "tf_bzip.fasta",
        limit=1000
    )
    time.sleep(2)
    
    # Homeobox
    download_uniprot_sequences(
        query="(keyword:KW-0371) AND (reviewed:true) AND (length:[50 TO 1000])",
        filename=data_dir / "tf_homeobox.fasta",
        limit=1000
    )
    time.sleep(2)
    
    # DNA-binding (general TF)
    download_uniprot_sequences(
        query="(keyword:KW-0238) AND (reviewed:true) AND (length:[50 TO 1000])",
        filename=data_dir / "tf_dna_binding.fasta",
        limit=1500
    )
    time.sleep(2)
    
    # Descargar proteínas No-TF
    # Enzimas (Hydrolase)
    download_uniprot_sequences(
        query="(keyword:KW-0378) AND (reviewed:true) AND (length:[50 TO 1000]) NOT (keyword:KW-0238)",
        filename=data_dir / "non_tf_hydrolase.fasta",
        limit=2000
    )
    time.sleep(2)
    
    # Transferase
    download_uniprot_sequences(
        query="(keyword:KW-0808) AND (reviewed:true) AND (length:[50 TO 1000]) NOT (keyword:KW-0238)",
        filename=data_dir / "non_tf_transferase.fasta",
        limit=2000
    )
    time.sleep(2)
    
    # Transportadores
    download_uniprot_sequences(
        query="(keyword:KW-0813) AND (reviewed:true) AND (length:[50 TO 1000]) NOT (keyword:KW-0238)",
        filename=data_dir / "non_tf_transport.fasta",
        limit=1500
    )
    
    # Parsear y combinar
    print("\nProcesando secuencias...")
    
    tf_sequences = []
    non_tf_sequences = []
    
    # TFs
    for file in ["tf_zf.fasta", "tf_bzip.fasta", "tf_homeobox.fasta", "tf_dna_binding.fasta"]:
        filepath = data_dir / file
        if filepath.exists():
            seqs = parse_fasta(filepath)
            print(f"  {file}: {len(seqs)} secuencias")
            tf_sequences.extend(seqs)
    
    # No-TFs
    for file in ["non_tf_hydrolase.fasta", "non_tf_transferase.fasta", "non_tf_transport.fasta"]:
        filepath = data_dir / file
        if filepath.exists():
            seqs = parse_fasta(filepath)
            print(f"  {file}: {len(seqs)} secuencias")
            non_tf_sequences.extend(seqs)
    
    # Verificar que tenemos datos
    if not tf_sequences or not non_tf_sequences:
        print("\n✗ Error: No se pudieron descargar suficientes secuencias.")
        print(f"TF sequences: {len(tf_sequences)}")
        print(f"Non-TF sequences: {len(non_tf_sequences)}")
        return None
    
    # Balancear dataset
    min_size = min(len(tf_sequences), len(non_tf_sequences))
    print(f"\nSecuencias TF: {len(tf_sequences)}")
    print(f"Secuencias No-TF: {len(non_tf_sequences)}")
    print(f"Balanceando a: {min_size} muestras por clase")
    
    # Crear DataFrame balanceado
    df = pd.DataFrame({
        'sequence': tf_sequences[:min_size] + non_tf_sequences[:min_size],
        'label': [1] * min_size + [0] * min_size
    })
    
    # Shuffle
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Guardar
    output_file = data_dir / "protein_dataset.csv"
    df.to_csv(output_file, index=False)
    
    print(f"\n✓ Dataset creado: {output_file}")
    print(f"Total de muestras: {len(df)}")
    print(f"Distribución de clases:")
    print(df['label'].value_counts())
    
    return output_file

if __name__ == "__main__":
    print("=== Creando Dataset de Proteínas ===\n")
    create_balanced_dataset()
    print("\n=== Completado ===")
