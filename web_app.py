# NUEVO: instalador automático de dependencias (se ejecuta antes de cargar módulos pesados)
import os, sys
if True:
	# sólo ejecutar si se pidió explícitamente o si no se ha completado antes
	_auto_flag_env = os.environ.get("GF_AUTO_INSTALL", "0") == "1"
	_auto_flag_cli = "--install-deps" in sys.argv
	_already_done = os.environ.get("GF_AUTO_INSTALL_DONE", "0") == "1"
	if (_auto_flag_env or _auto_flag_cli) and not _already_done:
		try:
			import subprocess
			from pathlib import Path
			print("[AUTO-INSTALL] GF_AUTO_INSTALL activo -> instalando dependencias necesarias...", file=sys.stderr)
			# paquetes primarios que la app puede necesitar. El usuario puede ajustar GF_REQUIREMENTS
			default_pkgs = [
				"flask",
				"numpy",
				"torch",
				"pyyaml",
				"psutil",
				"werkzeug",
				"itsdangerous",
				"click",
				"requests"
				# NOTE: tensorflow/torchvision se pueden agregar mediante GF_REQUIREMENTS o requirements.txt
			]
			# permitir que el usuario añada paquetes extra vía GF_REQUIREMENTS (coma separada)
			extra = os.environ.get("GF_REQUIREMENTS", "")
			if extra:
				default_pkgs += [p.strip() for p in extra.split(",") if p.strip()]
			# si existe requirements.txt en el repo, instalar ese archivo en prioridad
			req_file = Path(__file__).parent / "requirements.txt"
			if req_file.exists():
				cmd = [sys.executable, "-m", "pip", "install", "-r", str(req_file)]
				print(f"[AUTO-INSTALL] instalando desde {req_file} ...", file=sys.stderr)
				subprocess.check_call(cmd)
			else:
				# instalar paquetes uno a uno para poder continuar si alguno falla
				for pkg in default_pkgs:
					try:
						print(f"[AUTO-INSTALL] instalando {pkg} ...", file=sys.stderr)
						subprocess.check_call([sys.executable, "-m", "pip", "install", pkg])
					except subprocess.CalledProcessError as e:
						print(f"[AUTO-INSTALL] fallo instalando {pkg}: {e}. Continúo con el siguiente.", file=sys.stderr)
			# marcar como completado para evitar bucles al re-exec
			os.environ["GF_AUTO_INSTALL_DONE"] = "1"
			# re-ejecutar el proceso actual para que los nuevos paquetes sean importables
			print("[AUTO-INSTALL] instalación finalizada. Reiniciando el proceso para aplicar cambios...", file=sys.stderr)
			os.execv(sys.executable, [sys.executable] + sys.argv)
		except Exception as e:
			# No abortar la app si la instalación falla; mostrar aviso y continuar.
			print(f"[AUTO-INSTALL] Error durante instalación automática: {e}", file=sys.stderr)

import os
import sys
import json
from pathlib import Path
from datetime import datetime, timedelta, timezone  # NUEVO: agregar timezone
from functools import wraps
import socket
import platform
import logging
# NUEVO: imports para caché/rehinicio
import time
import hashlib
import threading
# NUEVO: generador de contraseñas seguras
import secrets
import string
# Añadir importlib para cargar predict de forma flexible
import importlib

# Añadir el directorio raíz al path
root_dir = Path(__file__).parent
sys.path.append(str(root_dir))
# Añadir también el directorio `src/` para permitir importar `src/model.py` desde la app
sys.path.append(str(root_dir / "src"))

from flask import Flask, render_template, request, jsonify, session, redirect, url_for, flash, make_response
import torch
import numpy as np
import torch.nn as nn  # NUEVO: para detectar objetos modelo completos

# FIX: Importar desde model.py y no usar logger antes de crearlo
try:
    from model import SimpleProteinClassifier, ProteinClassifier
    MODELS_AVAILABLE = True
except ImportError:
    MODELS_AVAILABLE = False
    logger = logging.getLogger("GeneTFinder") if 'logging' in globals() else None
    if logger:
        logger.warning("WARN: No se pudo importar model.py - seguiremos intentando cargar checkpoints que contengan el modelo serializado")
    else:
        print("WARN: No se pudo importar model.py - seguiremos intentando cargar checkpoints que contengan el modelo serializado")

from werkzeug.security import generate_password_hash, check_password_hash

# Añadir import para plantilla inline
from flask import render_template_string, send_file

# Configuración básica de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("GeneTFinder")

# NUEVO: estructura de carpetas estándar
BASE_DIR = Path(__file__).parent.resolve()
DIRS = {
    "artifacts": BASE_DIR / "artifacts",
    "data": BASE_DIR / "data",
    "logs": BASE_DIR / "logs",
}
for d in DIRS.values():
    d.mkdir(parents=True, exist_ok=True)

app = Flask(__name__, static_folder="static", template_folder="templates")
app.secret_key = os.environ.get("GF_SECRET", "cambio_ante_produccion_poner_random")

# Parametrización de caché de estáticos y plantillas
app.config.update(
    SEND_FILE_MAX_AGE_DEFAULT=60 * 60 * 12,  # 12h para estáticos
    JSONIFY_PRETTYPRINT_REGULAR=False,
    TEMPLATES_AUTO_RELOAD=True
)

# SECCIÓN: configuración de cookies/ sesión (usar cookie de sesión por pestaña)
app.config.update({
    "SESSION_COOKIE_SAMESITE": "Lax",
    "SESSION_COOKIE_HTTPONLY": True,
    # opcional: en producción setear True y usar HTTPS
    "SESSION_COOKIE_SECURE": False,
})
# Duración de sesión persistente por defecto
app.permanent_session_lifetime = timedelta(days=int(os.environ.get("GF_SESSION_DAYS", "7")))

# Rutas de archivos (NUEVO: mover a /data si aplica)
# Model paths (intentar .pt primero, luego .h5)
MODEL_H5_CANDIDATES = [
    os.environ.get("MODEL_PATH", ""),
]


# Priorizar checkpoints PyTorch (.pth / .pt), incluyendo artifacts/best_model.pth y la raíz
MODEL_PT_CANDIDATES = [
    os.environ.get("MODEL_PATH", ""),                      # override por entorno
    str(DIRS["artifacts"] / "best_model.pth"),            # prioridad: artifacts
    str(BASE_DIR / "best_model.pth"),                     # fallback en raíz

]


# Rutas para metadata relacionada (ej. temperature) — evita NameError en load_model_backend
MODEL_META_PATHS = [
    str(DIRS["artifacts"] / "metadata.json"),
    str(DIRS["artifacts"] / "metadata.yaml"),
]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_BACKEND = None
model = None
MODEL_TEMPERATURE = 1.0

# Añadir import del umbral usado en src/predict.py (fallback si no existe)
try:
    from predict import THRESHOLD
except Exception:
    THRESHOLD = 0.6

# Intentar importar predict desde varias ubicaciones conocidas (raíz o src)
predict_module = None
predict_func = None
encode_sequence_func = None

for mod_name in ("predict", "src.predict"):
    try:
        predict_module = importlib.import_module(mod_name)
        # si define THRESHOLD, usarlo
        if hasattr(predict_module, "THRESHOLD"):
            try:
                THRESHOLD = float(getattr(predict_module, "THRESHOLD"))
            except Exception:
                pass
        # Importar funciones específicas
        if hasattr(predict_module, "predict"):
            predict_func = getattr(predict_module, "predict")
        if hasattr(predict_module, "encode_sequence"):
            encode_sequence_func = getattr(predict_module, "encode_sequence")
        if hasattr(predict_module, "load_model_for_predict"):
            load_model_for_predict = getattr(predict_module, "load_model_for_predict")
        
        logger.info(f"[INFO] predict module cargado: {mod_name}")
        break
    except Exception:
        predict_module = None

# Helper robusto para llamar a predict_module.predict con varias firmas posibles
def _use_predict_module(model_obj, seq: str, encoded_tensor, device):
    """
    Intenta llamar a predict_module.predict con diferentes firmas y normaliza la salida
    a un numpy.array [p_no_tf, p_tf]. Retorna None si no fue posible.
    """
    if predict_module is None or not hasattr(predict_module, "predict"):
        return None
    func = getattr(predict_module, "predict")
    # intentos con distintas firmas
    attempts = [
        (model_obj, seq, device),
        (model_obj, seq),                    # algunos implementan (model, seq)
        (seq,),                              # algunos implementan predict(seq)
        (model_obj, encoded_tensor, device), # algunos esperan tensor ya codificado
        (encoded_tensor,)                    # y otros solo tensor
    ]
    for args in attempts:
        try:
            res = func(*args)
            # Normalizar la respuesta
            # Caso tuple/list: (label, prob) o (prob,) o (probs_array,)
            if isinstance(res, (list, tuple)):
                if len(res) == 2 and isinstance(res[1], (float, int)):
                    prob_tf = float(res[1])
                    return np.array([1.0 - prob_tf, prob_tf])
                # si devuelve lista/array de probabilidades
                first = res[0]
                if isinstance(first, (list, np.ndarray)) and len(first) >= 2:
                    return np.array(first, dtype=float)
                # si devuelve (label, probs_array)
                if isinstance(res[1], (list, np.ndarray)):
                    arr = np.array(res[1], dtype=float)
                    if arr.ndim == 1 and arr.shape[0] >= 2:
                        return arr
            # Si devuelve un número -> probabilidad TF (binaria sigmoide)
            if isinstance(res, (float, int)):
                prob_tf = float(res)
                return np.array([1.0 - prob_tf, prob_tf])
            # Si devuelve array/np.ndarray directamente
            if isinstance(res, np.ndarray):
                if res.ndim == 1 and res.shape[0] >= 2:
                    return res.astype(float)
                # si es escalar dentro de ndarray
                if res.size == 1:
                    prob_tf = float(res.item())
                    return np.array([1.0 - prob_tf, prob_tf])
            # Si devuelve objeto con attribute 'prob' / 'probs'
            if hasattr(res, "prob"):
                prob_tf = float(getattr(res, "prob"))
                return np.array([1.0 - prob_tf, prob_tf])
            if hasattr(res, "probs"):
                arr = np.array(getattr(res, "probs"), dtype=float)
                if arr.ndim == 1 and arr.shape[0] >= 2:
                    return arr
            # no reconocido: intentar interpretar como mapping con 'tf' o 'prob'
            if isinstance(res, dict):
                if "prob" in res:
                    prob_tf = float(res["prob"])
                    return np.array([1.0 - prob_tf, prob_tf])
                if "probs" in res and isinstance(res["probs"], (list, np.ndarray)):
                    arr = np.array(res["probs"], dtype=float)
                    if arr.ndim == 1 and arr.shape[0] >= 2:
                        return arr
        except TypeError:
            # firma no compatible -> probar siguiente
            continue
        except Exception:
            # falló la ejecución -> devolver None (se usará fallback)
            logger.exception("predict_module.predict falló en ejecución")
            return None
    return None

# === NUEVO: helpers modulares para cargar/inspeccionar checkpoints ===
def _read_metadata():
    for meta in MODEL_META_PATHS:
        if os.path.exists(meta):
            try:
                with open(meta, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                try:
                    import yaml
                    with open(meta, "r", encoding="utf-8") as f:
                        return yaml.safe_load(f)
                except Exception:
                    pass
    return {}

def _try_jit_load(path):
    try:
        jm = torch.jit.load(path, map_location=DEVICE)
        if jm is not None:
            try: jm.to(DEVICE)
            except: pass
            try: jm.eval()
            except: pass
            return jm
    except Exception:
        return None

def _try_torch_load_module(path):
    try:
        obj = torch.load(path, map_location=DEVICE)
    except Exception:
        return None, None
    # 1) Si es un módulo/script con forward
    if not isinstance(obj, dict) and hasattr(obj, "forward") and callable(getattr(obj, "forward")):
        try:
            obj.to(DEVICE)
        except: pass
        try:
            obj.eval()
        except: pass
        return obj, "module"
    # 2) si es lista/tuple y primer elemento es módulo
    if isinstance(obj, (list, tuple)) and len(obj) > 0:
        first = obj[0]
        if hasattr(first, "forward") and callable(getattr(first, "forward")):
            try:
                first.to(DEVICE)
            except: pass
            try:
                first.eval()
            except: pass
            return first, "module_in_list"
    # 3) devolver dict (posible state_dict) para procesar
    return obj, "raw"

def _extract_state_dict(obj):
    if not isinstance(obj, dict):
        return None
    for k in ("state_dict","model_state_dict","model_state","net","params"):
        if k in obj and isinstance(obj[k], dict):
            return obj[k]
    # heurística: si valores parecen tensores/ndarray -> tratar dict como state_dict
    sample = list(obj.values())[:6]
    if any(hasattr(v, "shape") for v in sample):
        return obj
    return None

def _clean_state_dict(sd):
    # eliminar prefijo 'module.' si existe
    return { (k.replace("module.", "") if k.startswith("module.") else k): v for k,v in sd.items() }

def _try_load_with_known_classes(sd):
    # intenta instanciar clases definidas en model.py si están importadas
    if not MODELS_AVAILABLE:
        return None
    # probar ProteinClassifier (versión típica)
    try:
        m = ProteinClassifier(vocab_size=21, embedding_dim=32, hidden_dim=64, dropout=0.5)
        try:
            m.load_state_dict(sd)
        except Exception:
            m.load_state_dict(_clean_state_dict(sd))
        m.to(DEVICE); m.eval()
        return m
    except Exception:
        pass
    # probar SimpleProteinClassifier variantes
    if 'SimpleProteinClassifier' in globals():
        for emb,hid in ((32,64),(16,32)):
            try:
                m = SimpleProteinClassifier(vocab_size=21, embedding_dim=emb, hidden_dim=hid, dropout=0.3)
                try:
                    m.load_state_dict(sd)
                except Exception:
                    m.load_state_dict(_clean_state_dict(sd))
                m.to(DEVICE); m.eval()
                return m
            except Exception:
                continue
    return None

def _try_generic_model(sd, metadata):
    # intentar inferir vocab/emb/out desde metadata o desde state_dict
    vocab_size = int(metadata.get("vocab_size", 21)) if isinstance(metadata, dict) else 21
    emb_dim = int(metadata.get("embedding_dim", metadata.get("emb_dim", 32))) if isinstance(metadata, dict) else 32
    out_dim = int(metadata.get("out_dim", metadata.get("num_classes", 2))) if isinstance(metadata, dict) else 2

    # si state_dict contiene embedding.weight --> extraer shapes
    for k,v in sd.items():
        if "embed" in k.lower() and hasattr(v, "shape") and len(v.shape) == 2:
            vocab_size, emb_dim = int(v.shape[0]), int(v.shape[1])
            break

    # NUEVO: intentar inferir última capa si metadata no lo proporciona
    # buscar keys que contengan 'fc','classifier','out','linear' y tomar shape
    for k,v in sd.items():
        lk = k.lower()
        if any(x in lk for x in ("classifier","fc","out","linear","dense")) and hasattr(v, "shape") and len(v.shape) == 2:
            out_dim = int(v.shape[0])
            # si la dimensión entrante es igual a emb_dim -> asumimos fc directo
            # si es mayor, lo dejamos (GenericProteinModel usa emb_dim)
            break

    m = GenericProteinModel(vocab_size=vocab_size, emb_dim=emb_dim, hidden_dim=emb_dim, out_dim=out_dim)
    try:
        m.load_state_dict(_clean_state_dict(sd), strict=False)
        m.to(DEVICE); m.eval()
        return m
    except Exception:
        # último recurso: intentar asignar pesos parciales manualmente si existen claves compatibles
        try:
            clean = _clean_state_dict(sd)
            # asignar embedding si existe
            for k,v in list(clean.items()):
                if "embed" in k.lower() and hasattr(v, "shape"):
                    try:
                        m.embedding.weight.data[:v.shape[0], :v.shape[1]] = torch.tensor(v, device=DEVICE)
                    except Exception:
                        pass
                if any(x in k.lower() for x in ("fc.weight","classifier.weight","linear.weight")) and hasattr(v, "shape"):
                    try:
                        w = torch.tensor(v, device=DEVICE)
                        # adaptar tamaño si es compatible
                        if w.shape[1] == m.fc.weight.shape[1]:
                            m.fc.weight.data[:w.shape[0], :] = w[:m.fc.weight.shape[0], :]
                    except Exception:
                        pass
            m.to(DEVICE); m.eval()
            return m
        except Exception:
            return None


# --- Implementación ligera de GenericProteinModel y heurística de inferencia ---
class GenericProteinModel(torch.nn.Module):
    """
    Modelo genérico compacto: embedding + LSTM + fc. Usado como fallback si no
    se detecta una clase concreta en el checkpoint.
    """
    def __init__(self, vocab_size=21, emb_dim=32, hidden_dim=64, out_dim=2, dropout=0.3):
        super().__init__()
        self.embedding = torch.nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.lstm = torch.nn.LSTM(emb_dim, hidden_dim, num_layers=1, bidirectional=True, batch_first=True)
        self.dropout = torch.nn.Dropout(dropout)
        self.fc = torch.nn.Linear(hidden_dim * 2, out_dim)

    def forward(self, x):
        emb = self.embedding(x)
        out, _ = self.lstm(emb)
        # Pooling: mean over sequence
        pooled = out.mean(dim=1)
        h = self.dropout(pooled)
        logits = self.fc(h)
        if logits.shape[-1] == 1:
            return torch.sigmoid(logits).squeeze()
        return torch.softmax(logits, dim=-1)


def _infer_model_from_state_dict(sd, metadata):
    """Intenta inferir una arquitectura compatible con el state_dict.
    Estrategia:
    - Si hay claves de conv -> intentar instanciar `ProteinClassifier` con combinaciones razonables.
    - Si hay claves LSTM sin conv -> intentar `SimpleProteinClassifier` o `GenericProteinModel`.
    - Cargar con strict=False y devolver si es viable.
    """
    clean = _clean_state_dict(sd)
    keys = set(clean.keys())

    def has_prefix(pfxs):
        return any(any(k.startswith(p) for p in pfxs) for k in keys)

    # Detectar convs
    conv_like = has_prefix(["conv", "conv1", "conv2", "conv3"])
    lstm_like = has_prefix(["lstm", "lstm.weight_ih_l0", "lstm.weight_hh_l0"])
    embed_like = has_prefix(["embedding.weight", "embed.weight", "embedding.weight"])

    # Try known complex model (ProteinClassifier) if available
    if 'ProteinClassifier' in globals():
        # probar varias combinaciones razonables
        try_params = [
            (21, 128, 256, 0.3),
            (21, 64, 128, 0.3),
            (21, 32, 64, 0.5),
        ]
        for vocab, emb, hid, drop in try_params:
            try:
                m = ProteinClassifier(vocab_size=vocab, embedding_dim=emb, hidden_dim=hid, dropout=drop)
                m.load_state_dict(clean, strict=False)
                m.to(DEVICE); m.eval()
                return m
            except Exception:
                continue

    # Probar SimpleProteinClassifier
    if 'SimpleProteinClassifier' in globals() or lstm_like:
        try_params = [
            (21, 64, 128, 0.3),
            (21, 32, 64, 0.3),
            (21, 16, 32, 0.3),
        ]
        for vocab, emb, hid, drop in try_params:
            try:
                m = SimpleProteinClassifier(vocab_size=vocab, embedding_dim=emb, hidden_dim=hid, dropout=drop)
                m.load_state_dict(clean, strict=False)
                m.to(DEVICE); m.eval()
                return m
            except Exception:
                continue

    # Finalmente, usar GenericProteinModel como última opción
    try:
        vocab = int(metadata.get('vocab_size', 21)) if isinstance(metadata, dict) else 21
        emb_dim = int(metadata.get('embedding_dim', 32)) if isinstance(metadata, dict) else 32
        out_dim = int(metadata.get('out_dim', 2)) if isinstance(metadata, dict) else 2
        m = GenericProteinModel(vocab_size=vocab, emb_dim=emb_dim, hidden_dim=emb_dim, out_dim=out_dim)
        m.load_state_dict(clean, strict=False)
        m.to(DEVICE); m.eval()
        return m
    except Exception:
        return None

def inspect_checkpoint(path):
    """Resumen para diagnóstico (existencia, tamaño, tipo y primeras keys/shapes)."""
    info = {"path": str(path), "exists": os.path.exists(path)}
    if not info["exists"]:
        return info
    try:
        info["size_bytes"] = os.path.getsize(path)
    except Exception:
        pass
    try:
        obj = torch.load(path, map_location="cpu")
        info["loaded_type"] = type(obj).__name__
        if isinstance(obj, dict):
            keys = list(obj.keys())
            info["keys_sample"] = keys[:20]
            shapes = {}
            for k in keys[:30]:
                v = obj.get(k)
                try:
                    shapes[k] = tuple(v.shape) if hasattr(v, "shape") else str(type(v).__name__)
                except Exception:
                    shapes[k] = "?"
            info["shapes_sample"] = shapes
        elif isinstance(obj, (list,tuple)):
            info["len_list"] = len(obj)
            info["first_type"] = type(obj[0]).__name__ if len(obj)>0 else None
        else:
            info["has_forward"] = hasattr(obj, "forward") and callable(getattr(obj, "forward"))
    except Exception as e:
        info["load_error"] = repr(e)
    return info

# Reemplazar carga por función modular y enfocada en artifacts/best_model.pth
def load_model_backend():
    global MODEL_BACKEND, model, MODEL_TEMPERATURE
    MODEL_BACKEND, model, MODEL_TEMPERATURE = (None, None, 1.0)

    # leer metadata si existe
    metadata = _read_metadata()

    # iterar candidatos (prioriza artifacts/best_model.pth)
    for p in MODEL_PT_CANDIDATES:
        if not p:
            continue
        if not os.path.exists(p):
            logger.debug(f"[LOAD] candidato no existe: {p}")
            continue

        logger.info(f"[LOAD] probando checkpoint: {p}")

        # 1) scripted/traced
        jm = _try_jit_load(p)
        if jm is not None:
            MODEL_BACKEND = "pytorch"
            model = jm
            MODEL_TEMPERATURE = float(metadata.get("temperature", 1.0))
            logger.info(f"[INFO] Cargado ScriptModule desde {p}")
            return MODEL_BACKEND, model, MODEL_TEMPERATURE

        # 2) torch.load -> módulo o raw
        obj, kind = _try_torch_load_module(p)
        if obj is not None and kind in ("module","module_in_list"):
            MODEL_BACKEND = "pytorch"
            model = obj
            MODEL_TEMPERATURE = float(metadata.get("temperature", 1.0))
            logger.info(f"[INFO] Cargado módulo PyTorch desde {p} (kind={kind})")
            return MODEL_BACKEND, model, MODEL_TEMPERATURE

        # si torch.load devolvió 'raw' (dict u otro) -> intentar extraer state_dict
        try:
            raw = torch.load(p, map_location="cpu")
        except Exception as e:
            logger.warning(f"[WARN] torch.load fallo para {p}: {e}")
            continue

        sd = _extract_state_dict(raw)
        if sd is None:
            logger.info(f"[INFO] No se detectó state_dict en {p}. Inspeccion: {inspect_checkpoint(p)}")
            continue

        logger.info(f"[LOAD] encontrado state_dict con {len(sd)} keys en {p}")

        # 3) intentar clases conocidas
        m = _try_load_with_known_classes(sd)
        if m is not None:
            MODEL_BACKEND = "pytorch"; model = m
            MODEL_TEMPERATURE = float(metadata.get("temperature", 1.0))
            logger.info(f"[INFO] Cargado como clase conocida desde {p}")
            return MODEL_BACKEND, model, MODEL_TEMPERATURE

        # 4) intentar GenericProteinModel (usando metadata si existe)
        m = _try_generic_model(sd, metadata)
        if m is not None:
            MODEL_BACKEND = "pytorch"; model = m
            MODEL_TEMPERATURE = float(metadata.get("temperature", 1.0))
            logger.info(f"[INFO] Cargado GenericProteinModel desde {p}")
            return MODEL_BACKEND, model, MODEL_TEMPERATURE

        # 5) NUEVO: intentar inferir modelo desde state_dict (ConvProteinModel si hay convs)
        m = _infer_model_from_state_dict(sd, metadata)
        if m is not None:
            MODEL_BACKEND = "pytorch"; model = m
            MODEL_TEMPERATURE = float(metadata.get("temperature", 1.0))
            logger.info(f"[INFO] Cargado modelo inferido desde {p}")
            return MODEL_BACKEND, model, MODEL_TEMPERATURE

        # si llegamos aquí, no pudimos instanciar
        logger.warning(f"[WARN] No se pudo instanciar modelo desde state_dict en {p}. Inspección: {inspect_checkpoint(p)}")

    # intentar keras .h5 si existiera
    for h5 in MODEL_H5_CANDIDATES:
        if not h5 or not os.path.exists(h5):
            continue
        try:
            from tensorflow.keras.models import load_model
            km = load_model(h5, compile=False)
            MODEL_BACKEND = "keras"; model = km; MODEL_TEMPERATURE = float(metadata.get("temperature", 1.0))
            logger.info(f"[INFO] Cargado Keras desde {h5}")
            return MODEL_BACKEND, model, MODEL_TEMPERATURE
        except Exception as e:
            logger.warning(f"[WARN] No se pudo cargar Keras {h5}: {e}")

    logger.info("[INFO] No se cargó ningún modelo. Ver diagnostics vía /admin/reload_model")
    MODEL_BACKEND, model, MODEL_TEMPERATURE = (None, None, 1.0)
    return MODEL_BACKEND, model, MODEL_TEMPERATURE

def _load_model_temperature():
    t = 1.0
    for meta in MODEL_META_PATHS:
        if os.path.exists(meta):
            try:
                with open(meta, "r", encoding="utf-8") as f:
                    meta_d = json.load(f)
                t = float(meta_d.get("temperature", 1.0))
            except Exception:
                t = 1.0
            break
    return t

def _gen_password(length: int = 16) -> str:
    """Genera una contraseña segura aleatoria (no se registra en logs)."""
    alphabet = string.ascii_letters + string.digits + "!@#$%^&*()-_=+"
    return ''.join(secrets.choice(alphabet) for _ in range(length))

# ------------------ Usuarios ------------------
OLD_USERS_FILE = (BASE_DIR / "users.json")
OLD_PREDICTIONS_FILE = (BASE_DIR / "predictions.json")
USERS_FILE = DIRS["data"] / "users.json"
PREDICTIONS_FILE = DIRS["data"] / "predictions.json"
try:
    if OLD_USERS_FILE.exists() and not USERS_FILE.exists():
        os.replace(str(OLD_USERS_FILE), str(USERS_FILE))
    if OLD_PREDICTIONS_FILE.exists() and not PREDICTIONS_FILE.exists():
        os.replace(str(OLD_PREDICTIONS_FILE), str(PREDICTIONS_FILE))
except Exception:
    logger.warning("No se pudo migrar users.json/predictions.json a /data")

def save_users(users):
    with open(USERS_FILE, "w", encoding="utf-8") as f:
        json.dump(users, f, indent=2, ensure_ascii=False)

def init_users():
	admin_username = os.environ.get("GF_ADMIN_USER", "admin").strip() or "admin"
	admin_pass_env = os.environ.get("GF_ADMIN_PASS")  # si no está, se usará 'admin' por defecto para facilidad
	# Crear usuario estudiante por defecto para pruebas si no existe users.json
	student_username = os.environ.get("GF_USER_USER", "estudiante").strip() or "estudiante"
	student_pass_env = os.environ.get("GF_USER_PASS")  # si no está, se usará 'estudiante' por defecto

	# Si ya existe users.json, respetarlo
	if USERS_FILE.exists():
		try:
			with open(USERS_FILE, "r", encoding="utf-8") as f:
				data = json.load(f)
			if isinstance(data, dict):
				return data
		except Exception:
			pass

	# Si no hay password en entorno, usar credenciales simples para permitir login desde el formulario
	admin_password = admin_pass_env or "admin"
	student_password = student_pass_env or "estudiante"

	default = {
		admin_username: {
			"username": admin_username,
			"password_hash": generate_password_hash(admin_password),
			"role": "admin",
			"created_at": datetime.now().isoformat(),
			"connected": False,
			"device": None,
			"email": ""
		},
		student_username: {
			"username": student_username,
			"password_hash": generate_password_hash(student_password),
			"role": "user",
			"created_at": datetime.now().isoformat(),
			"connected": False,
			"device": None,
			"email": ""
		}
	}

	save_users(default)

	# Informar en logs (una sola vez) cómo ingresar — útil para pruebas locales.
	logger.info(
		f"[USERS] Usuarios inicializados: {', '.join(default.keys())}. "
		f"Credenciales por defecto -> {admin_username}/{admin_password} (admin), "
		f"{student_username}/{student_password} (user). "
		"En producción configure GF_ADMIN_PASS/GF_USER_PASS en el entorno."
	)

	return default

users = init_users()

# ------------------ Stats / Predictions ------------------
stats = {"total": 0, "tf": 0, "no_tf": 0, "history": []}

def load_predictions():
    try:
        with open(PREDICTIONS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}

def save_prediction(username, sequence, result, confidence):
    preds = load_predictions()
    if username not in preds:
        preds[username] = []
    preds[username].append({
        "timestamp": datetime.now().isoformat(),
        "sequence": sequence[:80] + ("..." if len(sequence) > 80 else ""),
        "result": result,
        "confidence": confidence
    })
    try:
        with open(PREDICTIONS_FILE, "w", encoding="utf-8") as f:
            json.dump(preds, f, indent=2, ensure_ascii=False)
    except Exception:
        logger.exception("No se pudo guardar predicción en archivo")

# ------------------ Capa de caché (NUEVO) ------------------
class TTLCache:
    def __init__(self, maxsize=1024, ttl=600):
        self.maxsize = maxsize
        self.ttl = ttl
        self._store = {}

    def _purge(self):
        now = time.time()
        keys = [k for k,(v,exp) in self._store.items() if exp < now]
        for k in keys:
            self._store.pop(k, None)
        # recorte por tamaño
        if len(self._store) > self.maxsize:
            # descartar entradas más antiguas (orden aproximado)
            for k in list(self._store.keys())[:len(self._store)-self.maxsize]:
                self._store.pop(k, None)

    def get(self, key):
        self._purge()
        v = self._store.get(key)
        if not v:
            return None
        value, exp = v
        if exp < time.time():
            self._store.pop(key, None)
            return None
        return value

    def set(self, key, value):
        self._purge()
        self._store[key] = (value, time.time() + self.ttl)

    def clear(self):
        self._store.clear()

prediction_cache = TTLCache(maxsize=2048, ttl=600)

def _cache_key(seq: str):
    base = f"{MODEL_BACKEND}|{MODEL_TEMPERATURE}|{(len(seq) if seq else 0)}|{(seq or '').upper()}"
    return hashlib.sha1(base.encode('utf-8')).hexdigest()

# NUEVO: helper para inspeccionar checkpoints y ayudar al diagnóstico
def inspect_checkpoint(path):
    """
    Intenta cargar el checkpoint en CPU y devuelve un resumen mínimo:
    - existencia, tamaño
    - tipo del objeto cargado por torch.load
    - si es dict: keys (primeras 20) y shapes de primeros tensores
    """
    info = {"path": str(path), "exists": os.path.exists(path)}
    try:
        if not info["exists"]:
            return info
        info["size_bytes"] = os.path.getsize(path)
        try:
            obj = torch.load(path, map_location="cpu")
            info["loaded_type"] = type(obj).__name__
            if isinstance(obj, dict):
                keys = list(obj.keys())
                info["keys_sample"] = keys[:20]
                # recolectar shapes para primeros tensores/ndarrays
                shapes = {}
                for k in keys[:30]:
                    v = obj.get(k)
                    try:
                        if hasattr(v, "shape"):
                            shapes[k] = tuple(v.shape)
                        else:
                            shapes[k] = str(type(v).__name__)
                    except Exception:
                        shapes[k] = "?";
                info["shapes_sample"] = shapes
            elif isinstance(obj, (list, tuple)):
                info["len_list"] = len(obj)
                try:
                    first = obj[0]
                    info["first_type"] = type(first).__name__
                    if hasattr(first, "shape"):
                        info["first_shape"] = tuple(first.shape)
                except Exception:
                    pass
            else:
                # objeto (ScriptModule u otro). indicar si tiene forward
                info["has_forward"] = hasattr(obj, "forward") and callable(getattr(obj, "forward"))
        except Exception as e_load:
            info["load_error"] = repr(e_load)
    except Exception as e:
        info["inspect_error"] = repr(e)
    return info

# ------------------ Headers de caché (NUEVO) ------------------
@app.after_request
def add_cache_headers(resp):
    path = request.path or ""
    if path.startswith("/static/") or any(path.endswith(ext) for ext in (".css",".js",".svg",".png",".jpg",".jpeg",".ico",".webp",".woff",".woff2",".ttf")):
        resp.headers["Cache-Control"] = "public, max-age=31536000, immutable"
    elif path.endswith("_json") or path.startswith("/api/") or path.startswith("/admin/"):
        resp.headers["Cache-Control"] = "no-store"
    return resp

# ------------------ Decoradores ----------------
def login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        # Si no está logueado -> para peticiones AJAX/JSON devolver 401 JSON en vez de redirect
        if not session.get("logged_in"):
            wants_json = (
                request.is_json
                or request.headers.get("X-Requested-With") == "XMLHttpRequest"
                or "application/json" in request.headers.get("Accept", "")
            )
            if wants_json:
                return jsonify({"error": "authentication_required", "login_url": url_for("login")}), 401
            flash("Debes iniciar sesión para acceder", "error")
            return redirect(url_for("login", next=request.url))

        # NOTA: quitamos el timeout de 30min que cerraba la sesión automáticamente.
        # Mantendremos last_activity solo como dato informativo, no como condición para destruir la sesión.
        session["last_activity"] = datetime.now().isoformat()
        # No hacemos session.permanent = True: mantenemos cookie de sesión que caduca al cerrar la pestaña.
        session.modified = True

        # Refrescar last_activity en memoria (sin escribir a disco cada request)
        u = session.get("username")
        if u and u in users:
            users[u]["last_activity"] = session["last_activity"]
        return f(*args, **kwargs)
    return decorated

def admin_required(f):
    @wraps(f)
    @login_required
    def decorated(*args, **kwargs):
        if session.get("role") != "admin":
            flash("Acceso restringido. Se requieren privilegios de administrador.", "error")
            # FIX: redirigir a home
            return redirect(url_for("home"))
        return f(*args, **kwargs)
    return decorated

# ------------------ Helper fallback ------------------
def fallback_predict(seq):
    """Heurística provisional marcada como fallback."""
    s = (seq or "").upper()
    L = max(1, len(s))
    basic = sum(s.count(c) for c in ("K","R","H")) / L
    len_score = min(1.0, L / 800.0)
    raw = 0.55 * basic + 0.45 * len_score
    score = max(0.02, min(0.98, raw))
    probs = [1.0 - score, score]
    label = "Factor de Transcripción (TF)" if score >= 0.5 else "No-TF"
    confidence = float(max(score, 1.0 - score))
    return label, probs, confidence

# Añadir validaciones biológicas MENOS ESTRICTAS
VALID_AMINO_ACIDS = set('ACDEFGHIKLMNPQRSTVWY')  # 20 aminoácidos estándar
MIN_SEQUENCE_LENGTH = 30   # REDUCIDO: mínimo más realista
MAX_SEQUENCE_LENGTH = 5000  # AUMENTADO: máximo más permisivo

def validate_protein_sequence(seq: str) -> tuple[bool, str]:
    """
    Valida que la secuencia sea una proteína válida (MENOS ESTRICTA).
    Retorna (es_valida, mensaje_error)
    """
    if not seq:
        return False, "Secuencia vacía"
    
    seq_upper = seq.upper().strip()
    
    # Verificar longitud (más permisivo)
    if len(seq_upper) < MIN_SEQUENCE_LENGTH:
        return False, f"Secuencia muy corta. Mínimo {MIN_SEQUENCE_LENGTH} aminoácidos (tienes {len(seq_upper)})"
    
    if len(seq_upper) > MAX_SEQUENCE_LENGTH:
        return False, f"Secuencia muy larga. Máximo {MAX_SEQUENCE_LENGTH} aminoácidos (tienes {len(seq_upper)})"
    
    # Verificar que solo contenga aminoácidos válidos
    invalid_chars = set(seq_upper) - VALID_AMINO_ACIDS
    if invalid_chars:
        return False, f"Caracteres inválidos: {', '.join(sorted(invalid_chars))}"
    
    return True, "Secuencia válida"

def calculate_sequence_quality_score(seq: str) -> float:
    """
    Calcula un score de calidad SIMPLE (0-1).
    """
    seq_upper = seq.upper().strip()
    
    # Solo diversidad básica
    diversity = len(set(seq_upper)) / 20.0
    
    return diversity  # Score simple basado solo en diversidad

# ...existing code...

# ------------------ Rutas auxiliares (NUEVO) ------------------
@app.route("/health")
def health():
    return jsonify({"status":"ok","time": datetime.now(timezone.utc).isoformat()}), 200

@app.route("/admin/clear_cache", methods=["POST"])
@admin_required
def clear_cache():
    prediction_cache.clear()
    return jsonify({"ok": True, "msg":"Cache limpiada"})

@app.route("/admin/restart", methods=["POST"])
@admin_required
def restart_server():
    """
    Reinicio seguro:
    - Si corre con Werkzeug, invoca shutdown.
    - Además toca este archivo para activar el reloader si está habilitado.
    """
    try:
        # Forzar reloader (tocar mtime del archivo)
        try:
            os.utime(__file__, None)
        except Exception:
            pass

        func = request.environ.get("werkzeug.server.shutdown")
        if func is None:
            return jsonify({"ok": False, "msg": "Restart no soportado en este servidor"}), 503

        # Ejecutar shutdown tras responder
        threading.Timer(0.25, func).start()
        return jsonify({"ok": True, "msg": "Servidor reiniciándose..."})
    except Exception as e:
        logger.exception("Fallo en restart")
        return jsonify({"ok": False, "msg": str(e)}), 500

# ------------------ Rutas ------------------
def _logo_path():
    """
    Retorna la ruta del logo si existe. Intenta varias ubicaciones comunes.
    """
    candidates = [
        BASE_DIR / "data" / "Logo.png",
        BASE_DIR / "data" / "logo.png",
        BASE_DIR / "static" / "logo.png",
        BASE_DIR / "static" / "Logo.png",
        BASE_DIR / "assets" / "logo.png",
        BASE_DIR / "Logo.png",
    ]
    for p in candidates:
        if p.exists():
            logger.info(f"[LOGO] Encontrado en: {p}")
            return p
    logger.error("[LOGO] ❌ No se encontró Logo.png en ninguna ubicación. Coloca el archivo en data/Logo.png")
    return None

@app.route("/logo.png")
def logo_png():
    """Sirve el logo del proyecto."""
    p = _logo_path()
    if p and p.exists():
        logger.info(f"[LOGO] ✅ Sirviendo logo desde: {p}")
        resp = make_response(send_file(str(p), mimetype="image/png"))
        resp.headers["Cache-Control"] = "public, max-age=86400"
        return resp
    logger.error("[LOGO] ❌ Archivo no encontrado")
    # Retornar imagen placeholder si no existe (evita error 404 visual)
    return make_response("Logo no encontrado. Coloca data/Logo.png en el proyecto.", 404)

@app.route("/favicon.ico")
def favicon():
    """Sirve el favicon (mismo logo)."""
    p = _logo_path()
    if p and p.exists():
        resp = make_response(send_file(str(p), mimetype="image/png"))
        resp.headers["Cache-Control"] = "public, max-age=86400"
        return resp
    return make_response("", 204)

@app.route("/")
def home():
    """Página de inicio con logo del proyecto."""
    model_info = {"backend": MODEL_BACKEND, "temperature": MODEL_TEMPERATURE}
    logo_url = "/logo.png"
    tpl = BASE_DIR / "templates" / "home.html"
    if tpl.exists():
        # Pasa logo_url a la plantilla existente
        return render_template("home.html", model_info=model_info, logo_url=logo_url)
    # Fallback inline si no hay plantilla
    return render_template_string("""
    <!doctype html>
    <html lang="es">
    <head>
      <meta charset="utf-8"/>
      <meta name="viewport" content="width=device-width,initial-scale=1"/>
      <title>Inicio</title>
      <link rel="icon" href="{{ logo }}" type="image/png"/>
      <style>
        body{margin:0;display:flex;min-height:100vh;align-items:center;justify-content:center;background:#0e1015;color:#e7e9ee;font-family:Inter,system-ui,sans-serif}
        .wrap{text-align:center;padding:24px}
        .logo{width:180px;height:auto;display:block;margin:0 auto 20px}
        .btn{display:inline-block;margin:8px 6px;padding:12px 16px;border-radius:10px;border:1px solid rgba(255,255,255,.15);color:#e7e9ee;text-decoration:none}
        .btn.primary{background:#7c5cff;border-color:#7c5cff}
      </style>
    </head>
    <body>
      <div class="wrap">
        <img class="logo" src="{{ logo }}" alt="Logo GeneTFinder"/>
        <div>
          <a class="btn primary" href="/predictor">Ir al Dashboard</a>
          <a class="btn" href="/about">Acerca</a>
          <a class="btn" href="/login">Login</a>
        </div>
      </div>
    </body>
    </html>
    """, logo=logo_url)

@app.route("/predictor")
@login_required
def predictor():
    """Página dedicada a predicciones."""
    model_info = {"backend": MODEL_BACKEND, "temperature": MODEL_TEMPERATURE}
    return render_template("predict.html", model_info=model_info)

@app.route("/login", methods=["GET","POST"])
def login():
    if request.method == "POST":
        username = request.form.get("username","").strip()
        password = request.form.get("password","")
        if username in users:
            u = users[username]
            if check_password_hash(u["password_hash"], password):
                session["logged_in"] = True
                session["username"] = username
                session["role"] = u.get("role", "user")
                # Mantener sesión persistente para no pedir login seguido
                session.permanent = True
                # Guardar marca de actividad (informativo)
                session["last_activity"] = datetime.now().isoformat()
                session.modified = True
                users[username]["connected"] = True
                users[username]["device"] = request.headers.get("User-Agent","")
                save_users(users)
                return redirect(url_for("predictor"))
        flash("Usuario o contraseña incorrectos", "error")
    return render_template("login.html")

@app.route("/logout")
def logout():
    u = session.get("username")
    if u and u in users:
        users[u]["connected"] = False
        save_users(users)
    session.clear()
    flash("Sesión cerrada correctamente.", "info")
    # FIX: redirigir a home
    return redirect(url_for("home"))

@app.route("/logout_silent", methods=["POST"])
def logout_silent():
    """
    Desconexión silenciosa al cerrar navegador/pestaña.
    No redirige, solo marca al usuario como desconectado.
    Llamado vía navigator.sendBeacon desde beforeunload.
    """
    try:
        u = session.get("username")
        if u and u in users:
            users[u]["connected"] = False
            users[u]["device"] = None
            save_users(users)
            logger.info(f"[LOGOUT] Usuario {u} desconectado por cierre de navegador")
        session.clear()
        return '', 204  # No Content
    except Exception as e:
        logger.exception("Error en logout_silent")
        return '', 204  # Responder siempre 204 para no interferir con cierre

@app.route("/admin")
@admin_required
def admin():
    """Panel de control del administrador."""
    try:
        model_info = {
            "backend": MODEL_BACKEND,
            "temperature": MODEL_TEMPERATURE,
            "device": str(DEVICE),
            "memory_usage": _get_memory_usage()
        }
        
        system_stats = {
            "users_connected": sum(1 for u in users.values() if u.get("connected")),
            "total_users": len(users),
            "total_predictions": stats["total"],
            "accuracy": _calculate_accuracy()
        }
        
        return render_template(
            "admin_stats.html",
            model_info=model_info,
            system_stats=system_stats,
            users=users,
            stats=stats
        )
    except Exception as e:
        logger.error(f"Error en panel admin: {str(e)}")
        return render_template("error.html", error="Error al cargar panel admin"), 500

def _get_memory_usage():
    """Obtiene uso de memoria del proceso."""
    try:
        import psutil
        process = psutil.Process()
        mem = process.memory_info().rss / 1024 / 1024  # MB
        return f"{mem:.1f} MB"
    except:
        return "N/A"

def _calculate_accuracy():
    """Calcula precisión promedio."""
    if not stats["history"]:
        return 0
    confidences = [h["confidence"] for h in stats["history"]]
    return sum(confidences) / len(confidences)

@app.route("/admin/stats_json")
@admin_required
def admin_stats_json():
    """API para datos en tiempo real del admin."""
    try:
        current_time = datetime.now().strftime("%H:%M:%S")
        
        # Actividad reciente
        recent_activity = []
        for h in stats["history"][:10]:
            confidence = h.get("confidence", 0) * 100
            recent_activity.append({
                "time": h["time"],
                "sequence": h["seq"][:30] + "...",
                "result": "TF" if h["class"] == 1 else "No-TF",
                "confidence": f"{confidence:.1f}%"
            })

        # Usuarios conectados
        active_users = []
        for username, user in users.items():
            if user.get("connected"):
                active_users.append({
                    "username": username,
                    "role": user.get("role", "user"),
                    "device": user.get("device", "Desconocido"),
                    "connected_since": user.get("last_activity", "N/A")
                })

        return jsonify({
            "timestamp": current_time,
            "model": {
                "status": "Activo" if MODEL_BACKEND else "No cargado",
                "backend": MODEL_BACKEND or "N/A",
                "memory": _get_memory_usage()
            },
            "stats": {
                "total_predictions": stats["total"],
                "tf_count": stats["tf"],
                "no_tf_count": stats["no_tf"],
                "accuracy": _calculate_accuracy() * 100
            },
            "active_users": active_users,
            "recent_activity": recent_activity
        })
    except Exception as e:
        logger.error(f"Error en stats_json: {str(e)}")
        return jsonify({"error": "Error interno"}), 500

@app.route("/register", methods=["GET", "POST"])
def register():
    """Ruta para el registro de nuevos usuarios."""
    if request.method == "POST":
        username = request.form.get("username")
        password = request.form.get("password")
        if not username or not password:
            flash("Usuario y contraseña requeridos", "error")
            return render_template("register.html")
        if username in users:
            flash("El usuario ya existe", "error")
            return render_template("register.html")
        users[username] = {
            "username": username,
            "password_hash": generate_password_hash(password),
            "role": "user",
            "created_at": datetime.now().isoformat()
        }
        save_users(users)
        flash("Usuario registrado correctamente. Ahora puedes ingresar.", "success")
        return redirect(url_for("login"))
    return render_template("register.html")

@app.route("/admin/users")
@admin_required
def admin_users():
    """Ruta para gestionar usuarios (solo admin)."""
    return render_template("admin_users.html", users=users)

@app.route("/admin/disconnect_user", methods=["POST"])
@admin_required
def disconnect_user():
    """Desconecta a un usuario específico."""
    try:
        data = request.get_json()
        username = data.get("username")
        
        if not username:
            return jsonify({"ok": False, "msg": "Username requerido"}), 400
        
        if username not in users:
            return jsonify({"ok": False, "msg": "Usuario no encontrado"}), 404
        
        # No permitir que el admin se desconecte a sí mismo
        if username == session.get("username"):
            return jsonify({"ok": False, "msg": "No puedes desconectarte a ti mismo"}), 400
        
        # Desconectar usuario
        users[username]["connected"] = False
        users[username]["device"] = None
        save_users(users)
        
        logger.info(f"[ADMIN] Usuario {username} desconectado por administrador {session.get('username')}")
        return jsonify({"ok": True, "msg": f"Usuario {username} desconectado"})
        
    except Exception as e:
        logger.exception("Error al desconectar usuario")
        return jsonify({"ok": False, "msg": str(e)}), 500

@app.route("/admin/delete_user", methods=["POST"])
@admin_required
def delete_user():
    """Elimina un usuario del sistema."""
    try:
        data = request.get_json()
        username = data.get("username")
        
        if not username:
            return jsonify({"ok": False, "msg": "Username requerido"}), 400
        
        if username not in users:
            return jsonify({"ok": False, "msg": "Usuario no encontrado"}), 404
        
        # No permitir que el admin se elimine a sí mismo
        if username == session.get("username"):
            return jsonify({"ok": False, "msg": "No puedes eliminarte a ti mismo"}), 400
        
        # Eliminar usuario
        del users[username]
        save_users(users)
        
        # Eliminar predicciones del usuario
        preds = load_predictions()
        if username in preds:
            del preds[username]
            try:
                with open(PREDICTIONS_FILE, "w", encoding="utf-8") as f:
                    json.dump(preds, f, indent=2, ensure_ascii=False)
            except Exception:
                pass
        
        logger.info(f"[ADMIN] Usuario {username} eliminado por administrador {session.get('username')}")
        return jsonify({"ok": True, "msg": f"Usuario {username} eliminado"})
        
    except Exception as e:
        logger.exception("Error al eliminar usuario")
        return jsonify({"ok": False, "msg": str(e)}), 500

@app.route("/admin/toggle_role", methods=["POST"])
@admin_required
def toggle_role():
    """Cambia el rol de un usuario entre admin y user."""
    try:
        data = request.get_json()
        username = data.get("username")
        
        if not username:
            return jsonify({"ok": False, "msg": "Username requerido"}), 400
        
        if username not in users:
            return jsonify({"ok": False, "msg": "Usuario no encontrado"}), 404
        
        # No permitir cambiar el rol del admin principal
        if username == session.get("username"):
            return jsonify({"ok": False, "msg": "No puedes cambiar tu propio rol"}), 400
        
        # Alternar rol
        current_role = users[username].get("role", "user")
        new_role = "admin" if current_role == "user" else "user"
        users[username]["role"] = new_role
        save_users(users)
        
        logger.info(f"[ADMIN] Rol de {username} cambiado a {new_role} por {session.get('username')}")
        return jsonify({"ok": True, "msg": f"Rol cambiado a {new_role}", "new_role": new_role})
        
    except Exception as e:
        logger.exception("Error al cambiar rol")
        return jsonify({"ok": False, "msg": str(e)}), 500

@app.route("/reload_model", methods=["POST"])
@admin_required
def reload_model():
    global MODEL_BACKEND, model, MODEL_TEMPERATURE
    MODEL_BACKEND, model, MODEL_TEMPERATURE = load_model_backend()
    if MODEL_BACKEND is None:
        # Devolver diagnóstico: inspeccionar candidatos para mostrar por qué falló
        diagnostics = []
        for p in MODEL_PT_CANDIDATES + MODEL_H5_CANDIDATES:
            if not p:
                continue
            diagnostics.append(inspect_checkpoint(p))
        return jsonify({"ok": False, "msg": "No se encontró modelo.", "diagnostics": diagnostics}), 404
    return jsonify({"ok": True, "backend": MODEL_BACKEND, "temperature": MODEL_TEMPERATURE})

# NUEVO: página inline para /predict (formulario con modal + fetch)
def _render_predict_form():
    return render_template_string("""
    <!doctype html>
    <html lang="es"><head><meta charset="utf-8"><title>GeneTFinder | Predictor</title>
    <link rel="icon" href="/logo.png" type="image/png"/>
    <style>
      body{font-family:system-ui,Arial;margin:20px;background:linear-gradient(180deg,#071028 0%, #0e1015 100%);color:#e7e9ee}
      textarea{width:100%;height:200px;border-radius:10px;padding:12px;border:1px solid rgba(255,255,255,.06);background:rgba(255,255,255,0.02);color:#e7e9ee;resize:vertical}
      .btn{padding:10px 14px;margin-top:10px;border-radius:10px;border:none;background:linear-gradient(90deg,#7c5cff,#21d4fd);color:#fff;cursor:pointer}
      .muted{color:#94a3b8}
      .row{margin:8px 0;color:#cbd5e1}
      .examples{display:flex;gap:8px;margin-bottom:10px}
      .ex-btn{padding:8px 10px;border-radius:8px;border:1px solid rgba(255,255,255,.06);background:transparent;color:#e7e9ee;cursor:pointer}
      .ex-btn.tf{background:linear-gradient(90deg,#2dbb6f,#7c5cff)}
      .ex-btn.notf{background:linear-gradient(90deg,#777,#444)}
      .warning{color:#ffb4b4;margin-top:8px}
      .card{background:#0f1720;border-radius:12px;padding:12px;border:1px solid rgba(255,255,255,.04);margin-top:12px}
      .label-tf{color:#7cfc9a;font-weight:700}
      .label-notf{color:#fca5a5;font-weight:700}
      .small{font-size:13px;color:#9ca3b0}
      .loader{width:16px;height:16px;border-radius:50%;border:3px solid rgba(255,255,255,.06);border-top-color:#7c5cff;animation:spin .9s linear infinite;display:inline-block;vertical-align:middle}
      @keyframes spin{to{transform:rotate(360deg)}}
    </style>
    </head><body>
    <h2>Predictor de Factores de Transcripción</h2>

    <div class="card">
      <div style="display:flex;align-items:center;justify-content:space-between;gap:12px">
        <div style="flex:1">
          <div class="examples" aria-hidden="false">
            <button type="button" id="exampleTF" class="ex-btn tf">Ejemplo TF</button>
            <button type="button" id="exampleNoTF" class="ex-btn notf">Ejemplo No-TF</button>
            <button type="button" id="clearBtn" class="ex-btn">Limpiar</button>
          </div>
          <form id="predictForm">
            <textarea name="sequence" id="sequence" placeholder="Pega tu secuencia de aminoácidos (20 AA estándar) ..."></textarea>
            <div style="display:flex;gap:8px;margin-top:8px">
              <button class="btn" type="submit">Predecir</button>
              <button type="button" id="btnQuick" class="btn" style="background:#444">Predecir rápido</button>
            </div>
          </form>
        </div>
        <div style="width:260px;margin-left:12px">
          <div class="small">Consejos</div>
          <ul class="small">
            <li>La secuencia debe usar los 20 aminoácidos estándar (A,C,D,...,Y).</li>
            <li>Longitud mínima: 50 aa. Si no cumple, se marcará como No-TF automáticamente.</li>
            <li>Ejemplos arriba para probar rápidamente.</li>
          </ul>
        </div>
      </div>
      <div id="formMsg" class="warning" style="display:none"></div>
    </div>

    <div class="card" id="resultCard" style="display:none;margin-top:18px">
      <h3>Resultado</h3>
      <div id="labelRow" class="row"></div>
      <div id="confRow" class="row"></div>
      <div id="qualityRow" class="row small"></div>
      <div id="cachedRow" class="row small muted"></div>
      <div id="detailsRow" class="row small"></div>
      <div style="margin-top:8px;text-align:right"><button id="reset" class="btn" type="button">Nueva</button></div>
    </div>

    <script>
      const seqInput = document.getElementById('sequence');
      const form = document.getElementById('predictForm');
      const resultCard = document.getElementById('resultCard');
      const labelRow = document.getElementById('labelRow');
      const confRow = document.getElementById('confRow');
      const qualityRow = document.getElementById('qualityRow');
      const cachedRow = document.getElementById('cachedRow');
      const detailsRow = document.getElementById('detailsRow');
      const formMsg = document.getElementById('formMsg');

      const exampleTF = document.getElementById('exampleTF');
      const exampleNoTF = document.getElementById('exampleNoTF');
      const clearBtn = document.getElementById('clearBtn');
      const resetBtn = document.getElementById('reset');
      const btnQuick = document.getElementById('btnQuick');

      const TF_EX = "MGRKKIQITRIMDERNRQVTFTKRKFGLMKKAYELSVLCDCEI";
      const NO_TF_EX = "AAAAATTTAAATTAAAGGGACACAGAACATAGACAGTACGTAGG";

      exampleTF.onclick = () => { seqInput.value = TF_EX; seqInput.focus(); };
      exampleNoTF.onclick = () => { seqInput.value = NO_TF_EX; seqInput.focus(); };
      clearBtn.onclick = () => { seqInput.value = ""; seqInput.focus(); };
      resetBtn.onclick = () => { resultCard.style.display='none'; seqInput.focus(); formMsg.style.display='none'; seqInput.value=''; };

      async function doPredict(seq){
        formMsg.style.display='none';
        try{
          const spinner = document.createElement('span'); spinner.className='loader';
          const body = {sequence: seq};
          btnQuick.disabled = true;
          const r = await fetch('/api/predict', {
            method:'POST', credentials: 'same-origin',
            headers:{'Content-Type':'application/json','Accept':'application/json'},
            body: JSON.stringify(body)
          });
          btnQuick.disabled = false;
          if (!r.ok){
            const txt = await r.text();
            formMsg.style.display='block'; formMsg.textContent = 'Error: '+(txt || r.status);
            return;
          }
          const data = await r.json();
          // Etiqueta y confianza
          const label = data.label || (data.class===1 ? 'TF' : 'No-TF');
          const conf = data.confidence != null ? Number(data.confidence).toFixed(4) : 'N/A';
          labelRow.innerHTML = "Etiqueta: <span class='"+(label.startsWith('TF')?'label-tf':'label-notf')+"'>"+label+"</span>";
          confRow.textContent = "Confianza: " + conf;
          qualityRow.textContent = data.quality_score != null ? ("Quality score: " + Number(data.quality_score).toFixed(3)) : "";
          cachedRow.textContent = data.cached ? "(resultado servido desde caché)" : "";
          detailsRow.innerHTML = "";
          if (data.validation_failed){
            detailsRow.innerHTML += "<div class='small warning'>Secuencia inválida: " + (data.validation_details || "") + " — marcada como No-TF</div>";
          }
          if (data.low_quality){
            detailsRow.innerHTML += "<div class='small warning'>Secuencia de baja calidad: " + (data.details || "") + "</div>";
          }
          if (data.fallback){
            detailsRow.innerHTML += "<div class='small muted'>Se usó heurístico fallback.</div>";
          }
          resultCard.style.display = 'block';
          window.scrollTo({ top: resultCard.offsetTop - 20, behavior: 'smooth' });
        }catch(e){
          btnQuick.disabled = false;
          formMsg.style.display='block';
          formMsg.textContent = 'Fallo de red o servidor.';
          console.error(e);
        }
      }

      form.addEventListener('submit', async (e) => {
        e.preventDefault();
        const seq = seqInput.value.trim();
        if (!seq){ formMsg.style.display='block'; formMsg.textContent='Secuencia vacía'; return; }
        await doPredict(seq);
      });

      btnQuick.addEventListener('click', async ()=> {
        const seq = seqInput.value.trim();
        if (!seq){ formMsg.style.display='block'; formMsg.textContent='Secuencia vacía'; return; }
        await doPredict(seq);
      });

      // atajo: ctrl+enter para enviar
      seqInput.addEventListener('keydown', (e)=>{ if(e.ctrlKey && e.key === 'Enter'){ e.preventDefault(); form.dispatchEvent(new Event('submit')); } });
    </script>
    </body></html>
    """)

# NUEVO: página inline para resultados si no hay templates/result.html
def _render_result_inline(seq, label=None, confidence=None, probs=None, error=None, fallback=False, cached=False):
    return render_template_string("""
    <!doctype html>
    <html lang="es"><head><meta charset="utf-8"><title>GeneTFinder | Resultado</title>
    <link rel="icon" href="/logo.png" type="image/png"/>
    <style>body{font-family:system-ui,Arial;margin:20px} pre{background:#f6f6f6;padding:10px} .muted{color:#666}</style>
    </head><body>
      <h2>Resultado de Predicción</h2>
      {% if error %}
        <p style="color:#b00020"><b>Error:</b> {{ error }}</p>
      {% else %}
        <p><b>Etiqueta:</b> {{ label }} {% if fallback %}<span class="muted">(fallback)</span>{% endif %} {% if cached %}<span class="muted">(cache)</span>{% endif %}</p>
        {% if confidence is not none %}<p><b>Confianza:</b> {{ '%.4f'|format(confidence) }}</p>{% endif %}
        {% if probs %}<p><b>Probabilidades [No-TF, TF]:</b> {{ probs }}</p>{% endif %}
      {% endif %}
      <h3>Secuencia</h3>
      <pre>{{ seq }}</pre>
      <p><a href="/predictor">Volver</a></p>
    </body></html>
    """, seq=seq, label=label, confidence=confidence, probs=probs, error=error, fallback=fallback, cached=cached)

@app.route("/predict", methods=["GET","POST"])
@login_required
def predict():
    if request.method == "GET":
        tpl = BASE_DIR / "templates" / "index.html"
        if tpl.exists():
            return render_template("index.html", stats=stats)
        return _render_predict_form()

    seq = request.form.get("sequence", "").strip()
    fasta_file = request.files.get("fasta_file")
    if fasta_file and fasta_file.filename:
        try:
            raw = fasta_file.read().decode("utf-8")
            if raw.startswith(">"):
                seq = "".join([line.strip() for line in raw.splitlines() if not line.startswith(">")])
            else:
                seq = raw.strip()
        except Exception:
            seq = ""
    if not seq:
        tpl_r = BASE_DIR / "templates" / "result.html"
        if tpl_r.exists():
            return render_template("result.html", error="Secuencia vacía", seq=seq, stats=stats)
        return _render_result_inline(seq, error="Secuencia vacía")

    # Caché
    try:
        key = _cache_key(seq)
        cached = prediction_cache.get(key)
        if cached:
            tpl_r = BASE_DIR / "templates" / "result.html"
            if tpl_r.exists():
                return render_template("result.html", seq=seq, label=cached["label"], confidence=cached["confidence"],
                                       probs=cached["probs"], stats=stats, fallback=cached.get("fallback", False), cached=True)
            return _render_result_inline(seq, label=cached["label"], confidence=cached["confidence"],
                                         probs=cached["probs"], fallback=cached.get("fallback", False), cached=True)
    except Exception:
        pass

    try:
        # CRÍTICO: Si no hay modelo, intentar cargar
        if MODEL_BACKEND is None or model is None:
            try_load = _try_load_via_predict_module()
            if not try_load:
                label, probs, confidence = fallback_predict(seq)
                prediction_cache.set(key, {"label": label, "probs": probs, "confidence": confidence, "fallback": True})
                stats["total"] += 1
                stats["history"].insert(0, {"time": datetime.now(timezone.utc).isoformat(), "seq": seq[:80], "class": int(probs[1]>=0.5), "confidence": confidence, "fallback": True})
                stats["history"] = stats["history"][:50]
                tpl_r = BASE_DIR / "templates" / "result.html"
                if tpl_r.exists():
                    return render_template("result.html", seq=seq, label=label, confidence=confidence, probs=probs, stats=stats, fallback=True)
                return _render_result_inline(seq, label=label, confidence=confidence, probs=probs, fallback=True)

        # NUEVO: Usar SOLO predict() de predict.py
        if predict_func is not None:
            logger.info("[PREDICT] Usando predict() de predict.py")
            prediction_label, prob_tf = predict_func(model, seq, DEVICE, max_length=300)
            probs = np.array([1.0 - prob_tf, prob_tf])
            prob_tf = float(probs[1])
            is_tf = prediction_label == "TF"
            pred = 1 if is_tf else 0
            confidence = float(prob_tf) if is_tf else float(1.0 - prob_tf)
            label = prediction_label
        else:
            logger.warning("[PREDICT] predict.py no disponible - usando fallback")
            label, probs, confidence = fallback_predict(seq)
            pred = 1 if probs[1] >= 0.5 else 0

        logger.info(f"[PREDICT] Resultado: {label} | Confianza: {confidence:.4f}")

        save_prediction(session.get('username'), seq, label, confidence)
        stats["total"] += 1
        if pred == 1:
            stats["tf"] += 1
        else:
            stats["no_tf"] += 1
        stats["history"].insert(0, {"time": datetime.now(timezone.utc).isoformat(), "seq": seq[:80], "class": pred, "confidence": confidence})
        stats["history"] = stats["history"][:50]

        prediction_cache.set(key, {"label": label, "probs": probs.tolist(), "confidence": confidence, "fallback": False})
        tpl_r = BASE_DIR / "templates" / "result.html"
        if tpl_r.exists():
            return render_template("result.html", seq=seq, label=label, confidence=confidence, probs=probs.tolist(), stats=stats, fallback=False)
        return _render_result_inline(seq, label=label, confidence=confidence, probs=probs.tolist(), fallback=False)

    except Exception:
        logger.exception("Error en /predict")
        label, probs, confidence = fallback_predict(seq)
        try:
            key = _cache_key(seq)
            probs_list = probs.tolist() if hasattr(probs, "tolist") else list(probs)
            prediction_cache.set(key, {"label": label, "probs": probs_list, "confidence": confidence, "fallback": True})
        except Exception:
            probs_list = [0.5, 0.5]
        try:
            stats["total"] += 1
            stats["history"].insert(0, {"time": datetime.now(timezone.utc).isoformat(), "seq": seq[:80], "class": int(probs_list[1] >= 0.5), "confidence": confidence, "fallback": True})
            stats["history"] = stats["history"][:50]
        except Exception:
            pass
        tpl_r = BASE_DIR / "templates" / "result.html"
        if tpl_r.exists():
            return render_template("result.html", seq=seq, label=label, confidence=confidence, probs=probs_list, stats=stats, fallback=True)
        return _render_result_inline(seq, label=label, confidence=confidence, probs=probs_list, fallback=True)

# Helper centralizado de encoding (evita duplicación en /predict y /api/predict)
def encode_for_model(seq: str, max_length: int = 300):
    """
    Codifica la secuencia usando encode_sequence de predict.py si está disponible,
    sino usa el fallback con AA_TO_IDX.
    Retorna un tensor Long [1, L] en DEVICE.
    """
    # NUEVO: Usar encode_sequence de predict.py si está disponible
    if encode_sequence_func is not None:
        try:
            return encode_sequence_func(seq, max_length=max_length).to(DEVICE)
        except Exception as e:
            logger.warning(f"Error usando encode_sequence de predict.py: {e}, usando fallback")
    
    # Fallback original
    try:
        from train import AA_TO_IDX
    except Exception:
        # Fallback mínimo si no se puede importar (X=0, 20 AA básicos)
        AA_TO_IDX = {
            'A': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6, 'H': 7, 'I': 8, 'K': 9, 'L': 10,
            'M': 11, 'N': 12, 'P': 13, 'Q': 14, 'R': 15, 'S': 16, 'T': 17, 'V': 18, 'W': 19, 'Y': 20
        }
    encoded_list = [AA_TO_IDX.get(aa, 0) for aa in (seq or "").upper()]
    if len(encoded_list) > max_length:
        encoded_list = encoded_list[:max_length]
    else:
        encoded_list += [0] * (max_length - len(encoded_list))
    return torch.tensor(encoded_list, dtype=torch.long).unsqueeze(0).to(DEVICE)

@app.route("/api/predict", methods=["POST"])
@login_required
def api_predict():
    global MODEL_BACKEND, model, MODEL_TEMPERATURE
    
    data = request.get_json(force=True)
    seq = data.get("sequence", "") or ""
    seq = seq.strip()
    if not seq:
        return jsonify({"error":"sequence required"}), 400

    # VALIDACIÓN MÍNIMA (solo caracteres y longitud básica)
    is_valid, error_msg = validate_protein_sequence(seq)
    if not is_valid:
        logger.warning(f"[PREDICT] Secuencia inválida: {error_msg}")
        return jsonify({
            "error": "Secuencia inválida",
            "details": error_msg,
            "label": "No-TF",
            "class": 0,
            "confidence": 0.95,
            "validation_failed": True
        }), 400

    # Calcular quality score simple
    quality_score = calculate_sequence_quality_score(seq)
    logger.info(f"[PREDICT] Quality score: {quality_score:.3f}")

    # Caché
    try:
        key = _cache_key(seq)
        cached = prediction_cache.get(key)
        if cached:
            return jsonify({
                "label": cached["label"],
                "class": 1 if cached["label"] == "TF" else 0,
                "confidence": cached["confidence"],
                "probs": cached["probs"],
                "quality_score": quality_score,
                "fallback": cached.get("fallback", False),
                "cached": True,
                "stats": stats
            })
    except Exception:
        pass

    try:
        # CRÍTICO: Verificar modelo
        if MODEL_BACKEND is None or model is None:
            logger.error("[PREDICT] NO HAY MODELO CARGADO")
            MODEL_BACKEND, model, MODEL_TEMPERATURE = load_model_backend()
            
            if MODEL_BACKEND is None or model is None:
                return jsonify({
                    "error": "Modelo no disponible",
                    "details": "Verifica que artifacts/best_model.pth existe",
                    "fallback_used": False
                }), 503

        logger.info(f"[PREDICT] Usando modelo: {MODEL_BACKEND} | Device: {DEVICE}")

        # USAR SOLO predict() de predict.py
        if predict_func is not None:
            logger.info("[PREDICT] Usando predict() de predict.py")
            prediction_label, prob_tf = predict_func(model, seq, DEVICE, max_length=300)
            
            # Normalizar salida
            probs = np.array([1.0 - prob_tf, prob_tf])
            probs = probs / probs.sum()
            
            prob_tf = float(probs[1])
            is_tf = prediction_label == "TF"
            pred = 1 if is_tf else 0
            confidence = float(prob_tf) if is_tf else float(1.0 - prob_tf)
            label = prediction_label
            
            logger.info(f"[PREDICT] predict.py -> Label: {label} | Prob TF: {prob_tf:.4f}")
            
        else:
            logger.warning("[PREDICT] predict.py no disponible - usando fallback")
            return jsonify({
                "error": "predict.py no disponible",
                "details": "No se pudo cargar el módulo de predicción",
                "fallback_used": True
            }), 500

        # Log resultado
        logger.info(f"[PREDICT] RESULTADO FINAL:")
        logger.info(f"  └─ Secuencia: {seq[:30]}...")
        logger.info(f"  └─ Prob TF: {prob_tf:.4f}")
        logger.info(f"  └─ Predicción: {label}")
        logger.info(f"  └─ Confianza: {confidence:.4f}")
        
        # Cachear y guardar
        try:
            key = _cache_key(seq)
            prediction_cache.set(key, {"label": label, "probs": probs.tolist(), "confidence": confidence, "fallback": False})
        except Exception:
            pass
            
        save_prediction(session.get("username"), seq, label, confidence)
        stats["total"] += 1
        if pred == 1:
            stats["tf"] += 1
        else:
            stats["no_tf"] += 1
        stats["history"].insert(0, {"time": datetime.now(timezone.utc).isoformat(), "seq": seq[:80], "class": pred, "confidence": confidence})
        stats["history"] = stats["history"][:50]
        
        return jsonify({
            "label": label, 
            "class": pred, 
            "confidence": confidence,
            "raw_probability": prob_tf,
            "quality_score": quality_score,
            "probs": probs.tolist(),
            "model_used": MODEL_BACKEND,
            "threshold": THRESHOLD,
            "fallback": False,
            "stats": stats
        })
    except Exception as e:
        logger.exception("Error en /api/predict")
        return jsonify({
            "error": "Error interno del servidor",
            "details": str(e),
            "fallback_used": False
        }), 500

@app.route("/admin/export_csv")
@admin_required
def export_csv():
    import csv
    from io import StringIO
    output = StringIO()
    writer = csv.writer(output)
    writer.writerow(["Fecha", "Secuencia", "Clase", "Confianza"])
    for h in stats["history"]:
        writer.writerow([
            h["time"],
            h["seq"],
            "TF" if h["class"] == 1 else "No-TF",
            "{:.4f}".format(h["confidence"])
        ])
    response = make_response(output.getvalue())
    response.headers["Content-Disposition"] = "attachment; filename=predicciones.csv"
    response.headers["Content-type"] = "text/csv"
    return response

@app.route("/admin/system_stats")
@admin_required
def system_stats():
    """API para estadísticas del sistema en tiempo real."""
    import psutil
    
    # Memoria
    memory = psutil.Process().memory_info()
    memory_usage = f"{memory.rss / 1024 / 1024:.1f} MB"
    
    # Usuarios activos y sus dispositivos
    active_users = {
        username: user for username, user in users.items() 
        if user.get("connected")
    }
    
    # Actividad reciente (últimas 10 acciones)
    recent = []
    for h in stats["history"][:10]:
        recent.append({
            "time": h["time"],
            "description": f"Predicción: {h['seq'][:30]}... ({h['confidence']*100:.1f}%)"
        })
    
    # Precisión del modelo
    confidences = [h["confidence"] for h in stats["history"]] if stats["history"] else []
    model_accuracy = sum(confidences)/len(confidences) if confidences else 0
    
    return jsonify({
        "active_users": len(active_users),
        "users": users,
        "memory_usage": memory_usage,
        "model_accuracy": model_accuracy,
        "recent_activity": recent
    })

@app.route("/admin/export_data")
@admin_required
def export_data():
    """Exporta todos los datos del sistema."""
    data = {
        "stats": stats,
        "users": users,
        "predictions": load_predictions(),
        "system_info": get_system_info()
    }
    
    response = make_response(json.dumps(data, indent=2))
    response.headers["Content-Disposition"] = "attachment; filename=genetfinder_data.json"
    response.headers["Content-Type"] = "application/json"
    return response

@app.route("/routes")
def list_routes():
    """Lista todas las rutas disponibles."""
    output = []
    for rule in app.url_map.iter_rules():
        methods = ','.join(rule.methods)
        output.append(f"{rule.endpoint}: {rule.rule} [{methods}]")
    return "<br>".join(output)

@app.route("/stats_json")
def stats_json():
    confidences = [h["confidence"] for h in stats["history"]] if stats["history"] else []
    avg_conf = sum(confidences)/len(confidences) if confidences else 0
    connected_users = sum(1 for u in users.values() if u.get("connected"))
    return jsonify({
        "total": stats["total"],
        "tf": stats["tf"],
               "no_tf": stats["no_tf"],
        "avg_confidence": avg_conf,
        "history": stats["history"],
        "connected_users": connected_users
    })

@app.route("/clear_stats", methods=["POST"])
def clear_stats():
    stats["total"] = 0
    stats["tf"] = 0
    stats["no_tf"] = 0
    stats["history"] = []
    try:
        if os.path.exists(PREDICTIONS_FILE):
            os.remove(PREDICTIONS_FILE)
    except Exception:
        pass
    return jsonify({"ok": True})

def get_system_info():
    try:
        hostname = socket.gethostname()
        try:
            local_ip = socket.gethostbyname(hostname)
        except Exception:
            local_ip = "127.0.0.1"
        system = platform.system()
        active_users = sum(1 for u in users.values() if u.get("connected"))
        return {
            "hostname": hostname,
            "ip": local_ip,
            "system": system,
            "active_users": active_users,
            "model_loaded": MODEL_BACKEND is not None,
            "model_type": MODEL_BACKEND,
            "predictions_total": stats["total"]
        }
    except Exception:
        return {"error": "No se pudo obtener información del sistema"}

# ------------------ Contenido detallado para "Acerca del proyecto" (NUEVO) ------------------
ABOUT_SECTIONS = {
    "title": "Acerca del Proyecto",
    "subtitle": "GeneTFinder: predicción de factores de transcripción con Deep Learning",
    "summary": (
        "GeneTFinder es una herramienta enfocada en identificar factores de transcripción (TF) "
        "a partir de secuencias proteicas. Integra un pipeline reproducible, inferencia eficiente, "
        "métricas transparentes y componentes listos para producción."
    ),
    "bullets": [
        "Modelo: arquitectura ligera (CNN pequeña) optimizada para inferencia en CPU/GPU.",
        "Heurístico fallback robusto y calibración por temperatura para probabilidades confiables.",
        "Entrada: codificación consistente y padding; inferencia batch y single‑sequence.",
        "Backend: PyTorch prioritario; fallback Keras y heurístico si no hay checkpoint.",
        "Web: sesiones persistentes, caché TTL para predicciones y endpoints JSON para integración.",
        "Admin: panel con uso de memoria, usuarios activos y actividad reciente.",
        "Despliegue: cabeceras de caché para estáticos y endpoint /health."
    ],
    "metrics_hint": "El backend expone /stats_json con métricas agregadas (total, TF/No‑TF, confianza promedio).",
    "roadmap": [
        "Soporte opcional de WebSocket/SSE para métricas en vivo.",
        "Mejoras de interpretabilidad (saliency/attributions por residuo).",
        "Entrenamiento semisupervisado con augmentations biológicamente plausibles.",
        "Exportación ONNX/TensorRT para inferencia acelerada."
    ]
}

def _render_about_page():
    """Renderiza 'Acerca' usando el logo como icono y cabecera."""
    repo_url = os.environ.get("REPO_URL", "https://github.com/MatiasRiosMR/GeneTFinder")
    html = """
    <!doctype html>
    <html lang="es" data-theme="dark">
    <head>
      <meta charset="utf-8"/>
      <meta name="viewport" content="width=device-width,initial-scale=1"/>
      <title>GeneTFinder | Acerca del Proyecto</title>
      <link rel="icon" href="/logo.png" type="image/png"/>
      <link rel="preconnect" href="https://fonts.googleapis.com">
      <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
      <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap" rel="stylesheet">
      <style>
        :root{
          --bg:#0a0b0f; --bg-alt:#0e1015; --text:#e7e9ee; --muted:#9ca3b0;
          --primary:#7c5cff; --primary-2:#21d4fd; --accent:#ff6b9d;
          --border:rgba(124,92,255,.15); --card:rgba(20,22,30,.8);
          --glow:rgba(124,92,255,.4); --radius:16px; --shadow:0 12px 40px rgba(0,0,0,.35);
        }
        *{box-sizing:border-box;margin:0;padding:0}
        body{background:var(--bg);color:var(--text);font-family:Inter,system-ui,sans-serif;overflow-x:hidden}
        
        body::before{
          content:'';position:fixed;inset:0;z-index:-1;
          background:
            radial-gradient(ellipse 1200px 800px at 10% 20%, rgba(124,92,255,.12), transparent),
            radial-gradient(ellipse 1200px 800px at 90% 60%, rgba(33,212,253,.08), transparent),
            radial-gradient(ellipse 800px 600px at 50% 80%, rgba(255,107,157,.06), transparent);
          animation:bgFloat 20s ease-in-out infinite alternate;
        }
        @keyframes bgFloat{
          0%{transform:translateY(0) scale(1)}
          50%{transform:translateY(-30px) scale(1.05)}
          100%{transform:translateY(0) scale(1)}
        }

        .nav{
          position:sticky;top:0;z-index:100;display:flex;align-items:center;justify-content:space-between;
          padding:14px 24px;background:rgba(10,11,15,.65);backdrop-filter:blur(20px) saturate(160%);
          border-bottom:1px solid var(--border);box-shadow:0 4px 20px rgba(0,0,0,.2);
        }
        .nav-links{display:flex;gap:8px}
        .nav a{
          color:var(--muted);text-decoration:none;padding:10px 14px;border-radius:10px;
          transition:all .3s cubic-bezier(.4,0,.2,1);font-weight:500;font-size:15px;
        }
        .nav a:hover{background:rgba(124,92,255,.12);color:var(--text);transform:translateY(-1px)}
        .nav a[aria-current="page"]{background:linear-gradient(135deg,var(--primary),var(--primary-2));color:#fff;box-shadow:0 4px 12px var(--glow)}

        .container{max-width:1200px;margin:0 auto;padding:48px 24px 32px}
        
        .hero{
          position:relative;padding:60px 32px 32px;margin-bottom:32px;border-radius:var(--radius);
          background:linear-gradient(135deg, rgba(124,92,255,.08), rgba(33,212,253,.05));
          border:1px solid var(--border);backdrop-filter:blur(10px);overflow:hidden;
          box-shadow:0 8px 32px rgba(0,0,0,.3), inset 0 1px 0 rgba(255,255,255,.05);
        }
        .hero::before{
          content:'';position:absolute;top:-50%;left:-50%;width:200%;height:200%;
          background:radial-gradient(circle, rgba(124,92,255,.15) 0%, transparent 70%);
          animation:pulse 8s ease-in-out infinite;pointer-events:none;
        }
        @keyframes pulse{
          0%, 100%{transform:scale(1) rotate(0deg);opacity:.6}
          50%{transform:scale(1.1) rotate(5deg);opacity:.8}
        }
        
        .hero-content{position:relative;z-index:1;text-align:center}
        .hero-logo{height:80px;width:auto;margin:0 auto 16px;display:block;animation:float 3s ease-in-out infinite}
        @keyframes float{
          0%,100%{transform:translateY(0)}
          50%{transform:translateY(-10px)}
        }
        .title{
          font-size:clamp(28px,5vw,40px);margin:8px 0 12px;font-weight:800;
          background:linear-gradient(120deg,var(--text),var(--primary-2));
          -webkit-background-clip:text;-webkit-text-fill-color:transparent;
          background-clip:text;line-height:1.1;
        }
        .subtitle{color:var(--muted);font-size:18px;line-height:1.6;margin:0 0 20px;max-width:880px;margin-left:auto;margin-right:auto}
        .cta{display:flex;justify-content:center;gap:12px}
        .btn{
          padding:13px 20px;border-radius:12px;border:1px solid var(--border);
          background:transparent;color:var(--text);text-decoration:none;font-weight:600;
          transition:all .3s ease;display:inline-flex;align-items:center;gap:8px;
        }
        .btn:hover{transform:translateY(-2px);box-shadow:0 8px 20px rgba(0,0,0,.25)}
        .btn.primary{
          background:linear-gradient(135deg,var(--primary),var(--primary-2));
          border:none;color:#fff;box-shadow:0 6px 18px var(--glow);
        }
        .btn.primary:hover{box-shadow:0 10px 28px var(--glow);transform:translateY(-3px) scale(1.02)}

        .row{display:grid;grid-template-columns:2fr 1fr;gap:24px;align-items:start;margin-top:32px}
        .card{
          background:var(--card);border:1px solid var(--border);border-radius:var(--radius);
          padding:24px;backdrop-filter:blur(16px);box-shadow:var(--shadow);
          transition:all .4s cubic-bezier(.4,0,.2,1);position:relative;overflow:hidden;
        }
        .card::before{
          content:'';position:absolute;inset:0;background:linear-gradient(135deg, rgba(124,92,255,.03), transparent);
          opacity:0;transition:opacity .4s;pointer-events:none;
        }
        .card:hover{transform:translateY(-4px);box-shadow:0 16px 50px rgba(0,0,0,.4);border-color:rgba(124,92,255,.3)}
        .card:hover::before{opacity:1}

        h3{
          font-size:22px;margin:0 0 14px;font-weight:700;color:var(--text);
          display:flex;align-items:center;gap:10px;
        }
        h3::before{content:'✨';font-size:20px}
        ul{margin:12px 0 8px 20px;color:var(--muted);line-height:1.8}
        ul li{margin:8px 0;position:relative;padding-left:8px}
        ul li::marker{color:var(--primary-2)}

        .footer{
          display:flex;justify-content:center;gap:18px;align-items:center;padding:24px;
          color:var(--muted);border-top:1px solid var(--border);background:var(--bg-alt);margin-top:48px;
        }
        .footer a{color:var(--muted);text-decoration:none;transition:color .3s}
        .footer a:hover{color:var(--primary-2)}

        @media (max-width:960px){
          .row{grid-template-columns:1fr}
          .title{font-size:clamp(28px,8vw,40px)}
        }

        @keyframes fadeInUp{
          from{opacity:0;transform:translateY(20px)}
          to{opacity:1;transform:translateY(0)}
        }
        .hero, .card{animation:fadeInUp .6s ease-out backwards}
        .card:nth-child(2){animation-delay:.1s}
        .card:nth-child(3){animation-delay:.2s}
      </style>
    </head>
    <body>
      <header class="nav">
        <div class="nav-links">
          <a href="/">🏠 Inicio</a>
          <a aria-current="page">📖 Acerca</a>
          <a href="/predictor">🧬 Dashboard</a>
        </div>
        <div class="cta">
          <a class="btn" href="/routes">Rutas</a>
        </div>
      </header>

      <main class="container">
        <section class="hero">
          <div class="hero-content">
            <img class="hero-logo" src="/logo.png" alt="Logo GeneTFinder"/>
            <h1 class="title">{{ s.title }}</h1>
            <p class="subtitle">{{ s.subtitle }}</p>
            <div class="cta">
              <a class="btn primary" href="/predictor">Ir al Dashboard</a>
              <a class="btn" href="/">Volver al inicio</a>
              <a class="btn" href="{{ repo }}" target="_blank">Ver código</a>
            </div>
          </div>
        </section>

        <section class="row">
          <div class="card">
            <h3>Características principales</h3>
            <ul>
              {% for b in s.bullets %}<li>{{ b }}</li>{% endfor %}
            </ul>
          </div>
          <div class="card">
            <h3>Roadmap de desarrollo</h3>
            <ul>
              {% for r in s.roadmap %}<li>{{ r }}</li>{% endfor %}
            </ul>
          </div>
        </section>
      </main>

      <footer class="footer">
        <span>© {{ year }} GeneTFinder</span> · 
        <a href="/health">Health</a> · 
        <a href="/admin">Admin</a> · 
        <a href="{{ repo }}" target="_blank">GitHub</a>
      </footer>
    </body>
    </html>
    """
    return render_template_string(
        html,
        s=ABOUT_SECTIONS,
        model={"backend": MODEL_BACKEND, "temperature": MODEL_TEMPERATURE},
        device=str(DEVICE),
        year=datetime.now().year,
        repo=repo_url
    )

# --- NUEVO: intentar cargar modelo usando src/predict.py si la carga modular falló ---
def _try_load_via_predict_module():
    """
    Si predict_module.define load_model_for_predict está disponible, intenta cargar
    el primer candidato existente y actualizar model/MODEL_BACKEND/MODEL_TEMPERATURE.
    Retorna True si se cargó un modelo usable.
    """
    global MODEL_BACKEND, model, MODEL_TEMPERATURE
    if predict_module is None:
        return False
    if not hasattr(predict_module, "load_model_for_predict"):
        return False

    # revisar candidatos (priorizar PT)
    for p in MODEL_PT_CANDIDATES + MODEL_H5_CANDIDATES:
        if not p or not os.path.exists(p):
            continue
        try:
            logger.info(f"[PREDICT-LOAD] intentado load_model_for_predict desde: {p}")
            m = predict_module.load_model_for_predict(model_path=p, device=DEVICE)
            if m is not None:
                model = m
                MODEL_BACKEND = "pytorch"
                MODEL_TEMPERATURE = _load_model_temperature()
                logger.info(f"[PREDICT-LOAD] cargado modelo vía predict.load_model_for_predict desde {p}")
                return True
        except Exception as e:
            logger.debug(f"[PREDICT-LOAD] fallo al cargar {p} vía predict.load_model_for_predict: {e}")
            continue
    return False

# ----------------------- Main -----------------------
if __name__ == "__main__":
    # CRÍTICO: Intentar cargar modelo ANTES de arrancar la app
    logger.info("\n" + "="*60)
    logger.info("🚀 INICIANDO GeneTFinder WebApp")
    logger.info("="*60)
    
    try:
        logger.info("\n📦 Cargando modelo entrenado...")
        MODEL_BACKEND, model, MODEL_TEMPERATURE = load_model_backend()
        
        if MODEL_BACKEND is None or model is None:
            logger.error("\n" + "❌"*30)
            logger.error("❌ CRÍTICO: NO SE PUDO CARGAR EL MODELO")
            logger.error("❌ La aplicación NO funcionará correctamente")
            logger.error("❌"*30)
            logger.error("\n📋 Diagnóstico de modelos:")
            
            diagnostics = []
            for p in MODEL_PT_CANDIDATES + MODEL_H5_CANDIDATES:
                if p:
                    diag = inspect_checkpoint(p)
                    diagnostics.append(diag)
                    logger.error(f"\n  Checkpoint: {p}")
                    logger.error(f"  └─ Existe: {diag.get('exists', False)}")
                    if diag.get('size_bytes'):
                        logger.error(f"  └─ Tamaño: {diag['size_bytes']/1024/1024:.2f} MB")
                    if diag.get('load_error'):
                        logger.error(f"  └─ Error: {diag['load_error']}")
            
            logger.error("\n💡 Soluciones:")
            logger.error("  1. Verifica que artifacts/best_model.pth existe")
            logger.error("  2. Entrena el modelo con: python train.py")
            logger.error("  3. Copia el modelo entrenado a artifacts/")
            logger.error("\n⚠️  LA APP USARÁ SOLO ERRORES (NO FALLBACK)")
            logger.error("="*60 + "\n")
        else:
            logger.info("\n" + "✓"*30)
            logger.info(f"✓ Modelo cargado exitosamente")
            logger.info(f"✓ Backend: {MODEL_BACKEND}")
            logger.info(f"✓ Device: {DEVICE}")
            logger.info(f"✓ Temperature: {MODEL_TEMPERATURE}")
            logger.info(f"✓ Threshold: {THRESHOLD}")
            logger.info("✓"*30 + "\n")
            
            # Mostrar arquitectura del modelo
            logger.info("📐 Arquitectura del modelo:")
            logger.info(f"{model}\n")
            
            # Test rápido con secuencia dummy
            try:
                test_seq = "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKVKALPDAQFEVVHSLAKWKRQTLGQHDFSAGEGLYTHMKALRPDEDRLSALEYYALNRSIYNLSHGYEYIKDTLK"
                test_tensor = encode_for_model(test_seq, max_length=300)
                with torch.no_grad():
                    test_output = model(test_tensor)
                logger.info(f"✓ Test de inferencia exitoso")
                logger.info(f"  └─ Output shape: {test_output.shape}")
                logger.info(f"  └─ Output type: {type(test_output)}")
                logger.info("="*60 + "\n")
            except Exception as e:
                logger.error(f"Test de inferencia falló: {e}\n")
                
    except Exception:
        logger.exception("Fallo crítico al intentar cargar el modelo en el arranque")

    info = get_system_info()
    logger.info(f" Host: {info.get('ip','127.0.0.1')}")
    logger.info(f" Sistema: {info.get('system','N/A')}")
    logger.info(f" Usuarios activos: {info.get('active_users',0)}")
    logger.info("="*60 + "\n")

    debug_mode = os.environ.get("FLASK_DEBUG", "0") == "1"
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT",5000)), debug=debug_mode, use_reloader=debug_mode)
