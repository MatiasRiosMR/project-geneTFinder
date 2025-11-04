# GeneTFinder

Clasificador de proteínas que distingue Factores de Transcripción (TF) de No‑TF usando Deep Learning. Proyecto ligero con pipeline de datos, modelos (ligero y completo) y una app web para inferencia.

Nota: el modelo principal se entrenó en Google Colab debido a la escasez de hardware local (GPU).

Qué hace el modelo (resumen):
- Entrada: secuencia de aminoácidos → codificada a índices.
- Embedding: vectoriza aminoácidos para aprendizaje denso.
- CNN (filtros k=3,5,7): detecta motivos locales (features locales).
- BiLSTM: captura dependencias a largo plazo en ambas direcciones.
- Attention: pondera posiciones relevantes de la secuencia.
- Fully Connected + Softmax: produce probabilidad TF vs No‑TF.

Hay también una versión simple (BiLSTM ligero) para pruebas rápidas.

## Requisitos

- Python 3.10+ recomendado.
- Instalar dependencias:
```bash
pip install -r requirements.txt
```

## Prueba rápida (sin descargar datos)

Para validar que todo funciona sin descargar datos:
```bash
python train.py
# genera dataset sintético pequeño y entrena un modelo simple
python web_app.py
# abrir /login y luego /predictor
```
El script:
- Genera automáticamente un dataset sintético balanceado de 100 secuencias (50 TF / 50 No‑TF)
- Entrena la red SimpleProteinClassifier por 3 épocas
- Guarda el mejor checkpoint en best_model.pth

Luego levanta la web:
```bash
python web_app.py
```
- Abre /login, ingresa con el usuario configurado (ver “Usuarios y Seguridad” abajo).
- Ve a /predictor, pega una secuencia y presiona “Predecir”.
- El resultado aparece en una ventana emergente (modal) sin navegar a otra página.

## Dataset real (UniProt) y entrenamiento completo

1) Descargar y construir dataset balanceado:
```bash
python download_dataset.py
```
Crea data/protein_dataset.csv balanceado (TF/No‑TF).

2) Entrenar el modelo grande:
- Edita train.py y aumenta:
  - EPOCHS a 30–50
  - Usa ProteinClassifier en lugar de SimpleProteinClassifier
  - MAX_LENGTH a 1000 si tus secuencias son largas
- Ejecuta:
```bash
python train.py
```
Se guardará best_model.pth. La web lo prioriza automáticamente.

## App Web

Lanzar:
```bash
python web_app.py
```

Características:
- Predictor en /predictor (usa best_model.pth si está disponible)
- Modal de resultado (no redirige)
- Caché de predicciones con TTL para respuestas rápidas
- Panel admin en /admin y versión “Live” en /admin/live con refresco cada 1s
- Endpoints admin:
  - POST /admin/clear_cache
  - POST /admin/restart
  - POST /admin/toggle_role
  - POST /admin/force_disconnect
- Exportaciones:
  - /admin/export_csv (historial reciente)
  - /admin/export_data (JSON completo)

## API

- POST /api/predict
  - JSON: { "sequence": "MKTAYI..." }
  - Respuesta: { "label": "TF|No-TF", "class": 1|0, "confidence": float, "probs": [no_tf, tf] }

Nota: La API y el predictor web usan el mismo encoding que train.py (AA_TO_IDX) y max_length=300 por defecto.

## Usuarios y Seguridad

- Al iniciar, se genera users.json en /data o se migra el existente.
- Variables de entorno:
  - GF_SECRET: clave de sesión Flask
  - GF_ADMIN_USER: usuario admin (por defecto “admin”)
  - GF_ADMIN_PASS: contraseña admin (si no se define se genera aleatoria en disco)
  - GF_CREATE_DEFAULT_STUDENT=1 para crear usuario “estudiante”
  - GF_USER_USER, GF_USER_PASS: credenciales del usuario por defecto
  - MODEL_PATH: ruta alternativa del checkpoint del modelo (.pth)

Evita exponer credenciales por defecto en producción. Define GF_SECRET y GF_ADMIN_PASS.

## Estructura

```
geneTFinder/
├─ data/ (users.json, datasets)
├─ artifacts/ (checkpoints)
├─ download_dataset.py
├─ model.py
├─ train.py
├─ web_app.py
├─ predict.py
└─ README.md
```

## Modelos

- SimpleProteinClassifier: LSTM bidireccional ligero (ideal para pruebas)
- ProteinClassifier: CNN (k=3,5,7) + BiLSTM + Attention + FC (para entrenamiento completo)
- best_model.pth: checkpoint priorizado por la web
- La app detecta automáticamente si best_model.pth fue entrenado con el modelo simple o el grande.

## Notas de implementación

- Encoding centralizado en web_app.encode_for_model para mantener consistencia con train.py
- Admin “Live” con polling cada 1s a /admin/stats_json
- Caché TTL para predicciones (evita re‑cálculo de entradas repetidas)
- Fallback heurístico cuando no hay modelo cargado o hay errores de inferencia

## PWA (opcional)

Incluye site.webmanifest y favicon. Puedes servir estáticos desde /static y usar la cabecera cache-control ya configurada para recursos inmutables.

## Problemas comunes

- “ModuleNotFoundError: model”: ejecuta web_app.py desde la carpeta del proyecto.
- “best_model.pth no se carga”: verifica que exista en la raíz del proyecto o en /artifacts.
- Descarga UniProt 400: usa el endpoint /stream y keywords (ya corregido en download_dataset.py).

## Notas rápidas
- El encoding usado por la web y train.py debe coincidir (AA_TO_IDX, max_length por defecto 300).
- Si no hay modelo cargado la app usa heurísticos/feedback útil.
- Para entrenamiento serio usar Colab/GPU y aumentar épocas/longitud máxima.

Fin.
