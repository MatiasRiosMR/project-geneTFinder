# GeneTFinder

Pequeña aplicación web para predecir si una secuencia proteica es un factor de transcripción (TF).

Qué hace
- Permite pegar una secuencia de aminoácidos y obtener una predicción rápida.
- Interfaz web con panel de administración y caché para respuestas repetidas.
- Usa un modelo entrenado (PyTorch o Keras) si está disponible; si no, aplica un heurístico sencillo.

Cómo usar (rápido)
1. Instala dependencias (si existe requirements.txt):
   - pip install -r requirements.txt
2. Ejecuta la app:
   - python web_app.py
3. Abre en tu navegador:
   - http://localhost:5000
   - Ir a /predictor para probar el predictor

Notas prácticas
- Coloca el checkpoint del modelo en: artifacts/best_model.pth (o define MODEL_PATH en el entorno).
- Usuarios por defecto para pruebas: admin/admin y estudiante/estudiante (configurable con variables de entorno GF_ADMIN_PASS / GF_USER_PASS).
- Endpoint de salud: /health
- Panel admin: /admin

Soporte
- Código fuente: revisa el repositorio para detalles de entrenamiento y opciones avanzadas.
- Si algo no funciona, revisa logs en la carpeta `logs/` y la carpeta `artifacts/` para el modelo.

Licencia
- Revisa el repositorio para información de licencia.
