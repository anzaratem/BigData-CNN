# 🏫 Inventario Automático del Salón de Cómputo

Sistema de detección de objetos utilizando YOLOv8 y TensorFlow Lite para identificar y contar automáticamente los elementos del salón de cómputo.

## 📋 Descripción del Proyecto

Este proyecto implementa un sistema de visión por computadora que detecta y cuenta automáticamente los siguientes objetos:

| Código | Objeto |
|--------|--------|
| 0 | CPU |
| 1 | Mesa |
| 2 | Mouse |
| 3 | Pantalla |
| 4 | Silla |
| 5 | Teclado |

## 🎯 Características

- ✅ Detección de múltiples objetos en tiempo real
- ✅ Ejecución 100% local (sin conexión a internet requerida)
- ✅ Modelo optimizado en TensorFlow Lite
- ✅ Interface web intuitiva y responsiva
- ✅ Visualización con bounding boxes en color azul
- ✅ Tabla de inventario automática

## 🏗️ Arquitectura del Modelo

### Modelo Base
- **Arquitectura**: YOLOv8n (nano)
- **Framework**: Ultralytics YOLOv8
- **Dataset**: 50 imágenes del salón anotadas manualmente en Roboflow
- **Augmentación**: Flip horizontal, rotación 90°, brightness ±15%, blur

### Entrenamiento
- **Épocas**: 100 (con early stopping)
- **Batch size**: 16
- **Optimizador**: AdamW
- **Learning rate**: 0.01 → 0.01 (cosine)
- **Image size**: 640x640
- **Hardware**: Google Colab con GPU T4

### Métricas del Modelo
- **mAP50**: 78.2% ✅
- **mAP50-95**: 65.4%
- **Precision**: 81.5%
- **Recall**: 76.5%

### Conversión a TFLite
```python
from ultralytics import YOLO

# Cargar modelo entrenado
model = YOLO('best.pt')

# Exportar a TFLite con optimizaciones
model.export(
    format='tflite',
    imgsz=640,
    int8=False,  # Usar float16 para mejor balance
    half=True    # Precisión float16
)
```

## 📦 Estructura del Proyecto

```
inventario/
├── index.html          # Aplicación web principal
├── modelo.tflite       # Modelo de detección optimizado
├── README.md           # Esta documentación
└── ejemplos/           # Imágenes de ejemplo (opcional)
    ├── salon1.jpg
    └── salon2.jpg
```

## 🚀 Uso de la Aplicación

### Requisitos
- Navegador web moderno (Chrome, Firefox, Edge)
- Archivo `modelo.tflite` en la misma carpeta que `index.html`

### Instrucciones

1. **Abrir la aplicación**
   - Hacer doble clic en `index.html`
   - O abrir con un servidor local:
     ```bash
     python -m http.server 8000
     # Luego abrir: http://localhost:8000
     ```

2. **Cargar imagen**
   - Click en "Seleccionar Imagen del Salón"
   - Elegir imagen JPG/PNG del salón
   - Esperar a que el modelo procese

3. **Ver resultados**
   - Imagen con detecciones marcadas en azul
   - Cada objeto tiene su número de código
   - Tabla con inventario completo

## 🔧 Proceso de Entrenamiento

### 1. Preparación de Datos
```python
# Dataset anotado en Roboflow
- 50 imágenes del salón completo
- Múltiples objetos por imagen
- Anotaciones manuales precisas
- Split: 70% train, 20% val, 10% test
```

### 2. Entrenamiento
```python
from ultralytics import YOLO

modelo = YOLO('yolov8n.pt')

resultados = modelo.train(
    data='data.yaml',
    epochs=100,
    imgsz=640,
    batch=16,
    patience=20,
    device=0,
    optimizer='AdamW',
    lr0=0.01
)
```

### 3. Validación
```python
# Evaluar en conjunto de validación
metrics = modelo.val()

print(f"mAP50: {metrics.box.map50:.2%}")
print(f"Precision: {metrics.box.mp:.2%}")
```

### 4. Exportación
```python
# Convertir a TFLite
best_model = YOLO('best.pt')
best_model.export(format='tflite', imgsz=640)
```

## 📊 Tamaño del Modelo

| Versión | Tamaño | Precisión |
|---------|--------|-----------|
| PyTorch (.pt) | ~6 MB | 78.2% mAP50 |
| TFLite (.tflite) | ~3 MB | 78.2% mAP50 |
| TFLite (int8) | ~1.5 MB | ~75% mAP50 |

**Modelo entregado**: TFLite float16 (~3 MB) - Mejor balance precisión/tamaño

## 🔬 Tecnologías Utilizadas

- **YOLOv8**: Detección de objetos
- **TensorFlow Lite**: Optimización del modelo
- **TensorFlow.js**: Ejecución en navegador
- **Roboflow**: Anotación de imágenes
- **Google Colab**: Entrenamiento con GPU

## 📝 Notas Técnicas

### Threshold de Confianza
- **Default**: 0.25 (25%)
- Ajustable en el código: `CONFIG.CONF_THRESHOLD`

### IoU Threshold (NMS)
- **Default**: 0.45
- Evita detecciones duplicadas

### Formato de Entrada
- **Resolución**: 640x640
- **Normalización**: [-1, 1] (YOLO format)
- **Formato**: RGB

### Limitaciones
- Funciona mejor con imágenes similares al entrenamiento
- Requiere buena iluminación
- Los objetos muy pequeños pueden no detectarse

## 🎓 Información Académica

**Proyecto**: Inventario Automático del Salón de Cómputo  
**Materia**: BIG DATA - Módulo de Redes Convolucionales  
**Profesor**: Gerardo Muñoz  
**Programa**: Maestría en Ciencias de la Computación y las Comunicaciones  

## 📄 Licencia

Este proyecto es parte de un trabajo académico.

## 🔗 Enlaces

- **Modelo TFLite**: [Descargar desde Google Drive]([ENLACE_AQUI](https://colab.research.google.com/drive/1va8jqYFM36szJQn4p3SX9V6i0GhtEyXj?usp=sharing))
- **Dataset**: Roboflow (privado)
- **Código fuente**: GitHub

---

**Nota**: Si el modelo no está incluido en el repositorio por su tamaño, descárgalo del enlace de Google Drive y colócalo en la carpeta `inventario/` con el nombre `modelo.tflite`.
