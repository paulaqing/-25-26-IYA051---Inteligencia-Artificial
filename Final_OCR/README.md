# 🔍 Sistema OCR con CNN

Sistema de Reconocimiento Óptico de Caracteres desarrollado desde cero para la asignatura de Inteligencia Artificial. Reconoce texto digital y manuscrito sin usar librerías OCR externas.

---

## 📋 Descripción

Software OCR que convierte imágenes con texto (tipografía digital o escritura manual) a texto digital mediante una Red Neuronal Convolucional entrenada desde cero.

**Formatos soportados:** JPG, PNG, BMP, TIFF

---

## 🎯 Características

- ✅ Reconocimiento de caracteres digitales (imprenta)
- ✅ Reconocimiento de escritura manual
- ✅ Red Neuronal Convolucional propia (sin Tesseract)
- ✅ Preprocesamiento avanzado de imágenes
- ✅ Data Augmentation para mejor rendimiento

---

## 🏗️ Arquitectura

```
Imagen → Preprocesamiento → Segmentación → CNN → Texto
```

**Componentes principales:**
1. **DatasetLoader:** Carga imágenes de caracteres
2. **ImagePreprocessor:** Binarización, denoise, deskewing
3. **CharacterSegmenter:** Separa caracteres individuales
4. **CNN:** Red de 4 capas convolucionales + 3 densas
5. **OCRSystem:** Integra todo el pipeline

---

## 🚀 Uso Rápido

### En Google Colab:

```python
# 1. Montar Drive
from google.colab import drive
drive.mount('/content/drive')

# 2. Entrenar modelo
model, label_to_char = train_model_improved()

# 3. Reconocer texto
upload_and_recognize()
```

---

## 📁 Estructura del Dataset

```
OCR_Dataset/
├── mayusculas/A, B, C...
├── minusculas/a, b, c...
└── numeros/0, 1, 2...
```

---

## 📚 Tecnologías

- **TensorFlow/Keras** - Deep Learning
- **OpenCV** - Procesamiento de imágenes
- **NumPy** - Operaciones matriciales
- **Google Colab** - Entorno de desarrollo
