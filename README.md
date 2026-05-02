# Kansushiji Classifier

Este repositório contém o classificador de caracteres Kuzushiji-49 treinado com Keras/TensorFlow.

## Como carregar o modelo
```python
from tensorflow.keras.models import load_model
model = load_model('models/model_2.h5')
```

## Especificações Técnicas
* **Arquivo de modelo:** `models/model_2.h5`
* **Shape de entrada:** 28x28 pixels (Grayscale).
* **Normalização:** A entrada deve ser convertida para o intervalo **0-1** (dividir por 255).
* **Padrão de imagem:** Fundo preto com letra branca (como o dataset original).
* **Mapeamento de classes:** Utilize o arquivo `k49_classmap.csv`. O índice previsto pelo modelo corresponde à coluna `index` do CSV.

## Exemplo de Inferência
```python
import numpy as np
import cv2
from tensorflow.keras.models import load_model

model = load_model('models/model_2.h5')
img = cv2.imread('caractere_teste.png', cv2.IMREAD_GRAYSCALE)
img = cv2.resize(img, (28, 28))
img = img.astype('float32') / 255.0
img = np.expand_dims(img, axis=(0, -1)) # Shape (1, 28, 28, 1)

predicao = model.predict(img)
classe_id = np.argmax(predicao)
print(f"Classe detectada: {classe_id}")
```