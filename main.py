import random
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import keras as keras
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Flatten, BatchNormalization
from keras.layers import RandomRotation, RandomZoom, RandomTranslation
from keras.models import Sequential
from keras.models import load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from tabulate import tabulate

filepath = './data/'
train_images = np.load(filepath+'k49-train-imgs.npz')['arr_0']
train_labels = np.load(filepath+'k49-train-labels.npz')['arr_0']
test_images = np.load(filepath+'k49-test-imgs.npz')['arr_0']
test_labels = np.load(filepath+'k49-test-labels.npz')['arr_0']
classmap_df = pd.read_csv(filepath+'k49_classmap.csv')

print('KMNIST train images shape:', train_images.shape)
print('KMNIST train labels shape:', train_labels.shape)
print('KMNIST test images shape:', test_images.shape)
print('KMNIST test labels shape:', test_labels.shape)

print('KMNIST character map shape:', classmap_df.shape)

print(classmap_df.head())

labels_series = pd.Series(train_labels)
plt.figure(figsize=(10,4))
sns.countplot(x=labels_series)
plt.title('Count Plot of Labels')
plt.xlabel('Labels')
plt.ylabel('Count')
plt.tight_layout()

def select_random_images_by_labels(images, labels, num_labels=5, num_images_per_label=5):
    unique_labels = np.unique(labels)
    random_labels = random.sample(list(unique_labels), min(num_labels, len(unique_labels)))
    selected_images = []
    selected_labels = []

    for label in random_labels:
        indices = np.where(labels == label)[0]
        random_indices = random.sample(list(indices), min(num_images_per_label, len(indices)))
        selected_images.extend(images[random_indices])
        selected_labels.extend([label] * len(random_indices))
    return np.array(selected_images), np.array(selected_labels)

def plot_random_images(images, labels, num_labels=5, num_images_per_label=5):
    selected_images, selected_labels = select_random_images_by_labels(images, labels, num_labels, num_images_per_label)
    num_rows = min(len(selected_images), num_images_per_label*num_labels)
    num_cols = num_images_per_label
    fig, axes = plt.subplots(num_rows // num_cols, num_cols, figsize=(10, 6))
    for i in range(num_rows):
        axes[i // num_cols, i % num_cols].imshow(selected_images[i], cmap='gray')
        axes[i // num_cols, i % num_cols].set_title(f"Label:{selected_labels[i]}")
        axes[i // num_cols, i % num_cols].axis('off')

    plt.tight_layout()
    plt.show()

plot_random_images(train_images, train_labels, num_labels=5, num_images_per_label=5)

X = train_images.reshape(-1, 28, 28, 1)
y = train_labels

X_test = test_images.reshape(-1, 28, 28, 1)
y_test = test_labels

test_size = 0.2
random_state = 42
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)

print("Train data shapes - X: ", X_train.shape, " Y: ", y_train.shape)
print("Validation data shapes - X: ", X_val.shape, " Y: ", y_val.shape)
print("Test data shapes - X: ", X_test.shape, " Y: ", y_test.shape)

def plot_model_validation(history):
    plt.figure(figsize=(10,4))

    plt.subplot(1, 2, 1)
    plt.plot(history['loss'], label='Loss')
    plt.plot(history['val_loss'], label='val_loss')
    plt.title('Loss Function evolution during training')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history['accuracy'], label='Accuracy')
    plt.plot(history['val_accuracy'], label='val_accuracy')
    plt.title('Accuracy evolution during training')
    plt.legend()
    plt.ylim((0,1))
    plt.tight_layout()
    plt.show()

IMG_ROWS, IMG_COLS = 28, 28
NUM_CLASSES = 49
BATCH_SIZE = 1024
EPOCHS = 100

model = Sequential()

model.add(RandomRotation(factor = 0.05, input_shape=(IMG_ROWS, IMG_COLS, 1)))
model.add(RandomZoom(height_factor = 0.1, width_factor = 0.1))
model.add(RandomTranslation(height_factor = 0.1, width_factor = 0.1))

model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', padding = 'same'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3), activation='relu', padding = 'same'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(128, (3, 3), activation='relu', padding = 'same'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))


model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(BatchNormalization()) 
model.add(Dropout(0.5))
model.add(Dense(NUM_CLASSES, activation='softmax'))


print("Calculando pesos individuais (Sample Weights) para equilibrar as classes...")

# 1. Calcula o peso geral de cada classe
pesos_classes = class_weight.compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train), 
    y=y_train
)
# Cria um dicionário mapeando "Classe -> Peso"
pesos_dict = dict(zip(np.unique(y_train), pesos_classes))

for classe in pesos_dict:
    pesos_dict[classe] = np.sqrt(pesos_dict[classe])

# 2. Cria um array de peso para CADA imagem do dataset de treino
sample_weights = np.array([pesos_dict[label] for label in y_train])

# 3. Adiciona os pesos individuais (sample_weights) dentro do Dataset!
train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train, sample_weights)).shuffle(buffer_size=10000).batch(BATCH_SIZE)

val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val)).batch(BATCH_SIZE)

loss_object = tf.keras.losses.SparseCategoricalCrossentropy()
# 1. Definimos o comportamento da nossa taxa de aprendizado
initial_learning_rate = 0.001 # A taxa padrão inicial do Adam (passos largos)

lr_schedule = ExponentialDecay(
    initial_learning_rate,
    decay_steps=2000, # A cada 2000 passos (batches processados)...
    decay_rate=0.9,   # ...multiplica a taxa por 0.9 (ou seja, cai 10%)
    staircase=True    # Faz a queda em formato de "escada" em vez de uma rampa suave
)

# 2. Entregamos esse cronograma para o Adam
optimizer = Adam(learning_rate=lr_schedule)

train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
val_loss = tf.keras.metrics.Mean(name='val_loss')
val_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='val_accuracy')

@tf.function
def train_step(images, labels, s_weights): # <-- Recebe os pesos do lote
    with tf.GradientTape() as tape:
        predictions = model(images, training=True)
        # 4. Aplica os pesos no cálculo do erro! O TensorFlow faz a matemática.
        loss = loss_object(labels, predictions, sample_weight=s_weights)
    
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    
    train_loss(loss)
    train_accuracy(labels, predictions)

@tf.function
def val_step(images, labels):
    predictions = model(images, training=False)
    v_loss = loss_object(labels, predictions)
    
    val_loss(v_loss)
    val_accuracy(labels, predictions)

history_dict = {'loss': [], 'accuracy': [], 'val_loss': [], 'val_accuracy': []}

print("Iniciando treinamento customizado com GradientTape...")
for epoch in range(EPOCHS):
    # Zera as métricas no início de cada época
    train_loss.reset_state()
    train_accuracy.reset_state()
    val_loss.reset_state()
    val_accuracy.reset_state()

    # Itera sobre os lotes do conjunto de treinamento
    for images, labels, s_weights in train_dataset:
        train_step(images, labels, s_weights)
        
    # Itera sobre os lotes do conjunto de validação
    for val_images, val_labels in val_dataset:
        val_step(val_images, val_labels)

    # Salva os resultados da época para o gráfico depois
    history_dict['loss'].append(train_loss.result().numpy())
    history_dict['accuracy'].append(train_accuracy.result().numpy())
    history_dict['val_loss'].append(val_loss.result().numpy())
    history_dict['val_accuracy'].append(val_accuracy.result().numpy())

    # Imprime o log da época (equivalente ao verbose=2)
    print(f'Epoch {epoch + 1}/{EPOCHS} - '
          f'loss: {train_loss.result():.4f} - '
          f'accuracy: {train_accuracy.result():.4f} - '
          f'val_loss: {val_loss.result():.4f} - '
          f'val_accuracy: {val_accuracy.result():.4f}')

model.save('models/model_2.h5')

plot_model_validation(history_dict)

def print_classification_metrics(y_true, predicted_classes):
    report = classification_report(y_true, predicted_classes, output_dict=True)
    accuracy = round(report['accuracy'], 2)
    macro_avg = report['macro avg']
    weighted_avg = report['weighted avg']

    data = {
        'Metric': ['Recall', 'Precision', 'F1-score', 'Support'],
        'Macro Average': [macro_avg['recall'], macro_avg['precision'], macro_avg['f1-score'], macro_avg['support']],
        'Weighted Average': [weighted_avg['recall'], weighted_avg['precision'], weighted_avg['f1-score'],
                             weighted_avg['support']]
    }

    df = pd.DataFrame(data)
    rounded_df = df.round(2)
    df.set_index("Metric", inplace=True)
    print(tabulate(rounded_df, headers='keys', tablefmt='pretty'))
    print('Accuracy:', accuracy)
    model = load_model('models/model_1.h5')
    model.summary()

    print("Evaluating on Validation data:")
    predictions = model.predict(X_val)
    predicted_classes = predictions.argmax(axis=1)
    print_classification_metrics(y_val, predicted_classes)

    print("Evaluating on Test data:")
    predictions = model.predict(X_test)
    predicted_classes = predictions.argmax(axis=1)
    print_classification_metrics(y_test, predicted_classes)

    model = load_model('models/model_2.h5')
    model.summary()

    print("Evaluating on Validation data:")
    predictions = model.predict(X_val)
    predicted_classes = predictions.argmax(axis=1)
    print_classification_metrics(y_val, predicted_classes)

    print('Evaluating on Test data:')
    predictions = model.predict(X_test)
    predicted_classes = predictions.argmax(axis=1)
    print_classification_metrics(y_test, predicted_classes)

    best_model = load_model('models/model_2.h5')
    model.summary()

predictions = model.predict(X_test)
predicted_classes = predictions.argmax(axis=1)

report = classification_report(y_test, predicted_classes)
print(report)

# best_model.save('models/best_model.h5')

def print_pred_images(X, y, prediction, show_correct=True, num_images_to_plot=5):
    if show_correct:
        indices = [i for i in range(len(prediction)) if prediction[i] == y[i]]
        indices_to_display = random.sample(indices, num_images_to_plot)
        cmap = 'Greens'
        title = 'Sample of Correctly Classified Images'
    else:
        indices = [i for i in range(len(prediction)) if prediction[i] != y[i]]
        indices_to_display = random.sample(indices, num_images_to_plot)
        cmap = 'Reds'
        title = 'Sample of Misclassified Images'

    fig,ax = plt.subplots(1, num_images_to_plot, figsize=(12,3))

    for i, idx in enumerate(indices_to_display):
        ax[i].imshow(X[idx], cmap=cmap)
        ax[i].set_title(f"True: {y[idx]}, Predicted: {prediction[idx]}")
        ax[i].axis('off')

    fig.suptitle(title, fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

print_pred_images(X_test, y_test, predicted_classes)

print_pred_images(X_test, y_test, predicted_classes, 0)