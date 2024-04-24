import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.utils import class_weight
from tensorflow_addons.metrics import F1Score
from sklearn.metrics import classification_report
import os
from keras.callbacks import ModelCheckpoint
import matplotlib
matplotlib.use('Agg')

class CustomAccuracy(tf.keras.metrics.Metric):
    def __init__(self, name='custom_accuracy', **kwargs):
        super(CustomAccuracy, self).__init__(name=name, **kwargs)
        self.correct = self.add_weight(name='tp', initializer='zeros')
        self.total = self.add_weight(name='fp', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        """
        Mise à jour de l'état des métriques de précision personnalisée.
        
        Args:
            y_true (tf.Tensor): Les vraies étiquettes.
            y_pred (tf.Tensor): Les étiquettes prédites.
            sample_weight (tf.Tensor, optional): Poids des échantillons. Defaults to None.
        """
        y_pred = tf.argmax(y_pred, axis=1)
        y_true = tf.argmax(y_true, axis=1)
        is_correct = tf.equal(y_pred, y_true)
        self.correct.assign_add(tf.reduce_sum(tf.cast(is_correct, tf.float32)))
        self.total.assign_add(tf.cast(tf.size(y_true), tf.float32))

    def result(self):
        """
        Calcule la précision personnalisée.
        
        Returns:
            tf.Tensor: La précision personnalisée.
        """
        return tf.math.divide_no_nan(self.correct, self.total)

    def reset_state(self):
        """
        Réinitialise l'état des métriques de précision personnalisée.
        """
        self.correct.assign(0.)
        self.total.assign(0.)

keyword = 'model_256'

# Crée un dossier pour les résultats s'il n'existe pas
if not os.path.exists(f'results_{keyword}'):
    os.makedirs(f'results_{keyword}')
    
# Définit le générateur de données pour l'entraînement
train_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.3)

# Définit le générateur de données pour la validation
validation_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.3)

# Définit le générateur de données pour le test
test_datagen = ImageDataGenerator(
    rescale=1./255)

# Prétraite les données d'entraînement
train_generator = train_datagen.flow_from_directory(
    'data/trainv2',
    target_size=(48,48),
    batch_size=64,
    color_mode='grayscale',
    class_mode='categorical',
    shuffle=True,
    subset='training'  # défini comme données d'entraînement
)

# Calcule les poids de classe
class_weights = class_weight.compute_class_weight(class_weight='balanced', classes=np.unique(train_generator.classes), y=train_generator.classes)

# Convertit les poids de classe en dictionnaire
class_weight_dict = dict(enumerate(class_weights))
print(class_weight_dict)


# Prétraite les données de validation
validation_generator = validation_datagen.flow_from_directory(
    'data/trainv2',
    target_size=(48,48),
    batch_size=64,
    color_mode='grayscale',
    class_mode='categorical',
    shuffle=True,
    subset='validation'  # défini comme données de validation
)

# Charge et prétraite les données de test
test_generator = test_datagen.flow_from_directory(
    'data/test',
    target_size=(48,48),
    batch_size=64,
    color_mode="grayscale",
    class_mode='categorical',
    shuffle=True
)

# Définit le modèle
emotion_model = Sequential()

emotion_model.add(Conv2D(32, kernel_size=(3,3), activation='relu', input_shape=(48,48,1)))
emotion_model.add(Conv2D(64, kernel_size=(3,3), activation='relu'))
emotion_model.add(MaxPooling2D(pool_size=(2,2)))
emotion_model.add(Dropout(0.5))

emotion_model.add(Conv2D(128, kernel_size=(3,3), activation='relu'))
emotion_model.add(MaxPooling2D(pool_size=(2,2)))
emotion_model.add(Dropout(0.25))

emotion_model.add(Flatten())
emotion_model.add(Dense(256, activation='relu'))
emotion_model.add(Dropout(0.25))
emotion_model.add(Dense(4, activation='softmax'))


# Définit le callback pour l'arrêt anticipé
early_stop = EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=10, verbose=1)



# Définit le programme de taux d'apprentissage
lr_schedule = ExponentialDecay(
    initial_learning_rate=0.001,
    decay_steps=100000,
    decay_rate=0.96,
    staircase=False)

# Définit l'optimiseur
optimizer = Adam(learning_rate=lr_schedule)

# Compile le modèle
emotion_model.compile(loss='categorical_crossentropy', 
                    optimizer=optimizer, 
                    metrics=[CustomAccuracy(), F1Score(num_classes=4)])

# Définit le callback pour enregistrer les poids du modèle
checkpoint = ModelCheckpoint(f'results_{keyword}/weights.' + '{epoch:02d}-{val_custom_accuracy:.2f}.hdf5', monitor='val_custom_accuracy', verbose=1, save_best_only=True, mode='max')

# Entraîne le modèle
emotion_model_info = emotion_model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples//64,
    epochs=200,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples//64,
    class_weight=class_weight_dict,
    callbacks=[early_stop, checkpoint]
)

# Tester le modèle sur les données d'entraînement
train_loss, train_custom_accuracy, train_f1 = emotion_model.evaluate(train_generator)
print('Train loss:', train_loss)
print('Train accuracy:', train_custom_accuracy)
print('Train F1 score:', train_f1)

# Tester le modèle sur les données de validation
val_loss, val_custom_accuracy, val_f1 = emotion_model.evaluate(validation_generator)
print('Validation loss:', val_loss)
print('Validation accuracy:', val_custom_accuracy)
print('Validation F1 score:', val_f1)

# Tester le modèle sur les données de test
test_loss, test_custom_accuracy, test_f1 = emotion_model.evaluate(test_generator)
print('Test loss:', test_loss)
print('Test accuracy:', test_custom_accuracy)
print('Test F1 score:', test_f1)

# Exporter les résultats dans un fichier
with open(f'results_{keyword}.txt', 'w') as f:
    f.write(f'Train loss: {train_loss}\n')
    f.write(f'Train accuracy: {train_custom_accuracy}\n')
    f.write(f'Train F1 score: {train_f1}\n')
    f.write(f'Validation loss: {val_loss}\n')
    f.write(f'Validation accuracy: {val_custom_accuracy}\n')
    f.write(f'Validation F1 score: {val_f1}\n')
    f.write(f'Test loss: {test_loss}\n')
    f.write(f'Test accuracy: {test_custom_accuracy}\n')
    f.write(f'Test F1 score: {test_f1}\n')

# Calculer la matrice de confusion pour l'ensemble d'entraînement
confusion_mtx_train = tf.math.confusion_matrix(train_generator.classes, np.argmax(emotion_model.predict(train_generator), axis=1))

# Convertir la matrice de confusion en tableau numpy
confusion_mtx_train = confusion_mtx_train.numpy()

# Afficher la matrice de confusion pour l'ensemble d'entraînement
plt.figure(figsize=(10, 8))
sns.heatmap(confusion_mtx_train, annot=True, fmt="d", cmap='Blues')
plt.title(f"Confusion Matrix for {keyword} Training Set")
plt.ylabel('True label')
plt.xlabel('Predicted label')

# Enregistrer le graphique sous forme de fichier image
plt.savefig(f'results_{keyword}\confusion_matrix__{keyword}_train.png')
plt.close()

# Calculer la matrice de confusion pour l'ensemble de validation
confusion_mtx_val = tf.math.confusion_matrix(validation_generator.classes, np.argmax(emotion_model.predict(validation_generator), axis=1))

# Convertir la matrice de confusion en tableau numpy
confusion_mtx_val = confusion_mtx_val.numpy()

# Afficher la matrice de confusion pour l'ensemble de validation
plt.figure(figsize=(10, 8))
sns.heatmap(confusion_mtx_val, annot=True, fmt="d", cmap='Blues')
plt.title(f"Confusion Matrix for {keyword} Validation Set")
plt.ylabel('True label')
plt.xlabel('Predicted label')

# Enregistrer le graphique sous forme de fichier image
plt.savefig(f'results_{keyword}/confusion_matrix_{keyword}_val.png')
plt.close()

# Calculer la matrice de confusion pour l'ensemble de test
confusion_mtx_test = tf.math.confusion_matrix(test_generator.classes, np.argmax(emotion_model.predict(test_generator), axis=1))

# Convertir la matrice de confusion en tableau numpy
confusion_mtx_test = confusion_mtx_test.numpy()

# Afficher la matrice de confusion pour l'ensemble de test
plt.figure(figsize=(10, 8))
sns.heatmap(confusion_mtx_test, annot=True, fmt="d", cmap='Blues')
plt.title(f"Confusion Matrix for {keyword} Test Set")
plt.ylabel('True label')
plt.xlabel('Predicted label')

# Enregistrer le graphique sous forme de fichier image
plt.savefig(f'results_{keyword}/confusion_matrix_{keyword}_test.png')
plt.close()

# Prédire les classes
y_pred = emotion_model.predict(test_generator)
y_pred_classes = np.argmax(y_pred, axis=1)

# Générer le rapport de classification
report = classification_report(test_generator.classes, y_pred_classes, target_names=test_generator.class_indices.keys(), output_dict=True)

# Convertir le rapport en DataFrame
report_df = pd.DataFrame(report).transpose()

# Enregistrer le rapport en tant que fichier CSV
report_df.to_csv(f'results_{keyword}/classification_report_{keyword}.csv')


# Obtenir l'historique du processus d'entraînement
history = emotion_model_info.history
# Créer un DataFrame à partir de l'historique
df = pd.DataFrame(history)
# Enregistrer le DataFrame en tant que fichier CSV
df.to_csv(f'results_{keyword}/training_history_{keyword}.csv', index=False)


# Sauvegarder la structure du modèle dans un fichier json
model_json = emotion_model.to_json()
with open(f"results_{keyword}/model_{keyword}.json", "w") as json_file:
    json_file.write(model_json)

# Sauvegarder le modèle
emotion_model.save_weights(f'results_{keyword}/emotion_model_{keyword}.h5')