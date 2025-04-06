from tensorflow.keras import models, layers
import tensorflow as tf
import json
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
from preprocess import get_data_generators
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt
from tensorflow.keras import models, layers 

# Getting training generator
train_dir = "dataset/"
train_generator, validation_generator = get_data_generators(train_dir)

# Crating CNN model  
model = models.Sequential()

# Conv Block 1
model.add(layers.Conv2D(32, (3,3), padding='same', input_shape=(64,64,3)))
model.add(layers.BatchNormalization())
model.add(layers.Activation('relu'))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Dropout(0.3))

# Conv Block 2
model.add(layers.Conv2D(64, (3,3), padding='same'))
model.add(layers.BatchNormalization())
model.add(layers.Activation('relu'))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Dropout(0.3))

# Conv Block 3
model.add(layers.Conv2D(128, (3,3), padding='same'))
model.add(layers.BatchNormalization())
model.add(layers.Activation('relu'))
model.add(layers.MaxPooling2D((2,2)))  
model.add(layers.Dropout(0.3))

# Conv Block 4
model.add(layers.Conv2D(128, (3,3), padding='same'))
model.add(layers.BatchNormalization())
model.add(layers.Activation('relu')) 
model.add(layers.Dropout(0.3))

# Conv Block 5
model.add(layers.Conv2D(256, (3,3), padding='same'))
model.add(layers.BatchNormalization())
model.add(layers.Activation('relu'))
model.add(layers.Dropout(0.3))

# Fully Connected Layers
model.add(layers.Flatten())
model.add(layers.Dropout(0.4))
model.add(layers.Dense(512, kernel_regularizer=tf.keras.regularizers.l2(0.001)))
model.add(layers.BatchNormalization())
model.add(layers.Activation('relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(128, kernel_regularizer=tf.keras.regularizers.l2(0.001)))
model.add(layers.BatchNormalization())
model.add(layers.Activation('relu'))
model.add(layers.Dropout(0.4))
model.add(layers.Dense(43, activation='softmax'))

model.summary()

optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
model.compile(optimizer=optimizer,
            loss=tf.keras.losses.SparseCategoricalCrossentropy(), # using SparseCategoricalCrossentropy() as the labels are not one-hot encoded
            metrics=["accuracy"])

cls_labels = np.unique(train_generator.classes)
class_weights = compute_class_weight(class_weight="balanced", classes=cls_labels, y=train_generator.classes)
class_weights_dict = dict(zip(cls_labels, class_weights))

lr_scheduler = ReduceLROnPlateau(
    monitor="val_loss",  # Watch validation loss
    factor=0.5,          # Reduce LR by half when triggered
    patience=3,          # Wait 3 epochs before reducing LR
    verbose=1,           # Print updates
    min_lr=1e-6          # Minimum possible learning rate
)

# Stopping the training early if the model is not improving
early_stopping = EarlyStopping(monitor='val_accuracy', patience=3, restore_best_weights=True)

cnn_model = model.fit(train_generator, validation_data=validation_generator, epochs=40, callbacks=[early_stopping, lr_scheduler], class_weight=class_weights_dict)

print(cnn_model.history.keys())
train_acc = cnn_model.history["accuracy"]
val_acc =cnn_model.history["val_accuracy"]

train_loss = cnn_model.history['loss']
val_loss = cnn_model.history['val_loss']

class_indices = train_generator.class_indices

model.save("assets/cnn_model.keras", save_format="keras")

# Saving the data
with open("class_indices.json", "w") as f:
    json.dump(class_indices, f)
    
    
# epoch range
epochs = range(1, len(train_acc) + 1)

plt.figure(figsize=(9, 4))

# Subplot 1: training and validation Accuracy
plt.subplot(1, 2, 1)  
plt.plot(epochs, train_acc, 'b-', label='Training Accuracy')
plt.plot(epochs, val_acc, 'r-', label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Training vs Validation Accuracy')
plt.legend()

# Subplot 2: training and validation Loss
plt.subplot(1, 2, 2)  
plt.plot(epochs, train_loss, 'b-', label='Training Loss')
plt.plot(epochs, val_loss, 'r-', label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training vs Validation Loss')
plt.legend()

plt.tight_layout() 
plt.show()

