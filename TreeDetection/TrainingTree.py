import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from TreeDataset import TreeDataset
import matplotlib.pyplot as plt

def main():
    print("Loading dataset...")
    patches, labels = TreeDataset.load_dataset()
    X = np.array(patches)
    y = np.array(labels)
    indices = np.arange(len(X))
    np.random.shuffle(indices)
    X = X[indices]
    y = y[indices]
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full, test_size=0.2, random_state=42, stratify=y_train_full
    )
    X_train = X_train / 255.0
    X_val = X_val / 255.0
    X_test = X_test / 255.0

    input_shape = X_train.shape[1:]
    model = build_cnn_model(input_shape)
    early_stopping = keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True
    )
    reduce_lr = keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=2,
        min_lr=1e-6
    )

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=5,
        batch_size=32,
        callbacks=[early_stopping, reduce_lr],
        verbose=1
    )

    loss, accuracy, precision, recall = model.evaluate(X_test, y_test)
    print(f"\nTest Accuracy: {accuracy * 100:.2f}%")
    print(f"Test Precision: {precision:.2f}")
    print(f"Test Recall: {recall:.2f}")
    model.save('tree_cnn_classifier.h5')
    print("Model saved as 'tree_cnn_classifier.h5'.")
    plot_history(history)

def build_cnn_model(input_shape):
    model = keras.Sequential([
        layers.Input(shape=input_shape),
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPooling2D(),
        layers.BatchNormalization(),

        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D(),
        layers.BatchNormalization(),

        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D(),
        layers.BatchNormalization(),

        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(1, activation='sigmoid'),
    ])
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy', keras.metrics.Precision(), keras.metrics.Recall()]
    )
    return model

def plot_history(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    precision = history.history.get('precision')
    val_precision = history.history.get('val_precision')
    recall = history.history.get('recall')
    val_recall = history.history.get('val_recall')
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(acc) + 1)

    plt.figure(figsize=(14, 8))

    plt.subplot(2, 2, 1)
    plt.plot(epochs, acc, 'b-', label='Training accuracy')
    plt.plot(epochs, val_acc, 'r-', label='Validation accuracy')
    plt.title('Training and validation accuracy')
    plt.legend()

    plt.subplot(2, 2, 2)
    plt.plot(epochs, loss, 'b-', label='Training loss')
    plt.plot(epochs, val_loss, 'r-', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()

    if precision and val_precision:
        plt.subplot(2, 2, 3)
        plt.plot(epochs, precision, 'b-', label='Training precision')
        plt.plot(epochs, val_precision, 'r-', label='Validation precision')
        plt.title('Training and validation precision')
        plt.legend()

    if recall and val_recall:
        plt.subplot(2, 2, 4)
        plt.plot(epochs, recall, 'b-', label='Training recall')
        plt.plot(epochs, val_recall, 'r-', label='Validation recall')
        plt.title('Training and validation recall')
        plt.legend()

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
