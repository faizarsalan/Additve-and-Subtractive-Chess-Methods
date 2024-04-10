import pickle
import matplotlib.pyplot as plt
import numpy as np
import chess
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, CSVLogger

# Define the mapping dictionary
mapping = {
    'P': 1, 'N': 2, 'B': 3, 'R': 4, 'Q': 5, 'K': 6,
    'p': -1, 'n': -2, 'b': -3, 'r': -4, 'q': -5, 'k': -6
}

def create_model():
    model = tf.keras.models.Sequential([
    tf.keras.layers.InputLayer(input_shape=(8, 8, 1)),
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Dropout(0.25),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(1)
    ])

    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mean_squared_error'])
    return model

def fen_to_board_tensor(fen):
    board = chess.Board(fen)
    tensor = [[0 for _ in range(8)] for _ in range(8)]
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            tensor[7 - square // 8][square % 8] = mapping[piece.symbol()]
    return tensor

# Loading datasets
X_path = "../X_data_gmonly_addition.pkl"
y_path = "../y_data_gmonly_addition.pkl"

print(f"Loading X data on Progress ⌛")
with open(X_path, 'rb') as f:
    X = pickle.load(f)
print("X is Loaded ✅")

print(f"Loading y data on Progress ⌛")
with open(y_path, 'rb') as f:
    y = pickle.load(f)
print("y is Loaded ✅")

# Splitting the dataset
X_train, X_test_temp, y_train, y_test_temp = train_test_split(X, y, test_size=0.2,random_state=42) # 80%
X_test, X_val, y_test, y_val = train_test_split(X_test_temp, y_test_temp, test_size=0.5,random_state=42) # 20% --> 10% 10% hence the 0.5

# Reshaped
X_train_reshaped = np.array(X_train).reshape(len(X_train), 8, 8, 1)
X_test_reshaped = np.array(X_test).reshape(len(X_test), 8, 8, 1)
X_val_reshaped = np.array(X_val).reshape(len(X_val), 8, 8, 1)

y_train_np = np.array(y_train)
y_val_np = np.array(y_val)

# Model Configurations
min_seed = 1
max_seed = 11
constant_batch_size = 64

# Performance Metrices
total_metrics = []

for SEED in range(min_seed, max_seed):
    print(f"Training with SEED: {SEED}")
    tf.random.set_seed(SEED)
    future_file = f'{constant_batch_size}seed_{SEED}.h5'
    history_file = f'hist{constant_batch_size}seed_{SEED}.csv'

    # Create a new instance
    model = create_model()

    # Callbacks
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=1, restore_best_weights=True)
    model_checkpoint = ModelCheckpoint(future_file, monitor='val_loss', save_best_only=True, verbose=1)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, verbose=1, min_delta=0.0001)
    logs = CSVLogger(filename=history_file)

    callbacks_list = [model_checkpoint, early_stopping, reduce_lr, logs]

    history = model.fit(
        X_train_reshaped,
        y_train_np,
        epochs=100,
        batch_size=constant_batch_size,
        validation_data=(X_val_reshaped, y_val_np),
        callbacks=callbacks_list
    )
    
    # Plotting training and validation loss
    method_name = 'ADDITIVE'

    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title(f'SEED {SEED} - Training & Validation Loss ({method_name})')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    # Save the plot to an image
    plt.savefig(f'lossplot_seed_{SEED}.svg', format='svg')
    plt.clf()

    # Performance Metrics
    from sklearn.metrics import mean_squared_error
    mse = mean_squared_error(y_test, model.predict(X_test))

    from sklearn.metrics import mean_squared_error
    rmse = np.sqrt(mse)

    from sklearn.metrics import mean_absolute_error
    mae = mean_absolute_error(y_test, model.predict(X_test))

    from sklearn.metrics import r2_score
    r2 = r2_score(y_test, model.predict(X_test))

    # the model metric
    model_metric = {"seed": SEED, 'mse':mse, 'rmse':rmse, 'mae':mae, 'r2':r2}

    #save it into the list
    total_metrics.append(model_metric)

save_path = f'eval{constant_batch_size}_{min_seed}_{max_seed}.txt'
# Save results to a file
with open(save_path, 'w') as file:
    for model_metric in total_metrics:
        seed = model_metric['seed']
        mse = model_metric['mse']
        rmse = model_metric['rmse']
        mae = model_metric['mae']
        r2 = model_metric['r2']
        file.write(f"{seed};{mse};{rmse};{mae};{r2}\n")

print(f"Results saved to {save_path}")
import time
print(time.ctime())