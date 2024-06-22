"""
Same code as tensorflow/multi_worker_main.py with the addition of:

- Custom dataset
- Callbacks:
    - Checkpointing
    - Learning rate scheduling
    - Tensorboard
- Model saving
"""

import os
import json
import pickle
import tensorflow as tf
import numpy as np

# Set up environment variables for the cluster and task configuration

# Atenea -> 138.4.32.110
# Minerva -> 138.4.32.111

# Export the following variables in each node (PC)

# In PC1 (ATENEA)
"""
export TF_CONFIG='{
    "cluster": {
        "worker": ["138.4.32.110:12345", "138.4.32.111:23456"]
    },
    "task": {"type": "worker", "index": 0}
}'
"""

# In PC2 (MINERVA)
"""
export TF_CONFIG='{
    "cluster": {
        "worker": ["138.4.32.110:12345", "138.4.32.111:23456"]
    },
    "task": {"type": "worker", "index": 1}
}'
"""

# Then execute this "python multi_worker_main.py" in each node (PC)


def unpickle(file):
    with open(file, "rb") as fo:
        dict = pickle.load(fo, encoding="bytes")
    return dict


def load_cifar10(data_dir):
    def load_batch(file):
        batch = unpickle(file)
        data = batch[b"data"]
        labels = batch[b"labels"]
        return data, labels

    # Load training data
    train_data = []
    train_labels = []
    for i in range(1, 6):
        data, labels = load_batch(os.path.join(data_dir, f"data_batch_{i}"))
        train_data.append(data)
        train_labels.extend(labels)

    train_data = np.concatenate(train_data)
    train_data = train_data.reshape(-1, 32, 32, 3).astype("float32") / 255.0  # Normalize data
    train_labels = np.array(train_labels)

    # Load test data
    test_data, test_labels = load_batch(os.path.join(data_dir, "test_batch"))
    test_data = test_data.reshape(-1, 32, 32, 3).astype("float32") / 255.0  # Normalize data
    test_labels = np.array(test_labels)

    # Create TensorFlow datasets
    train_ds = tf.data.Dataset.from_tensor_slices((train_data, train_labels)).shuffle(50000).batch(32)
    test_ds = tf.data.Dataset.from_tensor_slices((test_data, test_labels)).batch(32)

    return train_ds, test_ds


def build_and_compile_cnn_model():
    model = tf.keras.Sequential(
        [
            tf.keras.layers.InputLayer(input_shape=(32, 32, 3)),
            tf.keras.layers.Reshape(target_shape=(32, 32, 3)),
            tf.keras.layers.Conv2D(32, 3, activation="relu"),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dense(10),
        ]
    )
    model.compile(
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        optimizer=tf.keras.optimizers.SGD(learning_rate=0.001),
        metrics=["accuracy"],
    )
    return model


# Learning rate decay
def decay(epoch):
    if epoch < 3:
        return 1e-3
    if 3 <= epoch < 7:
        return 1e-4
    return 1e-5


# Custom callback to print the learning rate
class PrintLR(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        print(f"\nLearning rate for epoch {epoch + 1} is {multi_worker_model.optimizer.lr.numpy()}")


# For a single node (PC), the code to train would simply be:
# batch_size = 64
# single_worker_dataset = mnist_dataset(batch_size)
# single_worker_model = build_and_compile_cnn_model()
# single_worker_model.fit(single_worker_dataset, epochs=100, steps_per_epoch=70)


# For multi node training, the code goes as follows:
per_worker_batch_size = 64
tf_config = json.loads(os.environ["TF_CONFIG"])
num_workers = len(tf_config["cluster"]["worker"])

print(f"{tf_config=}")
print(f"{num_workers=}")

strategy = tf.distribute.MultiWorkerMirroredStrategy()

global_batch_size = per_worker_batch_size * num_workers
train_ds, test_ds = load_cifar10(f"../data/cifar10")

with strategy.scope():
    # Model building/compiling needs to be within `strategy.scope()`.
    multi_worker_model = build_and_compile_cnn_model()

# Callbacks
callbacks = [
    tf.keras.callbacks.TensorBoard(log_dir="logs"),
    tf.keras.callbacks.ModelCheckpoint(filepath="ckpts/epoch_{epoch}.ckpt", save_weights_only=True, verbose=1),
    tf.keras.callbacks.LearningRateScheduler(decay),
    PrintLR(),
]

# Train the model
multi_worker_model.fit(train_ds, epochs=5, steps_per_epoch=70, callbacks=callbacks)

# Evaluate the model
multi_worker_model.evaluate(test_ds)

# Save the model only on the "master" node
if tf_config["task"]["index"] == 0:
    multi_worker_model.save("my_model.keras")
