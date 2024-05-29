import os
import json
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


def mnist_dataset(batch_size):
    (x_train, y_train), _ = tf.keras.datasets.mnist.load_data()
    # The `x` arrays are in uint8 and have values in the [0, 255] range.
    # You need to convert them to float32 with values in the [0, 1] range.
    x_train = x_train / np.float32(255)
    y_train = y_train.astype(np.int64)
    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(60000).repeat().batch(batch_size)
    return train_dataset


def build_and_compile_cnn_model():
    model = tf.keras.Sequential(
        [
            tf.keras.layers.InputLayer(input_shape=(28, 28)),
            tf.keras.layers.Reshape(target_shape=(28, 28, 1)),
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
multi_worker_dataset = mnist_dataset(global_batch_size)

with strategy.scope():
    # Model building/compiling needs to be within `strategy.scope()`.
    multi_worker_model = build_and_compile_cnn_model()

multi_worker_model.fit(multi_worker_dataset, epochs=100, steps_per_epoch=70)
