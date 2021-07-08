import adanet
import os
import time
import tensorflow as tf
import adanet
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.metrics import mean_squared_error as mse

BATCH_SIZE = 32
RANDOM_SEED = 42
LEARNING_RATE = 0.001
TRAIN_STEPS = 60000
ADANET_ITERATIONS = 3
LAYER_SIZE = 64
ADANET_LAMBDA = 0.015
TRAINING_STEPS = 60
FEATURES_KEY = "x"

(x_train, y_train), (
    x_test,
    y_test,
) = tf.keras.datasets.boston_housing.load_data()
model_dir = os.path.join(
    os.getcwd(), "basic_adanet_dnn_" + str(int(time.time()))
)
head = tf.estimator.RegressionHead(
    loss_reduction=tf.losses.Reduction.SUM_OVER_BATCH_SIZE
)
feature_columns = [
    tf.feature_column.numeric_column(
        key=FEATURES_KEY, shape=(x_train.shape[1],)
    )
]


def input_fn(partition, training, batch_size):
    def _input_fn():

        if partition == "train":
            dataset = tf.data.Dataset.from_tensor_slices(
                (
                    {FEATURES_KEY: tf.math.log1p(x_train)},
                    tf.math.log1p(y_train),
                )
            )

            if training:
                dataset = dataset.shuffle(
                    10 * batch_size, seed=RANDOM_SEED
                ).repeat()

            dataset = dataset.batch(batch_size)

        elif partition == "test":
            dataset = tf.data.Dataset.from_tensor_slices(
                ({FEATURES_KEY: tf.math.log1p(x_test)}, tf.math.log1p(y_test))
            )

            dataset = dataset.batch(batch_size)

        else:
            dataset = tf.data.Dataset.from_tensor_slices(
                {FEATURES_KEY: tf.math.log1p(x_test)}
            )
            dataset = dataset.batch(batch_size)
        return dataset

    return _input_fn


estimator = adanet.AutoEnsembleEstimator(
    head=head,
    candidate_pool={
        "dnn2": tf.estimator.DNNEstimator(
            head=head,
            feature_columns=feature_columns,
            hidden_units=[LAYER_SIZE, LAYER_SIZE],
            model_dir=model_dir,
            optimizer="Adagrad",
        ),
        "dnn3": tf.estimator.DNNEstimator(
            head=head,
            feature_columns=feature_columns,
            hidden_units=[LAYER_SIZE, LAYER_SIZE, LAYER_SIZE],
            model_dir=model_dir,
            optimizer="Adagrad",
        ),
    },
    max_iteration_steps=TRAIN_STEPS // ADANET_ITERATIONS,
    model_dir=model_dir,
    adanet_lambda=ADANET_LAMBDA,
)


estimator.train(
    input_fn=input_fn("train", training=True, batch_size=BATCH_SIZE),
    steps=TRAIN_STEPS,
)

metrics = estimator.evaluate(
    input_fn=input_fn("train", training=False, batch_size=BATCH_SIZE),
    steps=None,
)

predictions = estimator.predict(
    input_fn=input_fn("pred", training=False, batch_size=y_test.shape[0]),
    yield_single_examples=False,
)

predictions_list = list(predictions)
predictions_array = predictions_list[0]["predictions"]

# convert original data to log scale.
y_test_log = tf.math.log1p(y_test)

results = pd.DataFrame(
    {"actual": y_test_log, "hypothesis": np.squeeze(predictions_array)}
)

print(
    "Basic Model MSE is {:,}".format(
        round(mse(results.actual.values, results.hypothesis.values), 3)
    )
)
