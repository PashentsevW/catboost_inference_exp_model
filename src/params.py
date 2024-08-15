RANDOM_STATE = 94

DATASET_PARAMS = {"n_samples": 100_000, "n_features": 40, "n_classes": 2, "random_state": RANDOM_STATE}

TEST_SIZE = 0.3

CATBOOST_MODEL_PARAMS = {
    "loss_function": "Logloss",
    "iterations": 1000,
    "learning_rate": 0.01,
    "random_state": RANDOM_STATE,
    "verbose": 100,
}

EVAL_METRICS = ["Precision", "Recall"]
