import argparse
import json
import logging

import catboost
import numpy
from catboost.utils import eval_metric
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

import constants
import params

if __name__ == "__main__":
    argparser = argparse.ArgumentParser()

    argparser.add_argument(
        "--output-model-path",
        help="Path to model`s binary",
        required=False,
        default=str(constants.MODELS_PATH / "model.bin"),
        type=str,
    )
    argparser.add_argument(
        "--output-features-path",
        help="Path to model`s features",
        required=False,
        default=str(constants.DATA_PATH / "features"),
        type=str,
    )
    argparser.add_argument(
        "--output-metrics-path",
        help="Path to model`s metrics",
        required=False,
        default=str(constants.REPORTS_PATH / "metrics.json"),
        type=str,
    )
    argparser.add_argument(
        "--log-level",
        help="DEBUG,INFO,WARNING,ERROR",
        required=False,
        default=logging.INFO,
        type=logging.getLevelName,
    )

    # Считывание аргументов
    args = argparser.parse_args()

    # Настраиваем параметры логирования
    logging.basicConfig(format=constants.LOGGING_FORMAT, level=args.log_level)

    logging.info("Run with args: %s", args)

    # Генерируем данные
    X, y = make_classification(**params.DATASET_PARAMS)

    logging.info("Got X, y with shapes: %s, %s", X.shape, y.shape)

    # Подготавливаем выборки
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=params.TEST_SIZE,
        stratify=y,
        random_state=params.RANDOM_STATE,
    )

    logging.info("Got train, test with sizes: %s, %s", X_train.shape[0], X_test.shape[0])

    # Обучаем модель
    logging.info("Start training model")

    train_ds = catboost.Pool(X_train, label=y_train)
    model = catboost.CatBoostClassifier(**params.CATBOOST_MODEL_PARAMS)
    model.fit(train_ds)

    # Замеряем качество модели
    logging.info("Start eval model")

    y_pred = model.predict(X_test)

    metrics = {}
    for metric in params.EVAL_METRICS:
        metrics[metric] = eval_metric(y_test, y_pred, metric)

        logging.info("%s - %s", metric, metrics[metric])

    # Сохраняем артефакты
    model.save_model(args.output_model_path)

    logging.info("Model`s binary saved to %s", args.output_model_path)

    numpy.savez(args.output_features_path, X)

    logging.info("Features saved to %s", args.output_features_path)

    with open(args.output_metrics_path, "w") as file:
        json.dump(metrics, file)

    logging.info("Metrics saved to %s", args.output_metrics_path)
