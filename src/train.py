import logging
from typing import Any
import os
from src.preprocess import PrepSystem

logging.basicConfig(
    format="%(asctime)s %(levelname)-8s %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)


def check_gpus():
    from os import environ

    from tensorflow import config

    gpus = config.list_physical_devices("GPU")
    if gpus:
        try:
            logging.info(gpus)
            config.experimental.set_memory_growth(gpus[0], True)
        except RuntimeError as err:
            logging.error(err)

    environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"


def get_token():
    from dotenv import dotenv_values

    token = dotenv_values(".env")["HF_TOKEN"]
    return token


def train(
    optimizer: Any,
    system: PrepSystem,
    verbose: int = 1,
    epochs: int = 3,
    tensorboard_callback: bool = False,
    push_to_hub_callback: bool = False,
    early_stopping: bool = False,
    early_stopping_patience: int = 3,
    experiment_name: str = "A",
) -> None:
    from transformers.keras_callbacks import KerasMetricCallback

    from src.metrics import Eval

    output_path = f"{system.pretrained_model_checkpoint}-finetuned-ner-{experiment_name}"
    
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    system.model.compile(optimizer=optimizer)

    logging.info(system.model.summary())

    # Initiate Eval class
    eval_metrics = Eval(
        metrics_results_filename=f"{output_path}/metrics_by_entity.jsonl",
        metric_name="seqeval",
        label_list=list(system.label2id.keys()),
    )

    # Set up metric callbacks
    callbacks = []
    metric_callback = KerasMetricCallback(
        metric_fn=eval_metrics.compute_metrics,
        eval_dataset=system.validation_set,
    )
    callbacks.append(metric_callback)

    if early_stopping:
        from tensorflow.keras.callbacks import EarlyStopping

        early_stopping_callback = EarlyStopping(
            monitor="val_loss",
            restore_best_weights=True,
            patience=early_stopping_patience,
        )
        callbacks.append(early_stopping_callback)
        logging.info(f"Early stopping with patience of {early_stopping_patience}")

    if tensorboard_callback:
        from tensorflow.keras.callbacks import TensorBoard

        logging.info("Adding tensorboard callback")
        tensorboard_callback = TensorBoard(log_dir=f"./{output_path}/logs")
        callbacks.append(tensorboard_callback)

    if push_to_hub_callback:
        logging.info("Pushing checkpoints to HF hub")
        from transformers.keras_callbacks import PushToHubCallback

        push_to_hub_callback = PushToHubCallback(
            output_dir=f"./{output_path}",
            tokenizer=system.tokenizer,
            hub_model_id=f"{system.pretrained_model_checkpoint}-finetuned-ner-{experiment_name}",
        )
        callbacks.append(push_to_hub_callback)

    elif not push_to_hub_callback:
        logging.info("Storing model checkpoint at each epoch")
        from tensorflow.keras.callbacks import ModelCheckpoint

        save_checkpoint = ModelCheckpoint(
            f"./{output_path}/checkpoint_" + "{epoch:02d}.model.h5",
            save_freq="epoch",
            verbose=1,
            save_best_only=False,
            save_weights_only=True,
        )
        callbacks.append(save_checkpoint)

    logging.info("Fitting model...")

    system.model.fit(
        system.train_set,
        validation_data=system.validation_set,
        epochs=epochs,
        callbacks=callbacks,
        verbose=verbose,
    )

    if not push_to_hub_callback:
        system.model.save_pretrained(f"./{output_path}/model")
