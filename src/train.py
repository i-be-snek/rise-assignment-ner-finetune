from typing import Any

from huggingface_hub import login

from src.preprocess import Data


def check_gpus():
    from os import environ

    from tensorflow import config

    gpus = config.list_physical_devices("GPU")
    if gpus:
        try:
            print(gpus)
            config.experimental.set_memory_growth(gpus[0], True)
        except RuntimeError as e:
            print(e)

    environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"


def get_token():
    from dotenv import dotenv_values

    token = dotenv_values(".env")["HF_TOKEN"]
    return token


def train(
    optimizer: Any,
    data_class_obj: Data,
    model_name: str = "distilroberta-base",
    verbose: int = 1,
    epochs: int = 3,
    tensorboard_callback: bool = False,
    push_to_hub_callback: bool = False,
    early_stopping: bool = False,
    early_stopping_patience: int = 3,
    model_checkpoint: str = "A",
) -> None:
    from random import randint

    from transformers import TFAutoModelForTokenClassification
    from transformers.keras_callbacks import KerasMetricCallback

    from src.metrics import Eval

    callbacks = []

    output_path = f"tc_model_save_{model_checkpoint}"

    if early_stopping:
        from tensorflow.keras.callbacks import EarlyStopping

        early_stopping_callback = EarlyStopping(
            monitor="loss",
            restore_best_weights=True,
            patience=early_stopping_patience,
        )
        callbacks.append(early_stopping_callback)
        print("Early stopping with patience of", early_stopping_patience)

    print("Dataset loaded and tokenized. Sample:")
    print(data_class_obj.tokenized_dataset["train"][randint(0, 200)])

    data_class_obj.model.compile(optimizer=optimizer)

    print(data_class_obj.model.summary())
    print()

    eval_metrics = Eval(
        metrics_results_filename=f"{output_path}/metrics_by_entity.txt",
        metric_name="seqeval",
        label_list=list(data_class_obj.label2id.keys()),
    )

    metric_callback = KerasMetricCallback(
        metric_fn=eval_metrics.compute_metrics,
        eval_dataset=data_class_obj.validation_set,
    )
    callbacks.append(metric_callback)

    if tensorboard_callback:
        from tensorflow.keras.callbacks import TensorBoard

        print("Adding tensorboard callback")
        tensorboard_callback = TensorBoard(log_dir=f"./{output_path}/logs")
        callbacks.append(tensorboard_callback)

    if push_to_hub_callback:
        print("Pushing checkpoints to HF hub")
        from transformers.keras_callbacks import PushToHubCallback

        push_to_hub_callback = PushToHubCallback(
            output_dir="./{output_path}",
            tokenizer=data_class_obj.tokenizer,
            hub_model_id=f"{model_name}-finetuned-ner-{model_checkpoint}",
        )
        callbacks.append(push_to_hub_callback)

    print("Fitting model...")
    data_class_obj.model.fit(
        data_class_obj.train_set,
        validation_data=data_class_obj.validation_set,
        epochs=epochs,
        callbacks=callbacks,
        verbose=verbose,
    )
