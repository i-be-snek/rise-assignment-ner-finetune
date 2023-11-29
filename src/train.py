from typing import Any

from src.preprocess import Data


def check_gpus():
    import os

    from tensorflow import config

    gpus = config.list_physical_devices("GPU")
    if gpus:
        try:
            print(gpus)
            config.experimental.set_memory_growth(gpus[0], True)
        except RuntimeError as e:
            print(e)

    os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"


def train(
    optimizer: Any,
    tokenized_tf_dataset: Data,
    model_name: str = "distilroberta-base",
    verbose: int = 1,
    epochs: int = 3,
    tensorboard_callback: bool = False,
    model_checkpoint: str = "A",
) -> tf.keras.Model:
    from transformers import TFAutoModelForTokenClassification

    from src.metrics import Eval

    print("Dataset loaded and tokenized. Sample:")
    print(tokenized_tf_dataset.tokenized_dataset["train"][random.randint(0, 200)])

    model = TFAutoModelForTokenClassification.from_pretrained(model_name)
    model.compile(optimizer=optimizer)

    print(model.summary())
    print()

    metric_callback = KerasMetricCallback(
        metric_fn=Eval.compute_metrics,  # eval_dataset=tokenized_tf_dataset.validation_set
    )

    callbacks = [metric_callback]

    if tensorboard_callback:
        print("Adding tensorboard callback")
        tensorboard_callback = TensorBoard(log_dir="./tc_model_save/logs")
        callbacks.append(tensorboard_callback)

    print("Fitting model...")
    model.fit(
        tokenized_tf_dataset.train_set,
        validation_data=sys.validation_set,
        epochs=epochs,
        callbacks=callbacks,
        verbose=verbose,
    )

    print("Evaluating model")
    # todo: evaluate model

    print("Storing results")
    # todo: store results

    print("Saving the model...")
    model.save(f"model_{model_checkpoint}.keras")

    print("Inference sample:")
    # todo: print sample
