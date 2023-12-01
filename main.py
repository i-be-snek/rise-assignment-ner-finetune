import logging
from random import randint

from huggingface_hub import login
from transformers import AdamWeightDecay
from transformers.utils import send_example_telemetry

from src.preprocess import Data
from src.tag import TagInfo
from src.train import check_gpus, get_token, train

logging.basicConfig(
    format="%(asctime)s %(levelname)-8s %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)

if __name__ == "__main__":
    login(get_token())

    send_example_telemetry("finetuning fill-mask model on NER", framework="tensorflow")

    check_gpus()

    optimizer = AdamWeightDecay(learning_rate=2e-5, weight_decay_rate=0.0)

    pretrained_model_checkpoint = "distilbert-base-uncased"
    # "distilroberta-base"

    # Training A
    system_a = Data(
        labels=TagInfo.full_tagset,
        pretrained_model_checkpoint=pretrained_model_checkpoint,
        dataset_batch_size=16,
        filter_tagset=False,
        language="en",
    )
    sample = system_a.tokenized_dataset["train"][randint(0, 200)]
    logging.info(f"Dataset loaded and tokenized.\nSample: {sample}")
    logging.info(f"Decoded: {system_a.tokenizer.convert_ids_to_tokens(sample['input_ids'])}")

    train(
        optimizer=optimizer,
        data_class_obj=system_a,
        verbose=1,
        epochs=6,
        tensorboard_callback=True,
        push_to_hub_callback=False,
        early_stopping=True,
        early_stopping_patience=2,
        experiment_name="exp_A",
    )

    # Training B

    # system_b = Data(labels=TagInfo.main_five,
    #                 pretrained_model_checkpoint=pretrained_model_checkpoint,
    #                 dataset_batch_size=16,
    #                 filter_tagset=False,
    #                 language="en"
    #                 )

    # train(
    #     optimizer=optimizer,
    #     data_class_obj=system_b,
    #     verbose=1,
    #     epochs=10,
    #     tensorboard_callback=True,
    #     push_to_hub_callback=True,
    #     early_stopping=True,
    #     early_stopping_patience=3,
    #     experiment_name="B",
    # )
