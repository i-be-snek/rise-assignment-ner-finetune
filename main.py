import logging
from random import randint

from huggingface_hub import login
from transformers import AdamWeightDecay
from transformers.utils import send_example_telemetry

from src.preprocess import PrepSystem
from src.tag import TagInfo
from src.train import check_gpus, get_token, train

logging.basicConfig(
    format="%(asctime)s %(levelname)-8s %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)

if __name__ == "__main__":
    # Login to HuggginFaceHub
    login(get_token())

    send_example_telemetry("finetuning fill-mask model on NER", framework="tensorflow")

    pretrained_model_checkpoint = "distilbert-base-uncased"
    experiment = "A"
    # experiment = "B"

    learning_rate = 2e-5
    optimizer = AdamWeightDecay(learning_rate=learning_rate, weight_decay_rate=0.0)

    if experiment == "A":
        labels = TagInfo.full_tagset
        filter_tagset = False
        dataset_batch_size = 8

    if experiment == "B":
        labels = TagInfo.main_five
        filter_tagset = True
        dataset_batch_size = 16

    experiment_name = f"exp_{experiment}"
    logging.info(f"Initializeding {experiment_name}")

    check_gpus()

    system = PrepSystem(
        labels=labels,
        pretrained_model_checkpoint=pretrained_model_checkpoint,
        dataset_batch_size=dataset_batch_size,
        filter_tagset=filter_tagset,
        language="en",
        split_filter=None,
        # None to get all data splits available (train, test, validation)
        # Otherwise, consult: https://huggingface.co/docs/datasets/v2.15.0/loading#slice-splits
        # The split filter arg in this function only supports splits like 'train' or 'validation[:100]'. but not using the plus ´+', example: 'test+train[:100]' )
    )

    system.get_model()

    system.load_tokenizer()

    system.tokenize_dataset()

    sample = system.tokenized_dataset["train"][randint(0, 200)]
    logging.info(f"Dataset loaded and tokenized.\nSample: {sample}")
    logging.info(f"Decoded: {system.tokenizer.convert_ids_to_tokens(sample['input_ids'])}")

    train(
        optimizer=optimizer,
        system=system,
        verbose=1,
        epochs=6,
        tensorboard_callback=True,
        push_to_hub_callback=True,
        early_stopping=True,
        early_stopping_patience=2,
        experiment_name=experiment_name,
    )
