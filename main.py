from huggingface_hub import login
from transformers import AdamWeightDecay
from transformers.utils import send_example_telemetry

from src.preprocess import Data
from src.tag import TagInfo
from src.train import check_gpus, get_token, train

if __name__ == "__main__":
    print("Logging into huggingface")
    login(get_token())

    send_example_telemetry("finetuning fill-mask model on NER", framework="tensorflow")

    check_gpus()

    optimizer = AdamWeightDecay(learning_rate=2e-5, weight_decay_rate=0.0)

    # Training A
    system_a = Data(
        labels=TagInfo.full_tagset,
        pretrained_model_checkpoint="distilroberta-base",
        dataset_batch_size=16,
        filter_tagset=False,
        language="en",
    )

    train(
        optimizer=optimizer,
        data_class_obj=system_a,
        model_name="distilroberta-base",
        verbose=1,
        epochs=10,
        tensorboard_callback=True,
        push_to_hub_callback=True,
        early_stopping=True,
        early_stopping_patience=3,
        model_checkpoint="exp_A",
    )

    # Training B

    # system_b = Data(labels=TagInfo.main_five,
    #                 pretrained_model_checkpoint="distilroberta-base",
    #                 dataset_batch_size=16,
    #                 filter_tagset=False,
    #                 language="en"
    #                 )

    # train(
    #     optimizer=optimizer,
    #     data_class_obj=system_b,
    #     model_name="distilroberta-base",
    #     verbose=1,
    #     epochs=10,
    #     tensorboard_callback=True,
    #     push_to_hub_callback=True,
    #     early_stopping=True,
    #     early_stopping_patience=3,
    #     model_checkpoint="B",
    # )
