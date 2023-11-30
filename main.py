from src.train import check_gpus, train, get_token
from transformers import AdamWeightDecay
from src.tag import TagInfo
from src.preprocess import Data
from huggingface_hub import login
from transformers.utils import send_example_telemetry


if __name__ == "__main__":
    print("Logging into huggingface")
    login(get_token())

    send_example_telemetry("finetuning fill-mask model on NER", framework="tensorflow")

    check_gpus()
    
    optimizer = AdamWeightDecay(learning_rate=2e-5, weight_decay_rate=0.0)

    system_a = Data(labels=TagInfo.full_tagset,
                    pretrained_model_checkpoint="distilroberta-base",
                    dataset_batch_size=16,
                    filter_tagset=False,
                    language="en"
                    )

    train(
        optimizer=optimizer,
        data_class_obj=system_a,
        model_name="distilroberta-base",
        verbose=1,
        epochs=6,
        tensorboard_callback=True,
        push_to_hub_callback=True,
        early_stopping=False,
        #early_stopping_patience=4,
        model_checkpoint="A",
    )