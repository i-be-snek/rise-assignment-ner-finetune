from src.train import check_gpus, train
from transformers import AdamWeightDecay
from src.tag import TagInfo
from src.preprocess import Data

optimizer = AdamWeightDecay(lr=2e-4, weight_decay_rate=0.3)

if __name__ == "__main__":
    check_gpus()

    system_a = Data(reduced_tagset=())
    
    train(optimizer=optimizer,
        tokenized_tf_dataset=system_a,
        model_name="distilroberta-base", 
        verbose=1, 
        epochs=1)

    system_b = Data(reduced_tagset=list(TagInfo.reduced_tagset.keys()))

    train(optimizer=optimizer,
        tokenized_tf_dataset=system_b,
        model_name="distilroberta-base", 
        verbose=1, 
        epochs=1)
