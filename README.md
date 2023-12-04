## Finetuning a Fill-Masked model on Named Entity Recognition

This script fine-tunes a pre-trained fill-mask model (such as `distilbert-base-uncased`) on the NER token-classification task. It's specifically designed for [the multiNERD dataset](https://huggingface.co/datasets/Babelscape/multinerd) but any dataset with columns `tokens` (a list of string features), `ner_tags` (a list of classification labels (int)), and `lang` (a string feature).


### Summary


### Perquisites

#### Install dependencies
- **With `poetry` (recommended):**
    1. install [Poetry](https://python-poetry.org/docs/#installation)
    2. enter the poetry shell, which will create a venv and install all main and dev dependencies

    ```shell
    poetry shell # starts a venv
    poetry install --only main # installs all main deps from the lockfile
    ```

- **With `pip`**

    If you don't like `poetry`, you can also install the dependencies via pip

    ```shell
    python3
    pip3 install -r requirements.txt
    # TODO: add dev requirements too!
    ```

#### Logging in to Huggingface Hub

This script supports pushing models to the hugginface_hub. If you want to push model checkpoints and evaluations to the hub, grab a [user access token](https://huggingface.co/docs/hub/security-tokens) from huggingface and paste it into a dotenv (`.env`) file; example:

```shell
# .env in root dir
HF_TOKEN=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
```

### Quickstart

The `PrepSystem` class in `src.preprocess` handles the dataset preprcoessing, tokenization, and any additional transformations. This class allows experimenting with a limited tagset.

```python
from src.preprocess import PrepSystem
from src.train import train
from transformers import AdamWeightDecay

# choose a pretrained fill-mask model from hugginface hub
pretrained_model_checkpoint = "distilbert-base-uncased"

# choose an optimizer
optimizer = AdamWeightDecay(learning_rate=learning_rate, weight_decay_rate=0.0)

# choose a tagset (PER, ORG, LOC, DIS, ANIM)
tagset = {
        "O": 0,
        "B-PER": 1,
        "I-PER": 2,
        "B-ORG": 3,
        "I-ORG": 4,
        "B-LOC": 5,
        "I-LOC": 6,
        "B-ANIM": 7,
        "I-ANIM": 8,
        "B-DIS": 13,
        "I-DIS": 14,
    }

# if the tagset omits some tags from the multinerd dataset, it needs to be filtered out
# leave as False when using the full dataset
filter_tagset = True

# Initializing the PrepSystem dataset will load the dataset,  filter it by language,
# and swap omitted tags with the "O" label (if filter_tagset=True)

# When omitting some tags, the tag label ids need to be reset to be sequential numbers
# So all instances of "DIS" (which has the label IDs 13 and 14) will be reindexed and mapped to 9 and 10
system = PrepSystem(
        labels=tagset,
        pretrained_model_checkpoint=pretrained_model_checkpoint,
        dataset_batch_size=16,
        filter_tagset=filter_tagset,
        language="en",
        split_filter=None,  # None to load the entire dataset
    )

```

The `PrepSystem` object allows us to inspect the dataset. The dataset size is smaller since filtered by english, and the `lang` column has been removed.

```shell
> system.dataset

DatasetDict({
    train: Dataset({
        features: ['tokens', 'ner_tags'],
        num_rows: 262560
    })
    validation: Dataset({
        features: ['tokens', 'ner_tags'],
        num_rows: 32820
    })
    test: Dataset({
        features: ['tokens', 'ner_tags'],
        num_rows: 32908
    })
})
```

Now it's possible to load the model and tokenizer and preprocess and tokenize the dataset to ready it for training:

```python
# grab the pretrained model from hugginface
system.get_model()

# load the tokenizer from huggingface
system.load_tokenizer()

# tokenize the dataset
system.tokenize_dataset()

# inspect the model
print(system.model.summary())
```

Finally, the training:

```python
# choose a name for the experiment
# the output directory will contain this experiment name:
# {pretrained_model_checkpoint}-finetuned-ner-{experiment_name}"
experiment_name = "my_experiment"

# finally, train the model
train(
        optimizer=optimizer,
        system=system,
        verbose=1, # for training logs
        epochs=6,

        # False to disable
        tensorboard_callback=True,

        # True to push to hub (make sure you are logged in)
        # False to store results locally
        push_to_hub_callback=False,
        early_stopping=True, # False to disable
        early_stopping_patience=2,
        experiment_name=experiment_name,
    )
```

The `main.py` file contains an example that runs one experiment:

```shell
# with poetry:
poetry run python3 main.py

# with pip (activate your venv first)
python3 main.py
```

Training on a single GPU may take up to 20-30 minutes **per epoch**. It's possible to limit the size of the dataset by adding this code **before** tokenizing the dataset

```python
num_examples = 100
for ds in system.data_split:
    system.dataset[ds] = system.dataset[ds].select(range(num_examples))
```

### Evaluation
Follow the [`evaluation.ipynb`](evaluation.ipynb) notebook for evaluating the model using Huggingface's `evaluate` library

### HuggingFace
Two experiments can be found on the huggingface hub:

- [Experiment A](https://huggingface.co/i-be-snek/distilbert-base-uncased-finetuned-ner-exp_A)
- [Experiment B](https://huggingface.co/i-be-snek/distilbert-base-uncased-finetuned-ner-exp_B)
