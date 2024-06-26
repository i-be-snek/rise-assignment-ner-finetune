## Finetuning a Fill-Masked model on Named Entity Recognition

This script fine-tunes a pre-trained fill-mask model (such as `distilbert-base-uncased`) on the NER token-classification task. It's specifically designed for [the multiNERD dataset](https://huggingface.co/datasets/Babelscape/multinerd) but any dataset with columns `tokens` (a list of string features), `ner_tags` (a list of classification labels (int)), and `lang` (a string feature) should work.

### HuggingFace
[![Follow me on HF](https://huggingface.co/datasets/huggingface/badges/resolve/main/follow-me-on-HF-md-dark.svg)](https://huggingface.co/i-be-snek)

You can test the two models and view their performance metrics on HuggingFace hub:

- [Experiment A](https://huggingface.co/i-be-snek/distilbert-base-uncased-finetuned-ner-exp_A)
- [Experiment B](https://huggingface.co/i-be-snek/distilbert-base-uncased-finetuned-ner-exp_B)


### Summary

The experiments finetune an uncased fill-mask model `distillery-base-uncased`. Distilled models are smaller and faster to finetune with a slight drop in performance. The two finetuned models (A and B) were evaluated on the English subset of the [multiNERD](https://huggingface.co/datasets/Babelscape/multinerd) test set. In model A, `CEL`, `FOOD`, `INST`, `MYTH`, and `PLANT` performed worse on F1. Overall, model B, finetuned on a smaller tagset, outperforms model A on all five tags (a likely outcome). The results for model A are similar to those reported in the paper (which was finetuned on the [cased mBERT](https://huggingface.co/bert-base-multilingual-cased)), except for `CEL` and `MYTH` where model A performs worse.

Limitations:
- Categories with examples of [binomial nomenclature](https://en.wikipedia.org/wiki/Binomial_nomenclature) (like `PLANT`) might yield better performance if finetuned on a cased BERT instead.

- `BIO`, `INST`, `MYTH`, and `VEHI` have far fewer examples, which could be overcome by oversampling minority classes (maybe with something like [SMOTE](https://github.com/analyticalmindsltd/smote_variants#smote)).

- Finetuning the fill-mask model on domain-specific texts (with sentences featuring smaller classes like `MYTH`) before training a task-specific head for token classification could improve performance since the tokenizer won't treat these now-seen words as rare tokens.

----
[![Lint](https://github.com/i-be-snek/rise-assignment-ner-finetune/actions/workflows/lint.yaml/badge.svg)](https://github.com/i-be-snek/rise-assignment-ner-finetune/actions/workflows/lint.yaml)
<a href="https://gitmoji.dev">
  <img
    src="https://img.shields.io/badge/gitmoji-%20😜%20😍-FFDD67.svg?style=flat-square"
    alt="Gitmoji"
  />
</a>

### Prerequisites

#### Install dependencies
- **With [`poetry`](https://python-poetry.org/docs/#installation) (recommended):**

    ```shell
    poetry shell # activates a venv
    poetry install --only main # installs all main deps from the lockfile
    ```

- **With `pip`**

    If you don't like `poetry`, you can also install the dependencies via pip

    ```shell
    python3 -m venv my-venv
    source my-venv/bin/activate

    pip3 install -r requirements.txt
    ```

#### Logging in to Huggingface Hub

This script supports pushing models to the hugginface_hub. If you want to push model checkpoints and evaluations to the hub, grab a [user access token](https://huggingface.co/docs/hub/security-tokens) from huggingface and paste it into a dotenv (`.env`) file; example:

```shell
# .env in root dir
HF_TOKEN=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
```

#### Linting (optional)
Install [`pre-commit`](https://pre-commit.com/) for linting and formatting (with a hook that runs prior to making commits)

```shell
# with pip inside a venv
pip3 install pre-commit

# with poetry
poetry install --only dev
```

### Quickstart

The `PrepSystem` class in `src.preprocess` handles the dataset preprcoessing, tokenization, and any additional transformations needed for finetuning and evaluation. This class allows experimenting with a limited tagset.

You can follow the example below in the [`train_example.ipynb`](train_example.ipynb) notebook.

```python
from src.preprocess import PrepSystem
from src.train import train
from transformers import AdamWeightDecay

# choose a pretrained fill-mask model from hugginface hub
pretrained_model_checkpoint = "distilbert-base-uncased"

# choose an optimizer
learning_rate = 2e-5
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
# leave `filter_tagset` as False when using the full dataset
filter_tagset = True

# Initializing the PrepSystem dataset will load the dataset,  filter it by language,
# and swap omitted tags with the "O" label (if filter_tagset=True)

# When omitting some tags, the tag label ids need to be reset to be sequential numbers
# So all instances of "DIS" (which has the label IDs 13 and 14) will be reindexed and mapped to 9 and 10
system = PrepSystem(
        labels=tagset,
        pretrained_model_checkpoint=pretrained_model_checkpoint,
        # a smaller dataset batch size is recommended in case of OOM errors
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

# load the tokenizer from huggingface with AutoTokenizer
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

        # False to disable
        early_stopping=True,
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

> [!TIP]
> Training on a single GPU may take up to 20-30 minutes **per epoch**. It's possible to limit the size of the dataset by adding this code **before** tokenizing the dataset

```python
num_examples = 100
for ds in system.data_split:
    system.dataset[ds] = system.dataset[ds].select(range(num_examples))
```

### Evaluation
Follow the [`evaluation.ipynb`](evaluation.ipynb) notebook for evaluating the model using Huggingface's `evaluate` library
