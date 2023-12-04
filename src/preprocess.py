import logging
from dataclasses import dataclass, field
from typing import Dict, Union

from datasets import ClassLabel, DatasetDict, Sequence, load_dataset
from transformers import (AutoTokenizer, BertTokenizerFast,
                          DataCollatorForTokenClassification,
                          PreTrainedTokenizerFast,
                          TFAutoModelForTokenClassification)

logging.basicConfig(
    format="%(asctime)s %(levelname)-8s %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)


@dataclass
class PrepSystem:
    """
    A dataclass to to preprocess a dataset and loads the data and model. Labels will be tokenized and aligned.
    The class will also load the model, dataset, and tokenizer. The tokenizer must be of type PreTrainedTokenizerFast.
    The model mus be a TF/keras-compatible MLM model to be finetuned on Token Classification.
    Examples: https://huggingface.co/models?pipeline_tag=fill-mask&library=tf&sort=trending

    Args:
        labels (dict[str, int]): a dictionary of named entity tags to labels. Example: {"O": 0, "B-PER": 1}
        pretrained_model_checkpoint (str): a pretrained model checkpoint from HuggingFace; this will be loaded as a `TFAutoModelForTokenClassification`
        dataset_batch_size (int): tensorflow dataset batch size; select a smaller batch size if encountering OOM problems with GPU training
        filter_tagset (bool): whether or not to re-tag examples not found in the labels dict as "0"
        language (str): a two-letter language code to the dataset by; pass an empty string "" to disable
        huggingface_dataset_name (str): the name of the dataset to load from hugginface
        split_filter (str | None): Pass None to get all data splits available (train, test, validation). Otherwise, consult: https://huggingface.co/docs/datasets/v2.15.0/loading#slice-splits. The split filter arg in this function only supports splits like 'train' or 'validation[:100]'. but not using the plus ´+', example: 'test+train[:100]'.

    Raises:
        AssertionError: if the language selected for filtering isn't available

    """

    # labels: Dict[str, int] = field(default_factory=lambda: {TagInfo.full_tagset})
    labels: Dict[str, int] = field(default=dict)
    pretrained_model_checkpoint: str = "distilbert-base-uncased"
    dataset_batch_size: int = 8
    filter_tagset: bool = False
    language: str = "en"
    huggingface_dataset_name: str = "Babelscape/multinerd"
    split_filter: Union[str, None] = None

    def __post_init__(self) -> None:
        # self.swapped_labels: Union[dict, bool] = None

        self.label_names: list(str) = list(self.labels.keys())
        self.label_values: list(int) = list(self.labels.values())

        # Load dataset, must have columns: tokens (list[str]), ner_tags (list[int]), and lang (str)
        # https://huggingface.co/datasets/Babelscape/multinerd
        self.dataset: DatasetDict = load_dataset(self.huggingface_dataset_name, split=self.split_filter)
        if not isinstance(self.dataset, DatasetDict):
            self.dataset = DatasetDict({self.split_filter.split("[")[0]: self.dataset})

        # Filter by language
        self.data_split: tuple = tuple(self.dataset.keys())
        self.dataset_languages = self.dataset.unique("lang")[self.data_split[0]]
        if self.language not in self.dataset_languages:
            raise AssertionError(
                f"The selected language {self.language} is not available in the dataset. Available languages: {', '.join(self.dataset_languages)}"
            )
        if self.language:
            for ds in self.data_split:
                self.dataset[ds] = self.dataset[ds].filter(lambda x: x["lang"] == self.language)
                self.dataset[ds] = self.dataset[ds].remove_columns("lang")
            logging.info(f"Filtered language by {self.language}. \n{self.dataset}")

        # Filter out extra tags if training with a smaller tagset
        if self.filter_tagset:
            logging.info(f"Keeping these tags only: {str(self.label_names)}. All other tags will be set to '0'")
            for ds in self.data_split:
                self.dataset[ds] = self.dataset[ds].map(
                    self.filter_out_tags,
                    fn_kwargs={"tags_to_keep": self.label_values},
                    num_proc=4,
                )

        else:
            logging.info("Using the full tagset")

        logging.info("Making sure all labels have sequential IDs. This can happen if a reduced tagset is chosen")

        # Create id2label
        self.label2id = self.labels
        self.id2label = {v: k for k, v in self.labels.items()}
        self.id2label = dict(sorted(self.id2label.items()))

        # If the token IDs are not sequential, there will be issues computing loss (nan)
        # Swap non-sequential labels
        labels_to_swap = self.get_labels_to_swap(self.id2label)

        # if there are any labels to swap
        if labels_to_swap:
            logging.info(f"Swapping these labels: {labels_to_swap}")
            self.label2id, self.id2label = self.swap_labels_in_config(self.label2id, labels_to_swap=labels_to_swap)

            logging.info(f"Modified label to ID: {self.label2id}")
            logging.info(f"Modified ID to label: {self.id2label}")

            for ds in self.data_split:
                self.dataset[ds] = self.dataset[ds].map(
                    self.swap_labels_in_dataset,
                    fn_kwargs={"labels_to_swap": labels_to_swap},
                    num_proc=4,
                )

            self.swapped_labels: Union[dict, bool] = labels_to_swap

        elif not labels_to_swap:
            logging.info(f"All label ids are sequential, nothing to swap.")

        logging.info(
            "Adding Sequence(ClassLabel) feature to dataset to make it usable with the `TokenClassificationEvaluator` from `evaluation`."
            "\nRead more: https://huggingface.co/docs/evaluate/v0.4.0/en/package_reference/evaluator_classes"
        )
        class_labels = list(self.id2label.values())
        for ds in self.data_split:
            features = self.dataset[ds].features.copy()
            features["ner_tags"] = Sequence(feature=ClassLabel(names=class_labels))
            self.dataset[ds] = self.dataset[ds].map(features=features)

    def get_model(self):
        # Load model with X labels
        self.model = TFAutoModelForTokenClassification.from_pretrained(
            self.pretrained_model_checkpoint,
            num_labels=len(self.id2label.keys()),
            id2label=self.id2label,
            label2id=self.label2id,
        )

    def load_tokenizer(self):
        # Load Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.pretrained_model_checkpoint,
        )

        try:
            isinstance(self.tokenizer, PreTrainedTokenizerFast)
        except AssertionError as err:
            logging.error(f"Not a fast tokenizer!")
            raise err

        # https://huggingface.co/docs/transformers/main_classes/data_collator
        # applies random masking
        self.data_collator = DataCollatorForTokenClassification(
            self.tokenizer,
            return_tensors="np",
        )

    def tokenize_dataset(self):
        # Tokenize the dataset
        self.tokenized_dataset = self.dataset.map(
            self.tokenize_and_align_labels,
            fn_kwargs={"tokenizer": self.tokenizer},
            batched=True,
        )

        if "train" in self.data_split:
            self.train_set = self.model.prepare_tf_dataset(
                self.tokenized_dataset["train"],
                shuffle=True,
                batch_size=self.dataset_batch_size,
                collate_fn=self.data_collator,
            )
        if "val" in self.data_split or "validation" in self.data_split:
            self.validation_set = self.model.prepare_tf_dataset(
                self.tokenized_dataset["validation"],
                shuffle=False,
                batch_size=self.dataset_batch_size,
                collate_fn=self.data_collator,
            )
        if "test" in self.data_split:
            self.test_set = self.model.prepare_tf_dataset(
                self.tokenized_dataset["test"],
                shuffle=False,
                batch_size=self.dataset_batch_size,
                collate_fn=self.data_collator,
            )

    @staticmethod
    def swap_labels_in_config(label2id: dict[int, str], labels_to_swap: dict[int, int]):
        new_label2id = {}
        for _label, _id in label2id.items():
            if _id in labels_to_swap:
                new_label2id[_label] = labels_to_swap[_id]
            else:
                new_label2id[_label] = _id

        id2label = {v: k for k, v in new_label2id.items()}

        return new_label2id, id2label

    @staticmethod
    def swap_labels_in_dataset(example: dict, labels_to_swap: dict[int, int]):
        ner_tags = example["ner_tags"]
        # if there are only 0 label IDs (likely when removing tags
        # from the datatset) then return the example straight away
        if all(tag == 0 for tag in ner_tags):
            return example

        modified_ner_tags = []
        to_swap = list(labels_to_swap.keys())

        for i in ner_tags:
            if i in to_swap:
                label_id = labels_to_swap[i]
            else:
                label_id = i
            modified_ner_tags.append(label_id)

        example["ner_tags"] = modified_ner_tags
        return example

    @staticmethod
    def get_labels_to_swap(label_dict: dict[int, str]) -> dict:
        # begin index at 0
        expected_key = 0
        ids_to_relabel = {}  # original -> new

        for key, value in label_dict.items():
            if key != expected_key:
                ids_to_relabel[key] = expected_key
            expected_key += 1

        return dict(sorted(ids_to_relabel.items(), key=lambda x: x[0], reverse=True))

    @staticmethod
    def filter_out_tags(example: dict, tags_to_keep: list) -> dict:
        ner_tags: list = example["ner_tags"]
        example["ner_tags"] = [0 if tag not in tags_to_keep else tag for tag in ner_tags]
        return example

    @staticmethod
    def tokenize_and_align_labels(
        examples: dict,
        tokenizer: BertTokenizerFast,
        label_all_tokens: bool = True,
    ):
        """
        Ths function tokenizes sentences and aligns their labels. The label mismatch is caused by WordPiece tokenizers
        that adds special tokesn (like '[CLS]' and '[SEP]') and uses '#' signs to split words into subwords.
        When this happens, the length and labels of the sentence won't match the tokenized output.model
        Special tokens with a `word_id` of `None` will be set to -100 to be automatically ignored by the loss function

        This function tokenizes and aligns the examples. It's borrowed directly from a hugginface
        finetuning example on a token-classification model.
        Link: https://github.com/huggingface/notebooks/blob/main/examples/token_classification-tf.ipynb


        Args:
            examples (dict): a dictionary where the keys `tokens` and `ner_tags` each contain a list of equal length
            label_all_tokens (bool, optional): If set to True, token labels inherit the same label of their original word.
                If set to false, these sub-words are given a `-100` label instead.
                Defaults to True.

        Returns:
            tokenized_inputs (dict): A dictionary with tokenized inputs to feed into the model
        """
        tokenized_inputs = tokenizer(
            examples["tokens"],
            truncation=True,
            is_split_into_words=True,
        )

        labels = []
        for i, label in enumerate(examples["ner_tags"]):
            word_ids = tokenized_inputs.word_ids(batch_index=i)
            previous_word_idx = None
            label_ids = []
            for word_idx in word_ids:
                # Special tokens have a word id that is None. We set the label to -100 so they are automatically
                # ignored in the loss function.
                if word_idx is None:
                    label_ids.append(-100)
                # We set the label for the first token of each word.
                elif word_idx != previous_word_idx:
                    label_ids.append(label[word_idx])
                # For the other tokens in a word, we set the label to either the current label or -100, depending on
                # the label_all_tokens flag.
                else:
                    label_ids.append(label[word_idx] if label_all_tokens else -100)
                previous_word_idx = word_idx

            labels.append(label_ids)

        tokenized_inputs["labels"] = labels
        return tokenized_inputs
