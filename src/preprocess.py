from dataclasses import dataclass, field
from typing import Dict

from datasets import DatasetDict, load_dataset
from transformers import (AutoTokenizer, BertTokenizerFast,
                          DataCollatorForTokenClassification,
                          PreTrainedTokenizerFast,
                          TFAutoModelForTokenClassification)

from src.tag import TagInfo


@dataclass
class Data:
    """
    A dataclass to to preprocess a dataset. Labels will be tokenized and aligned.

    Args:
        labels (dict[str, int]): a dictionary of named entity tags to labels. Example: {"O": 0, "B-PER": 1}
        pretrained_model_checkpoint (str): a pretrained model checkpoint from HuggingFace; this will be loaded as a `TFAutoModelForTokenClassification`
        dataset_batch_size (int): tensorflow dataset batch size; select a smaller batch size if encountering OOM problems with GPU training
        filter_tagset (bool): whether or not to re-tag examples not found in the labels dict as "0"
        language (str): a two-letter language code to the dataset by; pass an empty string "" to disable

    Raises:
        AssertionError: if the language selected for filtering isn't available

    """

    labels: Dict[str, int] = field(default_factory=lambda: {TagInfo.full_tagset})
    pretrained_model_checkpoint: str = "distilroberta-base"
    dataset_batch_size: int = 8
    filter_tagset: bool = False
    language: str = "en"

    def __post_init__(self) -> None:
        self.label_names: list(str) = list(self.labels.keys())
        self.label_values: list(int) = list(self.labels.values())

        self.id2label = self.labels
        self.label2id = dict(self.id2label.items())

        self.model = TFAutoModelForTokenClassification.from_pretrained(
            self.pretrained_model_checkpoint,
            num_labels=len(self.label_names),
            id2label=self.id2label,
            label2id=self.label2id,
        )

        self.dataset: DatasetDict = load_dataset("Babelscape/multinerd")
        self.dataset_languages = self.dataset.unique("lang")["train"]
        self.data_split: tuple = tuple(self.dataset.keys())

        if self.language not in self.dataset_languages:
            raise AssertionError(
                f"The selected language {self.language} is not available in the dataset. Available languages: {', '.join(self.dataset_languages)}"
            )
        # Filter by language
        if self.language:
            for ds in self.data_split:
                self.dataset[ds] = self.dataset[ds].filter(
                    lambda x: x["lang"] == self.language
                )
                self.dataset[ds] = self.dataset[ds].remove_columns("lang")
            print(self.dataset)

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.pretrained_model_checkpoint, add_prefix_space=True
        )
        assert isinstance(self.tokenizer, PreTrainedTokenizerFast), print(
            "Not a fast tokenizer!"
        )

        self.data_collator = DataCollatorForTokenClassification(
            self.tokenizer,
            return_tensors="np",
        )

        # Filter out extra tags if training with a smaller tagset
        if self.filter_tagset:
            print(
                f"Keeping these tags only: {str(self.label_names)}. All other tags will be set to '0'"
            )
            for ds in self.data_split:
                self.dataset[ds] = self.dataset[ds].map(
                    self.filter_out_tags,
                    fn_kwargs={"tags_to_keep": self.label_values},
                    num_proc=4,
                )

        else:
            print("Using the full tagset")

        # Tokenize the dataset
        self.tokenized_dataset = self.dataset.map(
            self.tokenize_and_align_labels,
            fn_kwargs={"tokenizer": self.tokenizer},
            batched=True,
        )

        self.train_set = self.model.prepare_tf_dataset(
            self.tokenized_dataset["train"],
            shuffle=True,
            batch_size=self.dataset_batch_size,
            collate_fn=self.data_collator,
        )

        self.validation_set = self.model.prepare_tf_dataset(
            self.tokenized_dataset["validation"],
            shuffle=False,
            batch_size=self.dataset_batch_size,
            collate_fn=self.data_collator,
        )

        self.test_set = self.model.prepare_tf_dataset(
            self.tokenized_dataset["test"],
            shuffle=False,
            batch_size=self.dataset_batch_size,
            collate_fn=self.data_collator,
        )

    @staticmethod
    def filter_out_tags(example: dict, tags_to_keep: list) -> dict:
        ner_tags: list = example["ner_tags"]
        example["ner_tags"] = [
            0 if tag not in tags_to_keep else tag for tag in ner_tags
        ]
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
