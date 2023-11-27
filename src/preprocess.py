from datasets import load_dataset
from tag import TagInfo
from dataclasses import dataclass

dataset = load_dataset("Babelscape/multinerd")


@dataclass
class Data:
    data_split = ["test", "validation", "train"]

    @staticmethod
    def filter_out_tags(example, tags_to_keep: list) -> dict:
        ner_tags: list = example["ner_tags"]
        example["ner_tags"] = [
            0 if tag not in tags_to_keep else tag for tag in ner_tags
        ]
        return example

    def __post_init__(self):
        self.dataset = load_dataset("Babelscape/multinerd")

        # filter out non-English examples
        for ds in self.data_split:
            self.dataset[ds] = self.dataset[ds].filter(lambda x: x["lang"] == "en")

        print(self.dataset["train"][0])

        for ds in self.data_split:
            self.dataset[ds] = self.dataset[ds].remove_columns("lang")

        # initiate datasets for each system
        self.dataset_a = self.dataset.copy()
        self.dataset_b = self.dataset.copy()

        # limit the tags to a reduced tagset, tags not included in the list will be replaced by a 0 tag
        for ds in self.data_split:
            self.dataset_b[ds] = self.dataset_b[ds].map(
                self.filter_out_tags,
                fn_kwargs={"tags_to_keep": TagInfo.reduced_tagset},
                num_proc=4,
            )
