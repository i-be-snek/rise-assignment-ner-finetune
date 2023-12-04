import json
import logging
from dataclasses import dataclass
import os

logging.basicConfig(
    format="%(asctime)s %(levelname)-8s %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)


@dataclass
class Eval:
    label_list: list
    metrics_results_filename: str
    metric_name: str = "seqeval"

    def compute_metrics(self, p):
        """
        This function is taken from a transformer notebook example.
        https://github.com/huggingface/notebooks/blob/main/examples/token_classification-tf.ipynb

        Choosing the seqeval metric allows us to get precision, recall, and f1,
        making the results comparable to the scores in the paper https://aclanthology.org/2022.findings-naacl.60.pdf

        Args:
            p (tuple): predictions and labels

        Returns:
            dict: returns the evaluation results
        """
        from evaluate import load
        from numpy import argmax

        self.metric = load(self.metric_name)

        predictions, labels = p
        predictions = argmax(predictions, axis=2)

        # Remove ignored index (special tokens)
        true_predictions = [
            [self.label_list[p] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        true_labels = [
            [self.label_list[l] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]

        results = self.metric.compute(predictions=true_predictions, references=true_labels)

        # show results for individual entity types
        for i in results.keys():
            if not i.startswith("overall"):
                logging.info("")
                logging.info(f"Entity: {i}")
                for res in results[i]:
                    if res != "number":
                        logging.info(f"{res} \t {results[i][res]}")

        # create file if it doesn't exist
        if not os.path.exists(self.metrics_results_filename):
            open(self.metrics_results_filename, 'w').close()

        # append metrics per tag to the metrics file
        with open(self.metrics_results_filename, "a") as f:
            f.write(json.dumps(results, default=str) + "\n")

        return {
            "precision": results["overall_precision"],
            "recall": results["overall_recall"],
            "f1": results["overall_f1"],
            "accuracy": results["overall_accuracy"],
        }
