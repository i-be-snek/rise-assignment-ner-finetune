import json
from dataclasses import dataclass


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

        results = self.metric.compute(
            predictions=true_predictions, references=true_labels
        )

        # print results for individual entity types
        for i in results.keys():
            if not i.startswith("overall"):
                print()
                print(i)
                for res in results[i]:
                    if res != "number":
                        print("{}\t{}".format(res, results[i][res]))

        with open(self.metrics_results_filename, "a") as f:
            f.write(json.dumps(results, default=str) + "\n")

        return {
            "precision": results["overall_precision"],
            "recall": results["overall_recall"],
            "f1": results["overall_f1"],
            "accuracy": results["overall_accuracy"],
        }
