from dataclasses import dataclass


class Eval:
    metric: str = "seqval"
    label_list: list

    @staticmethod
    def compute_metrics(p):
        """
        This function is taken from a transformer notebook example.
        https://github.com/huggingface/notebooks/blob/main/examples/token_classification-tf.ipynb

        Choosing the seqval metric allows us to get precision, recall, and f1,
        making the results comparable to the scores in the paper https://aclanthology.org/2022.findings-naacl.60.pdf

        Args:
            p (tuple): predictions and labels

        Returns:
            dict: returns the evaluation results
        """
        from evaluate import load
        from numpy import argmax

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
        return {
            "precision": results["overall_precision"],
            "recall": results["overall_recall"],
            "f1": results["overall_f1"],
            "accuracy": results["overall_accuracy"],
        }