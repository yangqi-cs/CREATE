
class HierarchicalMetric:
    def __init__(self, y_true, y_pred,
                 f_beta: float = 1.0):
        # , y_true: List[str], y_pred: List[str]
        self.y_true = y_true
        self.y_pred = y_pred
        self.f_beta = f_beta

        assert len(y_true) == len(y_pred)

        self.correct_class = 0
        self.true_class_len = 0
        self.predict_class_len = 0
        for y_t, y_p in zip(y_true, y_pred):
            self.correct_class += len(set(y_t).intersection(set(y_p)))
            self.true_class_len += len(y_t)
            self.predict_class_len += len(y_p)

        # Compute precision score for hierarchical classification
        self.hierarchical_precision = self.correct_class / self.predict_class_len
        # Compute recall score for hierarchical classification
        self.hierarchical_recall = self.correct_class / self.true_class_len
        # Compute f1 score for hierarchical classification
        self.hierarchical_f1 = (self.f_beta ** 2 + 1.0) * self.hierarchical_precision * self.hierarchical_recall / \
                               ((self.f_beta**2) * self.hierarchical_precision + self.hierarchical_recall)



