# Evaluation utilities
from .evaluation_automodel import (
    eval_sts_task,
    eval_classification_task,
    eval_pair_task,
    STSDataset,
    ClasssifyDataset,
    PairDataset
)
from .retrieval import (
    eval_retrieval_task,
    load_benchmark,
    test_retrieval_tasks,
)
# from .evaluation_model_define import (
#     eval_sts_task as eval_sts_task_custom,
#     eval_classification_task as eval_classification_task_custom,
#     eval_pair_task as eval_pair_task_custom
# )

__all__ = [
    'eval_sts_task',
    'eval_classification_task',
    'eval_pair_task',
    'eval_retrieval_task',
    'load_benchmark',
    'test_retrieval_tasks',
    'STSDataset',
    'ClasssifyDataset',
    'PairDataset',
    'eval_sts_task_custom',
    'eval_classification_task_custom',
    'eval_pair_task_custom'
]
