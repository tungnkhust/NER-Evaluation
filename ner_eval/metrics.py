from typing import List
from ner_eval.utils import get_metric


def get_metrics(y_true: List[List], y_pred: List[List]):
    n_samples = len(y_true)
    metrics = {
        'support': 0,
        'cor': 0,
        'inc': 0,
        'par': 0,
        'mis': 0,
        'spu': 0
    }
    incorrects = []
    missings = []
    spuriuses = []

    for i in range(n_samples):
        metric, incorrect, missing, spurius = get_metric(y_true[i], y_pred[i])
        metrics['cor'] += metric['cor']
        metrics['inc'] += metric['inc']
        metrics['par'] += metric['par']
        metrics['mis'] += metric['mis']
        metrics['spu'] += metric['spu']
        metrics['support'] += metric['support']
        incorrects.append(incorrect)
        missings.append(missing)
        spuriuses.append(spurius)
    return metrics, incorrects, missings, spuriuses


def precision_score(y_true: List[List], y_pred: List[List], epsilon=1e-6):
    """
    Compute the precision score.
    :param y_true: List of ground truth labels, Each label is list of tag for each tokens.
    :param y_pred: List of predicted labels, Each label is list of tag for each tokens.
    :param epsilon:
    :return: precision score.
    """
    metrics, _, _, _ = get_metrics(y_true, y_pred)
    act = metrics['cor'] + metrics['inc'] + metrics['par'] + metrics['spu']
    precision = (metrics['cor'] + epsilon) / (act + epsilon)
    return precision


def recall_score(y_true: List[List], y_pred: List[List], epsilon=1e-6):
    """
    Compute the recall score.
    :param y_true: List of ground truth labels, Each label is list of tag for each tokens.
    :param y_pred: List of predicted labels, Each label is list of tag for each tokens.
    :param epsilon:
    :return: recall score.
    """
    metrics, _, _, _ = get_metrics(y_true, y_pred)
    pos = metrics['cor'] + metrics['inc'] + metrics['par'] + metrics['mis']
    recall = (metrics['cor']+epsilon)/(pos+epsilon)
    return recall


def f1_score(y_true: List[List], y_pred: List[List], epsilon=1e-6):
    """
    Compute the f1-score.
    :param y_true: List of ground truth labels, Each label is list of tag for each tokens.
    :param y_pred: List of predicted labels, Each label is list of tag for each tokens.
    :param epsilon:
    :return: f1-score score.
    """
    metrics, _, _, _ = get_metrics(y_true, y_pred)
    act = metrics['cor'] + metrics['inc'] + metrics['par'] + metrics['spu']
    pos = metrics['cor'] + metrics['inc'] + metrics['par'] + metrics['mis']
    precision = (metrics['cor']+epsilon)/(act+epsilon)
    recall = (metrics['cor']+epsilon)/(pos+epsilon)
    f1_score = (2*precision*recall)/(precision + recall)
    return f1_score


