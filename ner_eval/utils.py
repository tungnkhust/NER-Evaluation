from typing import List
import numpy as np
import matplotlib.pyplot as plt
import itertools
import sys
import os

ROOT_PATH = sys.path[1]


def get_entity_from_BIO(tags: List[str]) -> List:
    """
    Get entities from BIO tags.
    :param tags: List of tagging for each tokens.
    :return: List entities that tagged in sentence.
    """
    if tags is None:
        return []
    s = 0
    e = 0
    entity = None
    entities = []
    for i, tag in enumerate(tags):
        if tag[0] == 'B':
            entity = tag[2:]
            s = i
            e = i
            if i == len(tags) - 1:
                entities.append({'entity': entity, 'start': s, 'end': e})
        elif tag[0] == 'I':
            e += 1
            if i == len(tags) - 1:
                entities.append({'entity': entity, 'start': s, 'end': e})
        elif tag == 'O':
            if entity is not None:
                entities.append({'entity': entity, 'start': s, 'end': e})
                entity = None

    return entities


def compare(e_true: dict, e_pred: dict):
    """
    Compare 2 entities:
    Have 5 state of two entities:
        1 - Correct(cor): Both are the same.
        2 - Incorrect(inc): The predicted entity and the true entity don’t match
        3 - Partial(par): Both are the same entity but the boundaries of the surface string wrong
        4 - Missing(mis): The system doesn't predict entity
        5 - Spurius(spu): The system predict entity which doesn't exist in the true label.

    :param e_true: Entity in ground truth label. e
    :param e_pred: Entity in predicted label.
    :return:
    """
    s1 = int(e_true['start'])
    e1 = int(e_true['end'])
    s2 = int(e_pred['start'])
    e2 = int(e_pred['end'])
    if s1 == s2 and e1 == e2:
        if e_true['entity'] == e_pred['entity']:
            return 1
        else:
            return 2
    if ((s1 <= s2) and (s2 <= e1)) or ((s2 <= s1) and (s1 <= e2)):
        if e_true['entity'] == e_pred['entity']:
            return 3
        else:
            return 2
    if e1 < s2:
        return 4
    if e2 < s1:
        return 5


def get_metric(y_true: list, y_pred: list):
    """
    Get metric to evaluate for y_true and y_pred.
    :param y_true: List of tagging for each tokens of ground truth label .
    :param y_pred: List of tagging for each tokens of predicted label.
    :return: Dict include metric to evaluate for each entity
    and list of incorrect, missing and spurius entities.
    """
    entities_true = get_entity_from_BIO(y_true)
    entities_pred = get_entity_from_BIO(y_pred)
    metrics = {
        'support': len(entities_true),
        'cor': 0,
        'inc': 0,
        'par': 0,
        'mis': 0,
        'spu': 0,
    }
    incorrect = []
    missing = []
    spurius = []
    while len(entities_true) != 0 or len(entities_pred) != 0:
        if len(entities_true) == 0:
            metrics['spu'] += 1
            spurius.append(entities_pred[0])
            del entities_pred[0]
            continue

        if len(entities_pred) == 0:
            metrics['mis'] += 1
            spurius.append(entities_true[0])
            del entities_true[0]
            continue

        e1 = entities_true[0]
        e2 = entities_pred[0]

        state = compare(e1, e2)
        if state == 1:
            metrics['cor'] += 1
            del entities_true[0]
            del entities_pred[0]
        elif state == 2:
            metrics['inc'] += 1
            incorrect.append((e1, e2))
            del entities_true[0]
            del entities_pred[0]
        elif state == 3:
            metrics['par'] += 1
            del entities_true[0]
            del entities_pred[0]
        elif state == 4:
            metrics['mis'] += 1
            missing.append(e1)
            del entities_true[0]
        elif state == 5:
            metrics['spu'] += 1
            spurius.append(e2)
            del entities_pred[0]
    return metrics, incorrect, missing, spurius


def plot_confusion_matrix(cm,
                          target_names,
                          title='Confusion matrix',
                          cmap=None,
                          normalize=True,
                          save_dir=None):


    """Function to plot confusion matrics.

    :param cm: confusion_matrix: function in sklearn.
    :param target_names: list of classes.
    :param cmap: str or matplotlib Colormap: Colormap recognized by matplotlib.
    :param normalize: normalizes confusion matrix over the true (rows), predicted (columns) conditions or all the population.
    :param save_dir: str: directory address to save.
    """

    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=90)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.2f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label. Metrics: accuracy={:0.2f}; misclass={:0.2f}'.format(accuracy, misclass))
    if os.path.exists(ROOT_PATH + '/report') is False:
        os.mkdir(ROOT_PATH + '/report')
    plt.savefig((ROOT_PATH + '/report/{}.png'.format(title)))


class Column:
    def __init__(self, key, value=None):
        self.key = key
        if value is None:
            value = []
        self.value = value
        self.max_seq = self.max_line()

    def __getitem__(self, i):
        return self.value[i]

    def max_line(self):
        if len(self.value) == 0:
            return len(self.key)
        return max(max([len(str(v)) for v in self.value]), len(self.key))

    def print_item(self, i):
        return str(self.value[i]) + ' '*(self.max_seq-len(str(self.value[i])))

    def print_key(self):
        return str(self.key) + ' ' * (self.max_seq - len(str(self.key)))