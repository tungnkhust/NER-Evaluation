from typing import List


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
