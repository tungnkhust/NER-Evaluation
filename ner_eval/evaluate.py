from ner_eval.utils import plot_confusion_matrix, Column
from sklearn.metrics import confusion_matrix
from ner_eval.metrics import get_metrics
from pandas import DataFrame
import pandas as pd
import sys
from collections import Counter
import os


ROOT_PATH = sys.path[1]


def analyse_inc(incorrects):
    inc_true = []
    inc_pred = []

    for sample in incorrects:
        if sample:
            for inc in sample:
                inc_true.append(inc[0]['entity'])
                inc_pred.append(inc[1]['entity'])
    count_inc = Counter()
    for e in inc_true:
        count_inc[e] += 1
    entitys_inc = []
    for e in count_inc:
        if count_inc[e] > 2:
            entitys_inc.append(e)
    entitys = sorted(entitys_inc)
    entity_true = []
    entity_pred = []
    for i in range(len(inc_true)):
        if inc_true[i] in entitys:
            entity_true.append(inc_true[i])
            entity_pred.append(inc_pred[i])

    cm = confusion_matrix(entity_true, entity_pred, labels=entitys)
    plot_confusion_matrix(cm=cm, target_names=entitys, title='incorrect entity', normalize=False)
    return count_inc


def analyse_miss(missings):
    mis = []
    for sample in missings:
        if sample:
            for e in sample:
                mis.append(e['entity'])
    count_mis = Counter()
    for e in mis:
        count_mis[e] += 1
    return count_mis


def analyse_spu(spuriuses):
    spu = []
    for sample in spuriuses:
        if sample:
            for e in sample:
                spu.append(e['entity'])
    count_spu = Counter()
    for e in spu:
        count_spu[e] += 1
    return count_spu


def eval_ner(df: DataFrame, epsilon=1e-5):
    tags = [tag.split(' ') for tag in df.tag.tolist()]
    tag_preds = [tag.split(' ') for tag in df.tag_pred.tolist()]
    metrics, incorrects, missings, spuriuses = get_metrics(tags, tag_preds)
    act = metrics['cor'] + metrics['inc'] + metrics['par'] + metrics['spu']
    precision = (metrics['cor'] + epsilon) / (act + epsilon)
    pos = metrics['cor'] + metrics['inc'] + metrics['par'] + metrics['mis']
    recall = (metrics['cor'] + epsilon) / (pos + epsilon)
    f1_score = (2 * precision * recall) / (precision + recall)
    count_inc = analyse_inc(incorrects)
    count_mis = analyse_miss(missings)
    count_spu = analyse_spu(spuriuses)

    report_entity = {}
    entity_report = list(count_inc.keys()) + list(count_mis.keys()) + list(count_spu.keys())
    entity_report = sorted(set(entity_report))

    for e in entity_report:
        report = {'inc': 0, 'mis': 0, 'spu': 0}
        if e in count_inc:
            report['inc'] = count_inc[e]
        if e in count_mis:
            report['mis'] = count_mis[e]
        if e in count_spu:
            report['spu'] = count_spu[e]
        report_entity[e] = report

    report_entity = sorted(report_entity.items(), key=lambda x: x[1]['inc'], reverse=True)
    entity_report = [report[0] for report in report_entity]
    inc_report = [report[1]['inc'] for report in report_entity]
    mis_report = [report[1]['mis'] for report in report_entity]
    spu_report = [report[1]['spu'] for report in report_entity]

    e_c = Column('entity', entity_report)
    i_c = Column('inc', inc_report)
    m_c = Column('mis', mis_report)
    s_c = Column('spu', spu_report)

    if os.path.exists(ROOT_PATH + '/report') is False:
        os.mkdir(ROOT_PATH + '/report')
    with open(ROOT_PATH + '/report/report_all.md', 'w') as pf:
        c1 = Column(key='precision', value=[round(precision, 4)])
        c2 = Column(key='recall', value=[round(recall, 4)])
        c3 = Column(key='f1_score', value=[round(f1_score, 4)])
        pf.write('### Precision-Recall-F1 Score\n')
        pf.write(c1.print_key() + '    ' + c2.print_key() + '    ' + c3.print_key())
        pf.write('\n')
        pf.write(c1.print_item(0) + '    ' + c2.print_item(0) + '    ' + c3.print_item(0))
        pf.write('\n')
        pf.write('\n')
        pf.write('### MUC Score\n')
        pf.write(e_c.print_key() + '    ' + i_c.print_key() + '    ' + m_c.print_key() + '    ' + s_c.print_key())
        pf.write('\n')
        for i in range(len(entity_report)):
            pf.write(e_c.print_item(i) + '    ' + i_c.print_item(i) + '    ' + m_c.print_item(i) + '    ' + s_c.print_item(i))
            pf.write('\n')

df = pd.read_csv(ROOT_PATH + '/data/prediction-onenet.csv')
eval_ner(df)