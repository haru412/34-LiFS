import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score, recall_score, f1_score
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve


def calculate_all_metrics(class_label, pred_class, pred_scores):
    """
    计算所有评估指标, 包括AUC、ACC、sensitivity、specificity、F1等
    :param class_label: 真实类别标签
    :param pred_class: 预测的类别标签
    :param pred_scores: 预测的概率得分
    :return: 包含所有评估指标的字典
    """
    # 计算AUC
    auc_result = {}
    auc_result['auc'] = roc_auc_score(class_label, pred_scores[:, 1])
    auc_result['fpr'], auc_result['tpr'], _ = roc_curve(class_label, pred_scores[:, 1])

    # 计算ACC
    acc = accuracy_score(class_label, pred_class)

    # 计算sensitivity
    sen = recall_score(class_label, pred_class, pos_label=1)

    # 计算specificity
    spe = recall_score(class_label, pred_class, pos_label=0)

    # 计算F1
    f1 = f1_score(class_label, pred_class, pos_label=1)

    metrics = {
        'auc_result': auc_result,
        'ACC': acc,
        'Sensitivity': sen,
        'Specificity': spe,
        'F1': f1
    }

    return metrics


def plot_best_roc_curves(metrics, save_path):
    """
    绘制ROC曲线并保存为PDF文件
    :param metrics: 包含AUC指标的字典
    :param save_path: 保存ROC曲线的文件路径
    """
    auc = metrics['auc_result']['auc']
    fpr = metrics['auc_result']['fpr']
    tpr = metrics['auc_result']['tpr']

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'ROC curve (area = {auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic curve')
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

