from sklearn.metrics import (confusion_matrix, accuracy_score, average_precision_score,
precision_score, recall_score, f1_score, classification_report, precision_recall_curve)

import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 10

def get_classification_acc(preds, gts, plot_cm=False):  
    if len(preds) != len(gts):  
        raise ValueError("The predictions and truths lists must have the same length.")  
    for pred, gt in zip(preds, gts):  
        if pred[0] != gt[0]:  
            raise ValueError("The predictions and gt lists must have the same order.") 
    classes = ["safe","Low", "High"]
    y_pred = [i[1] for i in preds]
    y_true =   [i[1] for i in gts]
    
    acc = accuracy_score(y_pred, y_true) # (TP+TN)/total
    f1 = f1_score( y_true, y_pred, average='macro' )
    p = precision_score(y_true, y_pred, average='macro')
    r = recall_score(y_true, y_pred, average='macro')
    report = classification_report(y_true, y_pred, target_names=classes, digits=3)
    
    confusion = confusion_matrix(y_pred, y_true)
    if plot_cm:
        plt.figure()
        ax = plt.subplot(111)
        plt.imshow(confusion, cmap=plt.cm.Blues)
        plt.colorbar()

        indices = range(len(confusion))
        plt.xticks(indices, classes, )
        plt.yticks(indices, classes)
        plt.tick_params(labelsize=12)
        labels = ax.get_xticklabels()+ ax.get_yticklabels()
        [label.set_fontname('Verdana') for label in labels]

        plt.xlabel('pred',fontsize=12)
        plt.ylabel('true',fontsize=12)
        for first_index in range(len(confusion)):
            for second_index in range(len(confusion[first_index])):
                plt.text(first_index, second_index, confusion[first_index][second_index])
        plt.savefig('runs/tmp/cm.png')
        plt.show()
    return acc, f1, p, r, report

def get_data(GT_path):
    class_gt = []
    with open(GT_path, 'r') as file:  
        for line in file:  
            info = line.strip().split(' ')
            fn, idx = info[0], int(info[1])
            class_gt.append([fn, idx])
    class_gt = sorted(class_gt, key=lambda x: x[0]) 
    return class_gt

if __name__ == "__main__":
    gt = get_data("runs/tmp/gt.txt")
    pred = get_data("runs/tmp/pred.txt")
    acc, f1, p, r, report = get_classification_acc(pred, gt, plot_cm=True)
    print("Acc:{:.2f},  F1-score:{:.2f},  P:{:.2f},  R:{:.2f}".format(acc, f1, p, r))  

    print(report)

