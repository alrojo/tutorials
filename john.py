import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics as sk

def getCNNdata():
    p1 = np.load('predictions1.npy')
    p2 = np.load('predictions2.npy')
    p3 = np.load('predictions3.npy')
    p4 = np.load('predictions4.npy')
    p5 = np.load('predictions5.npy')
    t1 = np.load('labels1.npy')
    t2 = np.load('labels2.npy')
    t3 = np.load('labels3.npy')
    t4 = np.load('labels4.npy')
    t5 = np.load('labels5.npy')
    
    auc1 = sk.roc_auc_score(t1,p1)
    auc2 = sk.roc_auc_score(t2,p2)
    auc3 = sk.roc_auc_score(t3,p3)
    auc4 = sk.roc_auc_score(t4,p4)
    auc5 = sk.roc_auc_score(t5,p5)
    auc = (auc1+auc2+auc3+auc4+auc5)/5
    
    
    fpr1, tpr1, ths1 = sk.roc_curve(t1, p1)
    fpr2, tpr2, ths2 = sk.roc_curve(t2, p2)
    fpr3, tpr3, ths3 = sk.roc_curve(t3, p3)
    fpr4, tpr4, ths4 = sk.roc_curve(t4, p4)
    fpr5, tpr5, ths5 = sk.roc_curve(t5, p5)
    fpr = np.sort(np.hstack((fpr1, fpr2, fpr3, fpr4, fpr5)))
    tpr = np.sort(np.hstack((tpr1, tpr2, tpr3, tpr4, tpr5)))
    ths = np.sort(np.hstack((ths1, ths2, ths1, ths4, ths5)))
    return auc, tpr, fpr, ths

auc, tpr, fpr, ths = getCNNdata()
roc = np.genfromtxt('roc.csv', delimiter=',')
x = roc[:,0]
svm = roc[:,1]
knn = roc[:,2]
dt = roc[:,3]
rf = roc[:,4]
plt.plot(fpr,tpr,'r', label='CNN')
plt.plot(x,svm,'b', label='SVM')
plt.plot(x,knn,'y', label='kNN')
plt.plot(x,dt,'c', label='DT')
plt.plot(x,rf,'g', label='RF')
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.show()
