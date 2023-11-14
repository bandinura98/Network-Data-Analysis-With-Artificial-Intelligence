import matplotlib.pyplot as plt #visualise as graph 
import seaborn as sns #visualise as graph 
import pandas as pd #data manupilation 
from numpy import genfromtxt #numerical processing library like matlab


from sklearn.naive_bayes import GaussianNB #learning algorithm


from sklearn.preprocessing import LabelEncoder, StandardScaler #label encoder
from sklearn.model_selection import train_test_split #splitting data
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, precision_score, recall_score) #accuraccy f-mesure performance etc 



#############################################################################
############################# functions #####################################
#############################################################################


def print_stats_metrics(y_test,y_pred): #prints all the scores
    print('Accuraccy score is = %.2f' % accuracy_score(y_test, y_pred))
    print('F1-mesure score is = %.3f' % f1_score(y_test,y_pred, average="weighted"))
    print('Recall score is = %.3f' % recall_score(y_test,y_pred, average="weighted"))
    print('Precision score is = %.3f' % precision_score(y_test,y_pred, average="weighted" ))
    confmat = confusion_matrix(y_test,y_pred)
    print(confmat)
    print(pd.crosstab(y_test,y_pred, rownames=['True'], colnames=['Predicted'], margins=True))

def print_conf_plt(y_test,y_pred): #shows the confusion matrix as graph
    confmat = confusion_matrix(y_test,y_pred)
    plt.figure(figsize=(8,8))
    sns.set(font_scale = 1.5)
    ax = sns.heatmap(
        confmat, # confusion matrix 2D array 
        annot=True, # show numbers in the cells
        fmt='d', # show numbers as integers
        cbar=False, # don't show the color bar
        cmap='flag', # customize color map
        vmax=175 # to get better color contrast
    )
    ax.set_xlabel("Predicted", labelpad=20)
    ax.set_ylabel("Actual", labelpad=20)
    plt.show()  
    
#############################################################################
################################Reading######################################
#############################################################################

features = genfromtxt("network_datas.csv", delimiter=',', usecols=(1,2,3,4,5,6,7,8,9,10,11,12), dtype=float, skip_header=1)

class_value = genfromtxt("network_datas.csv", delimiter=',', usecols=(-1), dtype=str, skip_header=1)


#############################################################################
###############################Labeling######################################
#############################################################################

labels = LabelEncoder().fit_transform(class_value)

features_normalized = StandardScaler().fit_transform(features)

x_train , x_test , y_train , y_test = train_test_split(features_normalized, labels, test_size=0.70, random_state=31)


#############################################################################
###############################Training######################################
#############################################################################

clf = GaussianNB()
clf.fit(x_train, y_train)

#############################################################################
##############################Print and write################################
#############################################################################

y_pred = clf.predict(x_test,)
print("naive bayes PREDICTIONS")
print_stats_metrics(y_test,y_pred)
print_conf_plt(y_test,y_pred)


f = open("results.txt", "a")
f.write("naive bayes scores\n")
f.write("F score is : " + str(f1_score(y_test,y_pred, average="weighted")))
f.write("\n")
f.write(str(confusion_matrix(y_test,y_pred)))
f.write("\n")
f.close()
