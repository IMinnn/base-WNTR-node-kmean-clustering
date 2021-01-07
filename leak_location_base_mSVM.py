from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score
import sklearn.svm as SVM
import numpy as np
import pandas as pd

def data_processing(ori_data_field):
    ori_data = ori_data_field[1:, :]
    Y = ori_data[:, 0]
    #Y = Y.astype(np.int)
    X_24h = ori_data[:, 2:]
    #X_24h = X_24h.astype(np.float)
    x_train, x_test, y_train, y_test = train_test_split(X_24h, Y, test_size=0.3)
    return x_train, x_test, y_train, y_test

def svc(kernel,c):
    return SVM.SVC(kernel=kernel, decision_function_shape="ovo", C=c, probability=True)

def modelist():
    modelist = []
    kernalist = {"linear", "poly", "rbf", "sigmoid"}
    kernalist = {"linear"}
    for eachkerna in kernalist:
        for eachC in range(1, 10, 1):
            modelist.append(svc(eachkerna, eachC))
    return modelist

def svc_model(model):
    model.fit(x_train, y_train)
    acu_train = model.score(x_train, y_train)
    acu_test = model.score(x_test, y_test)
    y_pred = model.predict(x_test)
    y_proba = model.predict_proba(x_test)
    recall = recall_score(y_test, y_pred, average="macro")
    return acu_train, acu_test, recall, y_pred, y_proba

def pro_out(pro_score, model, ID_list):
    class_label = model.classes_
    frame = pd.DataFrame(pro_score, index=ID_list, columns=class_label)

    print(frame)
    #print(frame.sort_values(ID_list[0]))
    print(13)

def run_svc_model(modelist, if_proinf=False, ID_list=None):
    result = {"kernel": [],
              "C": [],
              "acu_train": [],
              "acu_test": [],
              "recall": []
              }
    best_acu = 0
    best_model = None
    best_y = None
    best_y_pro = None
    for model in modelist:
        acu_train, acu_test, recall, y_pred, y_proba = svc_model(model)
        try:
            result["kernel"].append(model.kernel)
        except:
            result["kernel"].append(None)
        result["C"].append(model.C)
        result["acu_train"].append(acu_train)
        result["acu_test"].append(acu_test)
        result["recall"].append(recall)
        #if acu_test >= best_acu:
           # best_model = model
           # best_y = y_pred
           # best_y_pro = y_proba
    #pro_out(best_y_pro, model, ID_list)

    return pd.DataFrame(result)





if __name__ == "__main__":
    file = "data/ky2/testdata.csv"
    out_file = "outdata/ky2/score_SVM.csv"
    ori_data_field = np.loadtxt(file, dtype=str, delimiter=',')
    x_train, x_test, y_train, y_test = data_processing(ori_data_field)
    result = run_svc_model(modelist(), ID_list=ori_data_field[1:, 1])
    result.to_csv(path_or_buf=out_file)


    print(result)


