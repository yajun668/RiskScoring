# This file contains common functions
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import statistics
import math
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix, accuracy_score, classification_report
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import auc as au_prc
import scipy.stats as st
import pickle
from os.path import exists
import statsmodels.formula.api as smf
import statsmodels.api as sm
import matplotlib.pyplot as plt

#find optimal parameter B
def findOptBvalue(B_select,inc_tol_p, output_folder, cut_off_method = 'expert',  is_rewrite_pkl = False):
    # Load all B values and its AUC
    #if is_write_pkl is set to true, write B_select values into pkl file
    if (not exists(output_folder + 'B_select_' + cut_off_method + '.pkl')) or is_rewrite_pkl:
        file_name = output_folder + 'B_select_' + cut_off_method + '.pkl'
        open_file = open(file_name, "wb")
        pickle.dump(B_select, open_file)
        open_file.close()
    else:
        file_name = output_folder  + 'B_select_' + cut_off_method + '.pkl'
        open_file = open(file_name, "rb")
        loaded_list = pickle.load(open_file)
        open_file.close()
        B_select = loaded_list

    # To Make B values sorted in descending order
    # Get B value and sort in descending order
    B_values = []
    AUC_max = 0 # used to find max AUC
    for b in B_select:
        B_values.append(b[0])
        if b[1] > AUC_max:
            AUC_max = b[1]
    B_values.sort(reverse=True)
    #to make B_selected sorted consistent with B_values
    B_select_sorted = []
    for i in range(len(B_select)):
        for b in B_select:
            if b[0] == B_values[i]:
                B_select_sorted.append(b)

    # input for a, beta, beta prime
    s = B_values[len(B_values)-1] # last element of B_values
    t = B_values[0]
    t_prime = t
    # auc for s and t
    auc_s = B_select_sorted[B_values.index(s)][1]
    auc_t = B_select_sorted[B_values.index(t)][1]
    # step 1
    while auc_s*0.99 >= auc_t:
        # step 2
        ell = (1-0.618) * (t - s) # steps
        # step 3
        # t_prime = t # because we have 0.95*auc_s, we can directly use last t as t_prime
        # step 4
        #t = find_smaller_element(t - ell, B_values) #find B whch is just smaller than t - ell in List B_values
        print('s, t, t-prime, auc_s and auc_t', s, t, t_prime, auc_s,auc_t)
        print('in While loop, t-ell, s, t and aut', t - ell, s, t, auc_t)
        t = B_values[closest(B_values, t - ell)]
        auc_t = B_select_sorted[B_values.index(t)][1]
        print('after step update, new t and auc_t', t, auc_t)

    t_prime = t
    print('t prime AND auc is:', t_prime, B_select_sorted[B_values.index(t_prime)][1])
    # step 5: get B_hat
    B_hat = 0
    auc_B_hat = 0
    for B in B_values:
        if s <= B <= t_prime:
            if auc_B_hat < B_select_sorted[B_values.index(B)][1]:
                auc_B_hat = B_select_sorted[B_values.index(B)][1]
                B_hat = B
                print('In the for loop, B, B hat and its AUC is: ',B, B_hat, auc_B_hat)

    print('B hat and its AUC is: ',B_hat, auc_B_hat)
    # step 6
    B_star = B_hat # B_star is used to store optimal B value
    auc_B_star = -1.0

    print('founded b hat and its AUC is:',B_hat, B_select_sorted[B_values.index(B_hat)][1])

    # step 7-10
    auc_B_hat = B_select_sorted[B_values.index(B_hat)][1]
    A10_B_hat = B_select_sorted[B_values.index(B_hat)][2]
    A01_B_hat = B_select_sorted[B_values.index(B_hat)][3]
    var_B = (get_S_entry(A10_B_hat, A10_B_hat, auc_B_hat, auc_B_hat) * 1 / len(A10_B_hat)
             + get_S_entry(A01_B_hat, A01_B_hat, auc_B_hat, auc_B_hat) * 1 / len(A01_B_hat))
    for B in B_values:
        if B_hat <= B <= t_prime:
            auc_B = B_select_sorted[B_values.index(B)][1]
            A10_B = B_select_sorted[B_values.index(B)][2]
            A01_B = B_select_sorted[B_values.index(B)][3]
            # Delong test
            # step 9
            var_A = (get_S_entry(A10_B, A10_B, auc_B, auc_B) * 1 / len(A10_B)
                     + get_S_entry(A01_B, A01_B, auc_B, auc_B) * 1 / len(A01_B))
            covar_AB = (get_S_entry(A10_B_hat, A10_B, auc_B_hat, auc_B) * 1 / len(A10_B_hat)
                        + get_S_entry(A01_B_hat, A01_B, auc_B_hat, auc_B) * 1 / len(A01_B_hat))
            z = z_score(var_A, var_B, covar_AB, auc_B, auc_B_hat)
            aucp_new_opt = (1 - st.norm.cdf(abs(z))) * 2
            if not pd.isna(aucp_new_opt):
                # find insignificant update
                if aucp_new_opt >= inc_tol_p:
                    B_star = B
                    auc_B_star = B_select_sorted[B_values.index(B_star)][1]
                    break
    return B_star, auc_B_star, AUC_max, t_prime



# This function is used to get estimate of parameters beta for logistic regression using GLM function
def glm_logreg(x_data, y_data, formular, output_folder):
    #x_train, x_test, y_train, y_test, feature_list = data_preprocess()
    data = x_data.join(y_data)
    # generalized linear model
    lg = smf.glm(formula=formular, data=data, family=sm.families.Binomial()).fit()
    #write logistic regression result summary and prediction results for test dataset
    auc_value = roc_auc_score(y_data, lg.predict(x_data))
    # print('B star and Test AUC is:', B, auc)
    fpr, tpr, thresholds = roc_curve(y_data, lg.predict(x_data), pos_label=1)

    # Calculate precision, recall and thresholds
    precision, recall, thresholds_prc = precision_recall_curve(y_data, lg.predict(x_data))
    # Calculate the AUPRC
    auprc = au_prc(recall, precision)
    print(f"Area Under the Precision-Recall Curve (AUPRC): {auprc:.3f}")

    with open(output_folder + 'lg_' + 'model.txt', 'w') as f:
        f.write(str(lg.summary()))
        f.write('\n roc_auc_score using Logistic regression : ' + str(roc_auc_score(y_data, lg.predict(x_data))))

    intercept = 0.0
    coefficient = {}
    for fea in lg.params.keys().tolist():
        if fea == 'Intercept':
            intercept = lg.params[fea]
        else:
            coefficient[fea] = lg.params[fea]
    return coefficient, intercept, auc_value, fpr, tpr, precision, recall,auprc

# draw graphs about relationship between B vs AUC
def draw_graphs(perf_evenlength,B_star, AUC_max, t_prime, output_folder, cut_off_method):
    x_b = perf_evenlength['B'].to_numpy().tolist()
    y_auc = perf_evenlength['AUC'].to_numpy().tolist()
    plt.scatter(x_b, y_auc, [2])
    #plt.plot(x_b[x_b.index(B_star)], y_auc[x_b.index(B_star)], 'r*')
    #plt.text(x_b[x_b.index(B_star)], y_auc[x_b.index(B_star)], 'cut off point B: ' + str(round(B_star,3)) + '      AUC: ' + str(round(y_auc[x_b.index(B_star)],3)))
    if 'DR' in output_folder:
        # global view for DR
        plt.axis([0, 10, 0.45, 0.85])
    else:
        # global view for hfr
        plt.axis([0, 3, 0.41, 0.65])
    #plt.show()
    figure_name_global = output_folder + cut_off_method + 'result_global.pdf'
    plt.savefig(figure_name_global)
    plt.close()

    plt.scatter(x_b, y_auc, [2])
    #plt.plot(x_b[x_b.index(B_hat)], y_auc[x_b.index(B_hat)], 'b*')
    #print('------in figure b hat and b hat value are-------', x_b[x_b.index(B_hat)], y_auc[x_b.index(B_hat)])
    plt.plot(x_b[x_b.index(B_star)], y_auc[x_b.index(B_star)], 'b*') # plot B star
    print('b star, auc are: ', B_star, y_auc[x_b.index(B_star)])
    print('b max auc and index', AUC_max,y_auc[y_auc.index(AUC_max)], x_b[y_auc.index(AUC_max)])
    plt.plot(x_b[y_auc.index(AUC_max)], AUC_max, 'r^') # plot max AUC
    #plt.plot(t_prime, y_auc[x_b.index(t_prime)], 'gs')  # plot turning point
    print('print t prime and B star', t_prime,B_star)
    if 'DR' in output_folder:
        # close view for DR
        #plt.axis([0, 1.0, 0.45, ])
        plt.axis([0, t_prime + 0.1, 0.69,0.85])
    else:
        # close view for hfr
        #plt.axis([0, 0.8, 0.45, 0.65])
        plt.axis([0, t_prime + 0.05, 0.49, 0.65])

    #plt.show()
    figure_name_local = output_folder + cut_off_method + 'result_local.pdf'
    plt.savefig(figure_name_local)
    plt.close()

# AUC-ROC curve improve function
def closest(lst, K):
    return min(range(len(lst)), key=lambda i: abs(lst[i] - K))
# In python list, find a item which is just smaller than the input number
def find_smaller_element(input_number, input_list):
    smaller_elements = [x for x in input_list if x < input_number]
    if smaller_elements:
        return max(smaller_elements)
    else:
        return None  # No smaller element found


def auc(X, Y):
    return 1 / (len(X) * len(Y)) * sum([kernel(x, y) for x in X for y in Y])

def kernel(X, Y):
    return .5 if Y == X else int(Y < X)

def structural_components(X, Y):
    V10 = [1 / len(Y) * sum([kernel(x, y) for y in Y]) for x in X]
    V01 = [1 / len(X) * sum([kernel(x, y) for x in X]) for y in Y]
    return V10, V01

def get_S_entry(V_A, V_B, auc_A, auc_B):
    return 1 / (len(V_A) - 1) * sum([(a - auc_A) * (b - auc_B) for a, b in zip(V_A, V_B)])

def z_score(var_A, var_B, covar_AB, auc_A, auc_B):
    return (auc_A - auc_B) / ((var_A + var_B - 2 * covar_AB) ** (.5))

def group_preds_by_label(preds, actual):
    X = [p for (p, a) in zip(preds, actual) if a]
    Y = [p for (p, a) in zip(preds, actual) if not a]
    return X, Y