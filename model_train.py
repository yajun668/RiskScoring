# This file is to preprocess dataset and train models for both DR and HFR cases.
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import statistics
import math
from sklearn.metrics import roc_auc_score,roc_curve,confusion_matrix,accuracy_score,classification_report
import scipy.stats as st
import pickle
from os.path import exists
import statsmodels.formula.api as smf
import statsmodels.api as sm
#import other python files
import common

pd.set_option("display.max_rows", 999)

# Preprocess data
def data_preprocess4DR(fs_method):
    new_train = pd.read_csv('./data/DR/' + 'DR_train_31features.csv')
    new_test = pd.read_csv('./data/DR/' + 'DR_test_31features.csv')
    feature_list = ['creatinine', 'hba1c', 'neuyes', 'nephyes', 'dia_duration', 'wbc', 'glucose', 'age', 'Hematocrit',
                    'sodium','bun', 'anion_gap', 'race_category']
    formular = 'DRYes ~ neuyes + nephyes + creatinine + hba1c + dia_duration + wbc + glucose + age + Hematocrit + sodium + bun + anion_gap + race_category'
    x_train = new_train[feature_list]
    y_train = new_train['DRYes']
    x_test = new_test[feature_list]
    y_test = new_test['DRYes']
    return x_train, x_test, y_train, y_test, feature_list,formular

def data_preprocess4HFX(fs_method):
    train = pd.read_csv('./data/HFX/' + 'train_hfx_data_ess0.01.csv')
    test = pd.read_csv('./data/HFX/' + 'test_hfx_data_ess0.01.csv')
    feature_list = ['Lab3','Lab0','Lab1', 'Los', 'Lab5', 'preInp1Y','preER1Y', 'cci', 'Lab2', 'AGE_IN_YEARS', 'Lab6', 'Lab4']
    formular = 'Next_Readm ~ Lab0 + Lab1 + Lab3 + Los + Lab5 + preInp1Y + preER1Y + cci + Lab2 + AGE_IN_YEARS + Lab6 + Lab4'
    x_train = train[feature_list]
    y_train = train['Next_Readm']
    x_test = test[feature_list]
    y_test = test['Next_Readm']
    return x_train, x_test, y_train, y_test, feature_list,formular


#This function is to find risk score for each feature of DR--model training
# cut_off_method: 1. expert; 2. even_length; 3. even_sample 4. auto_score
def risk_score4DR_train(coefficient,intercept,x_data, y_data, output_folder, cut_off_method = 'even_sample', is_rewrite_pkl = False):
    B_select = [] # store B and associated values
    cut_points = {} #store cutoff points for each feature
    base_value = 0.0 # base value from continuous features; later to add back to estimate risk index
    logreg_units = {} # logistic regression units: (level values - base) x beta
    file_name = pd.DataFrame(columns=['Variable', 'Coefficient', 'Stepsize', 'AUC', 'Accuracy', 'Sensitivity', 'Specificity', 'B', 'min_risk', 'max_risk'])
    step_range = 10
    cut_points_expert = {} # expert's input
    M_f = {}  # first level value
    M_l = {} # last level value
    ##############################################################
    # generate an Excel file for risk score; only need one for loop
    is_complete = False
    score_dict = {'Variable': [], 'Coefficient': [], 'Categories': [], 'Reference_value': [], 'logistic_unit':[]} #temporary data for risk score
    ##############################################################
    for key1 in coefficient: #enumerate each beta
        for j in range(1, step_range+1):
            #print("Base variable:", key1, ", step size:", j)
            B = j * abs(coefficient[key1])
            #print("B:", B)
            quan_even = [0, 0.2, 0.4, 0.6, 0.8, 1] # for even sample
            quan_auto = [0, 0.05, 0.2, 0.8, 0.95, 1]  # for even sample
            M_value = {} # store base value
            feature_name_score = [] # store feature name score list
            #for each beta, calculate scores
            for key2 in coefficient:
                beta = coefficient[key2]
                if key2 in ['neuyes', 'nephyes', 'race_category']: # method for categorical variable
                    fea_score = key2[0:len(key2) - 3] + '_score'
                    x_data[fea_score] = 0
                    if beta < 0:
                        if key2 == 'race_category':
                            x_data.loc[x_data[key2] == 0, fea_score] = round(abs(beta) * 2 / B)
                            x_data.loc[x_data[key2] == 1, fea_score] = round(abs(beta) / B)
                        else:
                            x_data.loc[x_data[key2] == 0, fea_score] = round(abs(beta) / B)
                    else:
                        if key2 == 'race_category':
                            x_data.loc[x_data[key2] == 1, fea_score] = round(abs(beta) / B)
                            x_data.loc[x_data[key2] == 2, fea_score] = round(abs(beta) * 2 / B)
                        else:
                            x_data.loc[x_data[key2] == 1, fea_score] = round(abs(beta) / B)
                    feature_name_score.append(fea_score)
                    ################################################################
                    # for creating Score Table only (categorical data); only run one time
                    if not is_complete:
                        if key2 == 'race_category':
                            for idx3 in range(3):
                                score_dict['Variable'].append(key2)
                                score_dict['Coefficient'].append(round(beta, 4))
                                score_dict['Reference_value'].append(idx3)
                                if idx3 == 0:
                                    score_dict['Categories'].append('Black')
                                elif idx3 == 1:
                                    score_dict['Categories'].append('Other')
                                elif idx3 == 2:
                                    score_dict['Categories'].append('White')
                                if beta < 0:
                                    score_dict['logistic_unit'].append(abs(round(beta, 4))*(2-idx3))
                                else:
                                    score_dict['logistic_unit'].append(abs(round(beta, 4))*idx3)
                        else:
                            for idx4 in range(2):
                                score_dict['Variable'].append(key2)
                                score_dict['Coefficient'].append(round(beta, 4))
                                score_dict['Reference_value'].append(idx4)
                                if idx4 == 0:
                                    score_dict['Categories'].append('No')
                                elif idx4 == 1:
                                    score_dict['Categories'].append('Yes')
                                if beta < 0:
                                    score_dict['logistic_unit'].append(abs(round(beta, 4)) * (1 - idx4))
                                else:
                                    score_dict['logistic_unit'].append(abs(round(beta, 4)) * idx4)
                    ################################################################
                else: # calculate scores for numerical features
                    min_value = x_data[key2].min()
                    max_value = x_data[key2].max()
                    if cut_off_method == 'even_length':
                    # even length method:
                        cut_points[key2] = [min_value, min_value + (max_value - min_value) / 5,
                                      min_value + (max_value - min_value) * 2 / 5, min_value + (max_value - min_value) * 3 / 5,
                                      min_value + (max_value - min_value) * 4 / 5, max_value + 1]
                        M_f[key2] = statistics.median(x_data.loc[x_data[key2] < cut_points[key2][1]][key2]) # first level
                        M_l[key2] = statistics.median(x_data.loc[x_data[key2] >= cut_points[key2][4]][key2])  # last level
                    elif cut_off_method == 'even_sample':
                        # even sample cut off method
                        cut_points[key2] = np.quantile(x_data[key2].to_numpy(), quan_even)
                        M_f[key2] = statistics.median(x_data.loc[x_data[key2] < cut_points[key2][1]][key2])  # first level
                        M_l[key2] = statistics.median(x_data.loc[x_data[key2] >= cut_points[key2][4]][key2])  # last level
                    elif cut_off_method == 'auto_score':
                        # even sample cut off method
                        cut_points[key2] = np.quantile(x_data[key2].to_numpy(), quan_auto)
                        M_f[key2] = statistics.median(x_data.loc[x_data[key2] < cut_points[key2][1]][key2])  # first level
                        M_l[key2] = statistics.median(x_data.loc[x_data[key2] >= cut_points[key2][4]][key2])  # last level
                    elif cut_off_method == 'expert':
                        M_f_tmp = {}
                        M_l_tmp = {}
                        # based on expert's input published in Wang Ru 2020 paper
                        cut_points_expert['creatinine'] = [x_data['creatinine'].min(), 0.5, 1, 1.5, 2, x_data['creatinine'].max()+1]
                        M_f_tmp['creatinine'] = 0.41
                        M_l_tmp['creatinine'] = 2.68
                        cut_points_expert['hba1c'] = [x_data['hba1c'].min(), 6, 8, 10, 12, x_data['hba1c'].max()+1]
                        M_f_tmp['hba1c'] = 5
                        M_l_tmp['hba1c'] = 14
                        cut_points_expert['dia_duration'] = [x_data['dia_duration'].min(), 1, 2, 3, 4, x_data['dia_duration'].max()+1]
                        M_f_tmp['dia_duration'] = 0.5
                        M_l_tmp['dia_duration'] = 9.2
                        cut_points_expert['wbc'] = [x_data['wbc'].min(), 4, 6, 8, 12, x_data['wbc'].max() +1]
                        M_f_tmp['wbc'] = 3.5
                        M_l_tmp['wbc'] = 18.2
                        cut_points_expert['glucose'] = [x_data['glucose'].min(), 60, 80, 100, 200, x_data['glucose'].max()+1]
                        M_f_tmp['glucose'] = 53
                        M_l_tmp['glucose'] = 364
                        cut_points_expert['age'] = [18, 35, 50, 65, 75, 85,  x_data['age'].max() + 1]
                        M_f_tmp['age'] = 26
                        M_l_tmp['age'] = 87.5
                        cut_points_expert['Hematocrit'] = [x_data['Hematocrit'].min(), 30, 35, 40, 50, x_data['Hematocrit'].max() + 1]
                        M_f_tmp['Hematocrit'] = 25.7
                        M_l_tmp['Hematocrit'] = 55
                        cut_points_expert['sodium'] = [x_data['sodium'].min(), 136, 144, x_data['sodium'].max() + 1]
                        M_f_tmp['sodium'] = 131.5
                        M_l_tmp['sodium'] = 146.5

                        #bun Lab3': [5,11,15,19,27,76.0001]
                        cut_points_expert['bun'] = [x_data['bun'].min(),11,15,19,27,x_data['bun'].max() + 1]
                        # print('key and cut off points', key2,cut_points[key2])
                        M_f_tmp['bun'] = statistics.median(x_data.loc[x_data['bun'] < cut_points_expert['bun'][1]]['bun'])  # first level
                        M_l_tmp['bun'] = statistics.median(x_data.loc[x_data['bun'] >= cut_points_expert['bun'][-2]]['bun'])  # last level

                        #anion_gap
                        cut_points_expert['anion_gap'] = [x_data['anion_gap'].min(),5,7,10,12,17, x_data['anion_gap'].max() + 1]
                        # print('key and cut off points', key2,cut_points[key2])
                        M_f_tmp['anion_gap'] = statistics.median(
                            x_data.loc[x_data['anion_gap'] < cut_points_expert['anion_gap'][1]]['anion_gap'])  # first level
                        M_l_tmp['anion_gap'] = statistics.median(
                            x_data.loc[x_data['anion_gap'] >= cut_points_expert['anion_gap'][-2]]['anion_gap'])  # last level

                        # assign values
                        cut_points[key2] = cut_points_expert[key2]
                        M_f[key2] = M_f_tmp[key2]
                        M_l[key2] = M_l_tmp[key2]

                    if beta > 0:
                        logreg_units[key2] = []
                        logreg_units[key2].append(0)
                        len_cut = len(cut_points[key2])
                        for idx in range(1,len_cut-2):
                            logreg_units[key2].append(beta * ((cut_points[key2][idx] + cut_points[key2][idx+1]) / 2 - M_f[key2]))
                        logreg_units[key2].append(beta * (M_l[key2] - M_f[key2]))
                        score = [round(val/B) for val in logreg_units[key2]]
                        M_value[key2] = M_f[key2] # add base value to M_value
                    else:
                        logreg_units[key2] = []
                        logreg_units[key2].append(beta * (M_f[key2] - M_l[key2]))
                        len_cut = len(cut_points[key2])
                        for idx in range(1, len_cut - 2):
                            logreg_units[key2].append(
                                beta * ((cut_points[key2][idx] + cut_points[key2][idx + 1]) / 2 - M_l[key2]))
                        logreg_units[key2].append(0)
                        score = [round(val / B) for val in logreg_units[key2]]
                        M_value[key2] = M_l[key2]  # add base value to M_value
                    x_data[key2 + '_score'] = pd.cut(x_data[key2], bins=cut_points[key2],right=False, labels=score, ordered=False)
                    feature_name_score.append(key2 + '_score')
                    ################################################################
                    # for creating Score Table only (numerical data); only run one time
                    if not is_complete:
                        Categories_list = []
                        for i in range(len(cut_points[key2])-1):
                            if i == 0:
                                mystr = '<' + str(round(cut_points[key2][1],1))
                                Categories_list.append(mystr)
                            elif i < len(cut_points[key2])-2:
                                mystr = str(round(cut_points[key2][i],1)) + '--'+ str(round(cut_points[key2][i+1],1))
                                Categories_list.append(mystr)
                            else:
                                mystr = '>' + str(round(cut_points[key2][i],1))
                                Categories_list.append(mystr)

                        refer_list = []
                        mylen = len(cut_points[key2])
                        refer_list.append(M_f[key2])
                        for idx2 in range(1,mylen-2):
                            refer_list.append((cut_points[key2][idx2] + cut_points[key2][idx2+1]) / 2)
                        refer_list.append(M_l[key2])
                        for j in range(len(refer_list)):
                            score_dict['Variable'].append(key2)
                            score_dict['Coefficient'].append(round(beta, 4))
                            score_dict['Categories'].append(Categories_list[j])
                            score_dict['Reference_value'].append(round(refer_list[j],1))
                            score_dict['logistic_unit'].append(round(logreg_units[key2][j],4))
                    ################################################################
            is_complete = True # only run once for-loop

            # add back base value and referent values for continuous variables
            base_value = 0
            for key2 in coefficient:
                if key2 not in ['neuyes','nephyes', 'race_category']:
                    base_value = base_value + coefficient[key2]*M_value[key2]
            xb = base_value
            xb = xb + intercept
            # reset  risk_score for each for loop
            risk_score = pd.DataFrame()
            risk_score = x_data[[x for x in feature_name_score]]
            #risk_score["total_risk"] = risk_score.sum(axis=1)

            # This will replace any null values in the DataFrame with zeros (fillna(0))
            # before calculating the sum along each row (sum(axis=1)).
            risk_score["total_risk"] = risk_score.fillna(0).sum(axis=1)

            xb = xb + B * risk_score['total_risk']
            risk_score['risk_prob'] = 1 / (1 + np.exp((-xb).to_numpy().tolist()))
            scale_max = risk_score['total_risk'].max()
            scale_min = risk_score['total_risk'].min()

            auc = roc_auc_score(y_data, risk_score['risk_prob'])
            fpr, tpr, thresholds = roc_curve(y_data, risk_score['risk_prob'], pos_label=1)

            # Calculate the Youden's J statistic
            youdenJ = tpr - fpr
            gmean = np.sqrt(tpr * (1 - fpr))
            # Find the optimal threshold
            index = np.argmax(youdenJ)
            thresholdOpt = round(thresholds[index], ndigits=4)
            youdenJOpt = round(gmean[index], ndigits=4)
            fprOpt = round(fpr[index], ndigits=4)
            tprOpt = round(tpr[index], ndigits=4)
            #print('Best Threshold: {} with Youden J statistic: {}'.format(thresholdOpt, youdenJOpt))

            pred = []
            for risk_prob in risk_score['risk_prob']:
                if risk_prob >= thresholdOpt:
                    pred.append(1)
                else:
                    pred.append(0)

            actual_binary = y_data.to_numpy().tolist()
            # if is_write_pkl is set to true or does not exist, write B_select values into pkl file
            if (not exists(output_folder  + 'B_select_' + cut_off_method + '.pkl')) or is_rewrite_pkl :
                X_A, Y_A = common.group_preds_by_label(pred, actual_binary)
                V_A10, V_A01 = common.structural_components(X_A, Y_A)
                B_select.append([B, auc, V_A10, V_A01])
            cm = classification_report(pred, y_data.to_numpy().tolist(), output_dict=True)
            #print(cm)
            #print('accuracy_score:', accuracy_score(pred, y_data.to_numpy().tolist()))

            file_name = file_name.append({'Variable': key1, 'Coefficient': coefficient[key1], 'Stepsize': j, 'AUC': auc, 'Accuracy': cm['accuracy'],
                                                  'Sensitivity': cm['1']['precision'], 'Specificity': cm['0']['precision'], 'B': B, 'min_risk': scale_min,
                                                  'max_risk': scale_max}, ignore_index=True)

    #write score table into a csv file
    score_Excel= pd.DataFrame(score_dict) #store risk score list
    score_Excel.to_csv(output_folder + cut_off_method + '_Score_Table.csv', index=False)
    #write summmary results into a csv file
    file_name.to_csv(output_folder + 'DR_' + 'perf_' + cut_off_method + '.csv', index=False)
    return B_select, file_name, cut_points, base_value, logreg_units

#This function is to find risk score for each feature of HFR--model training
# cut_off_method: 1. expert; 2. even_length; 3. even_sample 4. auto_score
def risk_score4HFX_train(coefficient,intercept,x_data, y_data, output_folder, cut_off_method = 'even_sample', is_rewrite_pkl = False):
    B_select = [] # store B and associated values
    cut_points = {} #store cutoff points for each feature
    base_value = 0.0 # base value from continuous features; later to add back to estimate risk index
    logreg_units = {} # logistic regression units: (level values - base) x beta
    perf_evenlength = pd.DataFrame(columns=['Variable', 'Coefficient', 'Stepsize', 'AUC', 'Accuracy', 'Sensitivity', 'Specificity', 'B', 'min_risk', 'max_risk'])
    step_range = 10
    cut_points_expert = {}  # expert's input
    M_f = {}  # first level value
    M_l = {}  # last level value
    ##############################################################
    # generate an Excel file for risk score; only need one for-loop
    is_complete = False
    score_dict = {'Variable': [], 'Coefficient': [], 'Categories': [], 'Reference_value': [],
                  'logistic_unit': []}  # temporary data for risk score
    ##############################################################
    for key1 in coefficient: #enumerate each beta
        for j in range(1, step_range+1):
            print("Base variable:", key1, ", step size:", j)
            B = j * abs(coefficient[key1])
            #print("B:", B)
            quan_even = [0, 0.2, 0.4, 0.6, 0.8, 1]  # for even sample
            quan_auto = [0, 0.05, 0.2, 0.8, 0.95, 1]  # for specific percentage
            M_value = {} # store base value
            feature_name_score = [] # store feature name score list
            #for each beta, calcualte scores
            for key2 in coefficient:
                beta = coefficient[key2]
                if key2 in ['preInp1Y', 'preER1Y']: # method for categorical variable
                    fea_score = key2 + '_score'
                    x_data[fea_score] = 0
                    x_data.loc[x_data[key2] == 1, fea_score] = round(abs(beta)/B)
                    x_data.loc[x_data[key2] == 2, fea_score] = round(abs(beta)*2/B)
                    feature_name_score.append(fea_score)
                    ################################################################
                    # for creating Score Table only (categorial data); only run one time
                    if not is_complete:
                        #each category has 3 values, so run 3 loops
                        for idx3 in range(3):
                            score_dict['Variable'].append(key2)
                            score_dict['Coefficient'].append(round(beta, 4))
                            if idx3==2:
                                score_dict['Categories'].append('>=2')
                            else:
                                score_dict['Categories'].append(idx3)
                            score_dict['Reference_value'].append(idx3)
                            score_dict['logistic_unit'].append(round(idx3*beta,4))
                    ################################################################
                else: # calculate scores for numerical features
                    min_value = x_data[key2].min()
                    max_value = x_data[key2].max()
                    if cut_off_method == 'even_length':
                        cut_points[key2] = [min_value, min_value + (max_value - min_value) / 5,
                                            min_value + (max_value - min_value) * 2 / 5,
                                            min_value + (max_value - min_value) * 3 / 5,
                                            min_value + (max_value - min_value) * 4 / 5, max_value + 1]
                        M_f[key2] = statistics.median(x_data.loc[x_data[key2] < cut_points[key2][1]][key2])  # first level
                        M_l[key2] = statistics.median(x_data.loc[x_data[key2] >= cut_points[key2][-2]][key2])  # last level
                        print('even length, variable, cut points and M_f and M_l are: ',key2, cut_points[key2],M_f[key2],M_l[key2] )
                    elif cut_off_method == 'even_sample':
                        # even sample cut off method
                        cut_points[key2] = np.quantile(x_data[key2].to_numpy(), quan_even)
                        print('Before removing duplicates, cut off points key2', key2, cut_points[key2])
                        cut_points[key2] = sorted(list(set(cut_points[key2])))  # remove duplicates in case some percentiles are the same
                        print('After removing duplicates, cut off points key2', key2, cut_points[key2])
                        M_f[key2] = statistics.median(x_data.loc[x_data[key2] < cut_points[key2][1]][key2])  # first level
                        M_l[key2] = statistics.median(x_data.loc[x_data[key2] >= cut_points[key2][-2]][key2])  # last level
                        print('even sample, variable, cut points and M_f and M_l are: ',key2, cut_points[key2],M_f[key2],M_l[key2] )
                    elif cut_off_method == 'auto_score':
                        # even sample cut off method
                        cut_points[key2] = np.quantile(x_data[key2].to_numpy(), quan_auto)
                        print('Before removing duplicates, cut off points key2', key2, cut_points[key2])
                        cut_points[key2] = cut_points[key2] = sorted(list(set(cut_points[key2]))) #remove duplicates in case some percentiles are the same
                        print('After removing duplicates, cut off points key2', key2,cut_points[key2])
                        M_f[key2] = statistics.median(x_data.loc[x_data[key2] < cut_points[key2][1]][key2])  # first level
                        M_l[key2] = statistics.median(x_data.loc[x_data[key2] >= cut_points[key2][-2]][key2])  # last level
                    elif cut_off_method == 'expert':
                        cut_points_expert = {'Lab0':[3,3.6,3.9,4.1,4.4,5.6001],'Lab1': [125,134,136,138,140,147.0001],
                            'Lab2': [66,143,177,213,268,513.0001], 'Lab3': [5,11,15,19,27,76.0001], 'Lab4': [6.9,9,9.7,10.3,11.1,15.1001],'Lab5': [0.39,0.6,0.76,0.9,1.2,5.9001],
                                            'Lab6': [20.4,26.9,28.8,30.7,33.1,44.8001],'Los': [1, 5, 7, 14, 600], 'cci': [0, 4, 6,6.1], 'AGE_IN_YEARS': [50,65,75,80,85,90,100] }
                        # assign values
                        cut_points[key2] = cut_points_expert[key2]
                        #print('key and cut off points', key2,cut_points[key2])
                        M_f[key2] = statistics.median(x_data.loc[x_data[key2] < cut_points[key2][1]][key2])  # first level
                        M_l[key2] = statistics.median(x_data.loc[x_data[key2] >= cut_points[key2][-2]][key2])  # last level
                    if beta > 0:
                        logreg_units[key2] = []
                        logreg_units[key2].append(0)
                        len_cut = len(cut_points[key2])
                        for idx in range(1, len_cut - 2):
                            logreg_units[key2].append(
                                beta * ((cut_points[key2][idx] + cut_points[key2][idx + 1]) / 2 - M_f[key2]))
                        logreg_units[key2].append(beta * (M_l[key2] - M_f[key2]))
                        score = [round(val / B) for val in logreg_units[key2]]
                        M_value[key2] = M_f[key2]  # add base value to M_value
                    else:
                        logreg_units[key2] = []
                        logreg_units[key2].append(beta * (M_f[key2] - M_l[key2]))
                        len_cut = len(cut_points[key2])
                        for idx in range(1, len_cut - 2):
                            logreg_units[key2].append(
                                beta * ((cut_points[key2][idx] + cut_points[key2][idx + 1]) / 2 - M_l[key2]))
                        logreg_units[key2].append(0)
                        score = [round(val / B) for val in logreg_units[key2]]
                        M_value[key2] = M_l[key2]  # add base value to M_value
                    x_data[key2 + '_score'] = pd.cut(x_data[key2], bins=cut_points[key2],right=False, labels=score, ordered=False)
                    feature_name_score.append(key2 + '_score')
                    ################################################################
                    # for creating Score Table only; only run one time
                    if not is_complete:
                        Categories_list = []
                        for i in range(len(cut_points[key2]) - 1):
                            if i == 0:
                                mystr = '<' + str(round(cut_points[key2][1], 1))
                                Categories_list.append(mystr)
                            elif i < len(cut_points[key2]) - 2:
                                mystr = str(round(cut_points[key2][i], 1)) + '--' + str(round(cut_points[key2][i + 1], 1))
                                Categories_list.append(mystr)
                            else:
                                mystr = '>' + str(round(cut_points[key2][i], 1))
                                Categories_list.append(mystr)
                        refer_list = []
                        mylen = len(cut_points[key2])
                        refer_list.append(M_f[key2])
                        for idx2 in range(1, mylen - 2):
                            refer_list.append((cut_points[key2][idx2] + cut_points[key2][idx2 + 1]) / 2)
                        refer_list.append(M_l[key2])

                        for j in range(len(refer_list)):
                            score_dict['Variable'].append(key2)
                            score_dict['Coefficient'].append(round(beta, 4))
                            score_dict['Categories'].append(Categories_list[j])
                            score_dict['Reference_value'].append(round(refer_list[j], 1))
                            score_dict['logistic_unit'].append(round(logreg_units[key2][j], 4))
                    ################################################################
            is_complete = True  # only run one for-loop

            # add back base value and referent values for continuous variables
            base_value = 0
            for key2 in coefficient:
                if key2 not in ['preInp1Y', 'preER1Y']:
                    base_value = base_value + coefficient[key2] * M_value[key2]
            xb = base_value
            xb = xb + intercept
            #reset  risk_score for each for loop
            risk_score =pd.DataFrame()
            risk_score = x_data[[x for x in feature_name_score]]

            #This will replace any null values in the DataFrame with zeros (fillna(0))
            # before calculating the sum along each row (sum(axis=1)).
            risk_score["total_risk"] = risk_score.fillna(0).sum(axis=1)

            #risk_score["total_risk"] = risk_score.sum(axis=1, skipna=True)
            xb = xb + B * risk_score['total_risk']
            risk_score['risk_prob'] = 1 / (1 + np.exp((-xb).to_numpy().tolist()))
            scale_max = risk_score['total_risk'].max()
            scale_min = risk_score['total_risk'].min()

            #write risk dataframe into csv file
            risk_score.to_csv(output_folder + cut_off_method + 'risk_score_debug.csv', index=False)
            auc = roc_auc_score(y_data, risk_score['risk_prob'])
            fpr, tpr, thresholds = roc_curve(y_data, risk_score['risk_prob'], pos_label=1)

            # Calculate the Youden's J statistic
            youdenJ = tpr - fpr
            gmean = np.sqrt(tpr * (1 - fpr))
            # Find the optimal threshold
            index = np.argmax(youdenJ)
            thresholdOpt = round(thresholds[index], ndigits=4)
            youdenJOpt = round(gmean[index], ndigits=4)
            fprOpt = round(fpr[index], ndigits=4)
            tprOpt = round(tpr[index], ndigits=4)
            # print('Best Threshold: {} with Youden J statistic: {}'.format(thresholdOpt, youdenJOpt))

            pred = []
            for risk_prob in risk_score['risk_prob']:
                if risk_prob >= thresholdOpt:
                    pred.append(1)
                else:
                    pred.append(0)

            actual_binary = y_data.to_numpy().tolist()
            # if is_write_pkl is set to true, write B_select values into pkl file
            if (not exists(output_folder  + 'B_select_' + cut_off_method + '.pkl')) or is_rewrite_pkl:
                X_A, Y_A = common.group_preds_by_label(pred, actual_binary)
                V_A10, V_A01 = common.structural_components(X_A, Y_A)
                B_select.append([B, auc, V_A10, V_A01])
            cm = classification_report(pred, y_data.to_numpy().tolist(), output_dict=True)

            perf_evenlength = perf_evenlength.append(
                {'Variable': key1, 'Coefficient': coefficient[key1], 'Stepsize': j, 'AUC': auc, 'Accuracy': cm['accuracy'],
                 'Sensitivity': cm['1']['precision'], 'Specificity': cm['0']['precision'], 'B': B, 'min_risk': scale_min,
                 'max_risk': scale_max}, ignore_index=True)

    #write score table into a csv file
    score_Excel= pd.DataFrame(score_dict) #store risk score list
    score_Excel.to_csv(output_folder + cut_off_method + '_HFX_Score_Table.csv', index=False)
    #write summmary results into a csv file
    perf_evenlength.to_csv(output_folder + cut_off_method + 'HFX_perf_evenlength' + '.csv', index=False)
    return B_select,perf_evenlength, cut_points, base_value, logreg_units