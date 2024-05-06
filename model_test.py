# coding: utf-8
# Load package and Data
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import statistics
import math
from sklearn.metrics import roc_auc_score,roc_curve,confusion_matrix,accuracy_score,classification_report
from sklearn.metrics import precision_recall_curve,auc
import sklearn.metrics as metrics
import scipy.stats as st
import pickle
from os.path import exists
import statsmodels.formula.api as smf
import statsmodels.api as sm
import matplotlib.pyplot as plt

pd.set_option("display.max_rows", 999)

# After geting B automatically, we use such B to test our dataset for DR risk scores
def RiskScore4DR_Test(B_value, coefficient,intercept,cut_points, base_value, logreg_units, x_data, y_data, output_folder):
    B = B_value
    feature_name_score = []  # store feature name score list
    score_UB = 0 # store upper bound under current B value
    # for each beta, calculate scores
    for key2 in coefficient:
        if key2 in ['neuyes', 'nephyes','race_category']:  # method for categorical variable
            fea_score = key2[0:len(key2) - 3] + '_score'
            x_data[fea_score] = 0
            if key2 =='race_category':
                if coefficient[key2] > 0:
                    x_data.loc[x_data[key2] == 1, fea_score] = round(abs(coefficient[key2]) / B)
                    x_data.loc[x_data[key2] == 2, fea_score] = round(abs(coefficient[key2])*2 / B)
                    score_UB += round(abs(coefficient[key2])*2 / B)
                else:
                    x_data.loc[x_data[key2] == 0, fea_score] = round(abs(coefficient[key2]) * 2 / B)
                    x_data.loc[x_data[key2] == 1, fea_score] = round(abs(coefficient[key2]) / B)
                    score_UB += round(abs(coefficient[key2]) * 2 / B)
            else:
                if coefficient[key2] > 0:
                    x_data.loc[x_data[key2] == 1, fea_score] = round(abs(coefficient[key2]) / B)
                    score_UB += round(abs(coefficient[key2]) / B)
                else:
                    x_data.loc[x_data[key2] == 0, fea_score] = round(abs(coefficient[key2]) / B)
                    score_UB += round(abs(coefficient[key2]) / B)

            print('Feature name, current UB score', key2, score_UB)
            feature_name_score.append(fea_score)
        else:  # calculate scores for numerical features
            score = [round(val/B) for val in logreg_units[key2]] #get risk scores
            x_data[key2 + '_score'] = pd.cut(x_data[key2], bins=cut_points[key2], right=False, labels=score,
                                             ordered=False)
            feature_name_score.append(key2 + '_score')

            if coefficient[key2] > 0:
                score_UB += round(logreg_units[key2][-1] / B) # get the last element by using index: -1
            else:
                score_UB += round(logreg_units[key2][0] / B)

            print('Feature name,cut points, all level, current UB score', key2,cut_points[key2], score, score_UB)
    # add back base value and referent values for continuous variables
    xb = base_value
    xb = xb + intercept
    risk_score = x_data[[x for x in feature_name_score]]
    #risk_score["total_risk"] = risk_score.sum(axis=1)

    # This will replace any null values in the DataFrame with zeros (fillna(0))
    # before calculating the sum along each row (sum(axis=1)).
    risk_score["total_risk"] = risk_score.fillna(0).sum(axis=1)

    xb = xb + B * risk_score['total_risk']
    risk_score['risk_prob'] = 1 / (1 + np.exp((-xb).to_numpy().tolist()))
    scale_min = risk_score['total_risk'].min()
    scale_max = risk_score['total_risk'].max()
    print('min and Max values are: ',scale_min,'-----',scale_max)
    print('Risk scale: ', 0, '-----', score_UB)

    auc_value = roc_auc_score(y_data, risk_score['risk_prob'])
    #print('B star and Test AUC is:', B, auc)
    fpr, tpr, thresholds = roc_curve(y_data, risk_score['risk_prob'], pos_label=1)

    # Calculate precision, recall and thresholds
    precision, recall, thresholds_prc = precision_recall_curve(y_data, risk_score['risk_prob'])

    # Calculate the AUPRC
    auprc = auc(recall, precision)
    print(f"Area Under the Precision-Recall Curve (AUPRC): {auprc:.3f}")


    # add total risk score, risk probability, and y data into new df, and then write result into csv file.
    x_data['total_risk'] = risk_score["total_risk"].copy()
    x_data['risk_prob'] = risk_score['risk_prob'].copy()
    new_df = x_data.join(y_data)
    test_result_name = output_folder + 'Risk_Score_DR_Test_' + str(B) + '.csv'
    new_df.to_csv(test_result_name, index = False)

    return auc_value, fpr, tpr, score_UB,precision, recall,auprc

# After geting B automatically, we use such B to test our dataset for HFX
def RiskScore4HFX_Test(B_value, coefficient,intercept,cut_points, base_value, logreg_units, x_data, y_data, output_folder):
    B = B_value
    feature_name_score = []  # store feature name score list
    score_UB = 0  # store upper bound under current B value
    # for each beta, calculate scores
    for key2 in coefficient:
        if key2 in ['preInp1Y', 'preER1Y']:  # method for categorical variable
            fea_score = key2 + '_score'
            x_data[fea_score] = 0
            x_data.loc[x_data[key2] == 1, fea_score] = round(abs(coefficient[key2])/B)
            x_data.loc[x_data[key2] == 2, fea_score] = round(abs(coefficient[key2])*2 / B)
            score_UB += round(abs(coefficient[key2])*2 / B)
            print('Feature name, all level, current UB score', key2, [0,1], score_UB)
            feature_name_score.append(fea_score)
        else:  # calculate scores for numerical features
            score = [round(val / B) for val in logreg_units[key2]]  # get risk scores
            x_data[key2 + '_score'] = pd.cut(x_data[key2], bins=cut_points[key2], right=False, labels=score, ordered=False)
            feature_name_score.append(key2 + '_score')
            if coefficient[key2] > 0:
                score_UB += round(logreg_units[key2][-1] / B) # get the last element by using index: -1
            else:
                score_UB += round(logreg_units[key2][0] / B)

            print('Feature name,cut points, all level, current UB score', key2, cut_points[key2], score, score_UB)
    # add back base value and referent values for continuous variables
    xb = base_value
    xb = xb + intercept
    risk_score = x_data[[x for x in feature_name_score]]
    # This will replace any null values in the DataFrame with zeros (fillna(0))
    # before calculating the sum along each row (sum(axis=1)).
    risk_score["total_risk"] = risk_score.fillna(0).sum(axis=1)

    xb = xb + B * risk_score['total_risk']
    risk_score['risk_prob'] = 1 / (1 + np.exp((-xb).to_numpy().tolist()))
    scale_min = risk_score['total_risk'].min()
    scale_max = risk_score['total_risk'].max()
    print('min and Max values are: ', scale_min, '-----', scale_max)
    print('Risk scale: ', 0, '-----', score_UB)

    auc_value = roc_auc_score(y_data, risk_score['risk_prob'])
    # print('B star and Test AUC is:', B, auc)
    fpr, tpr, thresholds = roc_curve(y_data, risk_score['risk_prob'], pos_label=1)

    # Calculate precision, recall and thresholds
    precision, recall, thresholds_prc = precision_recall_curve(y_data, risk_score['risk_prob'])
    # Calculate the AUPRC
    auprc = auc(recall, precision)
    print(f"Area Under the Precision-Recall Curve (AUPRC): {auprc:.3f}")

    # add total risk score, risk probability, and y data into new df, and then write result into csv file.
    x_data['total_risk'] = risk_score["total_risk"].copy()
    x_data['risk_prob'] = risk_score['risk_prob'].copy()
    new_df = x_data.join(y_data)
    test_result_name = output_folder + 'Risk_Score_HFX_Test_' + str(B) + '.csv'
    new_df.to_csv(test_result_name, index=False)

    return auc_value, fpr, tpr, score_UB, precision, recall,auprc