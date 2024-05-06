# start with this main file which outlines steps to generate risk scores
import pandas as pd
import os
import numpy as np
from sklearn.model_selection import train_test_split
import statistics
import math
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix, accuracy_score, classification_report
from sklearn.dummy import DummyClassifier
from sklearn.metrics import precision_recall_curve,auc
import scipy.stats as st
import pickle
from os.path import exists
import statsmodels.formula.api as smf
import statsmodels.api as sm
import matplotlib.pyplot as plt
#import other python files
import model_train,model_test,common

inc_tol_p = 0.01 #test significance level

def main(case_study, output_folder, cut_method,fs_method):
    result_output = {}
    global df_summary  # Declare df_summary as a global variable; use to save result summary
    if case_study == 'DR':
        # Step 1: data preprocessing
        x_train, x_test, y_train, y_test, feature_list, formular = model_train.data_preprocess4DR(fs_method)

        # step 2: get parameters of logistics regression
        coefficient, intercept, auc_value, fpr, tpr, precision1, recall1, auprc1 = common.glm_logreg(x_train, y_train, formular, output_folder)

        # step 3: get all risk scores and AUCs for different B values
        print('Step 3: get all risk scores and AUCs for different B values on training data.......')
        B_select, perf, cut_points, base_value, logreg_units = model_train.risk_score4DR_train(coefficient, intercept, x_train, y_train,output_folder, cut_off_method=cut_method,is_rewrite_pkl=False)

        # step 4: #find optimal parameter B
        print('Step 4: find optimal parameter B.......')
        # cut off method: 1. expert; 2. even_length; 3. even_sample  4. auto_score
        B_star, auc_B_star, AUC_max, t_prime = common.findOptBvalue(B_select, inc_tol_p, output_folder, cut_off_method=cut_method, is_rewrite_pkl=False)

        # step 5:draw graphs for training
        common.draw_graphs(perf, B_star, AUC_max, t_prime, output_folder, cut_method)

        # step 6: test
        print('step 6: test data...')
        # for B star result
        auc, fpr, tpr, score_UB, precision_bstar, recall_bstar, auprc_bstar = model_test.RiskScore4DR_Test(B_star, coefficient, intercept, cut_points, base_value,
                                                               logreg_units, x_test, y_test, output_folder)
        print('Score Upper Bound, B and Test AUC is:', score_UB, 'B star', B_star, auc)
        plt.plot(fpr, tpr, label=r'$B^{*}$, AUROC=%0.3f' % (auc))

        # write result to dict
        result_output['case_study']=[case_study]
        result_output['B'] = ['B star: ' + str(B_star)]
        result_output['fs_method']=[fs_method]
        result_output['cut_off']=[cut_method]
        result_output['auc'] = [auc]
        result_output['scoreUB']= [score_UB]

        # for B_min result
        min_key_dr = min(coefficient, key=lambda k: abs(coefficient[k])) # find the key whose absolute value is the minimum
        auc, fpr, tpr, score_UB, precision_bmin, recall_bmin, auprc_bmin = model_test.RiskScore4DR_Test(abs(coefficient[min_key_dr]), coefficient, intercept, cut_points,
                                                               base_value, logreg_units, x_test, y_test, output_folder)
        print('Score Upper Bound, B and Test AUC is:', score_UB, 'B_min', abs(coefficient[min_key_dr]), auc)
        plt.plot(fpr, tpr, label=r"$B_{min}$, AUROC=%0.3f" % (auc))

        result_output['case_study'].append(case_study)
        result_output['B'].append('B_min: ' + str(abs(coefficient[min_key_dr])))
        result_output['fs_method'].append(fs_method)
        result_output['cut_off'].append(cut_method)
        result_output['auc'].append(auc)
        result_output['scoreUB'].append(score_UB)

        #5 for 5*beta_age result
        auc, fpr, tpr, score_UB, precision_beta_age, recall_beta_age, auprc_beta_age = model_test.RiskScore4DR_Test(abs(coefficient['age']) * 5, coefficient, intercept,
                                                               cut_points, base_value, logreg_units, x_test, y_test,
                                                               output_folder)
        print('Score Upper Bound, B and Test AUC is:', score_UB, '5*beta age', abs(coefficient['age']) * 5, auc)
        plt.plot(fpr, tpr, label=r'$5\beta_{age}$, AUROC=%0.3f' % (auc)) #plot ROC curve

        result_output['case_study'].append(case_study)
        result_output['B'].append('5*beta_age: ' + str(abs(coefficient['age']) * 5))
        result_output['fs_method'].append(fs_method)
        result_output['cut_off'].append(cut_method)
        result_output['auc'].append(auc)
        result_output['scoreUB'].append(score_UB)

        #append result to df
        df_summary = pd.concat([df_summary, pd.DataFrame(result_output)], ignore_index=True)

        # write result to csv file
        print('Reuslt outp print', result_output)
        df = pd.DataFrame(result_output)
        df.to_csv(output_folder + cut_method + '_Scale_AUC.csv')

        # logistic regression
        coefficient, intercept, auc1, fpr1, tpr1,precision_lg, recall_lg, auprc_lg = common.glm_logreg(x_test, y_test, formular, output_folder)
        plt.plot(fpr1, tpr1, label='Logistic Regression AUROC=%0.3f' % (auc1))

        # Custom settings for the plot
        plt.plot([0, 1], [0, 1], linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel(r'$1-$Specificity')
        plt.ylabel('Sensitivity')
        plt.title('ROC Curve for DR')
        plt.legend(loc="lower right")
        plt.savefig(output_folder + cut_method + '_AUROC_DR.pdf')
        plt.close()
        # plt.show()  # Display

        # Plot the Precision-Recall Curve
        plt.figure(2)  # Create a figure with figure number 2
        plt.plot(recall_bstar, precision_bstar, label=r'$B^{*}$, AUPRC=%0.3f' % (auprc_bstar))
        plt.plot(recall_bmin, precision_bmin, label=r'$B_{min}$, AUPRC=%0.3f' % (auprc_bmin))
        plt.plot(recall_beta_age, precision_beta_age, label=r'$5\beta_{age}$, AUPRC=%0.3f' % (auprc_beta_age))
        plt.plot(recall_lg, precision_lg, label=r'Logistic Regression AUPRC=%0.3f' % (auprc_lg))

        # calculate the no skill (random model) line as the proportion of the positive class
        no_skill = len(y_test[y_test == 1]) / len(y_test)
        # plot the no skill precision-recall curve
        plt.plot([0, 1], [no_skill, no_skill], linestyle='--', label=r'Random Model AUPRC=%0.3f' % (no_skill))

        plt.xlabel('Recall')
        plt.ylabel('Precision')
        # Custom settings for the plot
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        #plt.title('Precision-Recall Curve', fontsize=16)
        plt.title('Precision-Recall Curve for DR')

        plt.legend(loc="upper right")
        plt.savefig(output_folder + cut_method + '_PRC curve.pdf')
        plt.close()


    elif case_study == 'HFX':
        # Step 1: data preprocessing
        x_train, x_test, y_train, y_test, feature_list, formular = model_train.data_preprocess4HFX(fs_method)

        # step 2: get parameters of logistics regression
        coefficient, intercept, auc_value, fpr, tpr, precision1, recall1, auprc1 = common.glm_logreg(x_train, y_train, formular, output_folder)

        # step 3: get all risk scores and AUCs for different B values
        B_select, perf, cut_points, base_value, logreg_units = model_train.risk_score4HFX_train(coefficient, intercept, x_train, y_train, output_folder,cut_off_method=cut_method, is_rewrite_pkl=False)

        # step 4: #find optimal parameter B
        B_star, auc_B_star, AUC_max, t_prime = common.findOptBvalue(B_select, inc_tol_p, output_folder,
                                                                    cut_off_method=cut_method, is_rewrite_pkl=False)
        print('B star and AUC are:', B_star, auc_B_star)

        # step 5:draw graphs for training
        common.draw_graphs(perf, B_star, AUC_max, t_prime, output_folder, cut_method)

        # step 6: test
        print('step 6: test data...')
        # for B_star result
        auc, fpr, tpr, score_UB, precision_bstar, recall_bstar, auprc_bstar = model_test.RiskScore4HFX_Test(B_star, coefficient, intercept, cut_points,
                                                               base_value, logreg_units, x_test, y_test,
                                                               output_folder)
        print('Score Upper Bound, B and Test AUC is:', score_UB, 'B star', B_star, auc)
        plt.plot(fpr, tpr, label=r'$B^{*}$, AUROC=%0.3f' % (auc)) #plot ROC curve
        # write result to dict
        result_output['case_study'] = [case_study]
        result_output['B'] = ['B star: ' + str(B_star)]
        result_output['fs_method'] = [fs_method]
        result_output['cut_off'] = [cut_method]
        result_output['auc'] = [auc]
        result_output['scoreUB'] = [score_UB]

        #for B_min result
        min_key_hfr = min(coefficient, key=lambda k: abs(coefficient[k])) # find the key whose absolute value is the minimum
        auc, fpr, tpr, score_UB, precision_bmin, recall_bmin, auprc_bmin = model_test.RiskScore4HFX_Test(abs(coefficient[min_key_hfr]), coefficient, intercept,
                                                               cut_points, base_value, logreg_units, x_test, y_test,output_folder)

        print('Score Upper Bound, B and Test AUC is:', score_UB, 'B_min', abs(coefficient[min_key_hfr]), auc)
        plt.plot(fpr, tpr, label=r"$B_{min}$, AUROC=%0.3f" % (auc))

        result_output['case_study'].append(case_study)
        result_output['B'].append('B_min: ' + str(abs(coefficient[min_key_hfr])))
        result_output['fs_method'].append(fs_method)
        result_output['cut_off'].append(cut_method)
        result_output['auc'].append(auc)
        result_output['scoreUB'].append(score_UB)

        # for 5*beta_age result
        auc, fpr, tpr, score_UB, precision_beta_age, recall_beta_age, auprc_beta_age = model_test.RiskScore4HFX_Test(abs(coefficient['AGE_IN_YEARS']) * 5, coefficient, intercept,
                                                               cut_points, base_value, logreg_units, x_test, y_test,
                                                               output_folder)
        print('Score Upper Bound, B and Test AUC is:', score_UB, '5*beta age', abs(coefficient['AGE_IN_YEARS']) * 5, auc)
        plt.plot(fpr, tpr, label=r'$5\beta_{age}$, AUROC=%0.3f' % (auc))
        result_output['case_study'].append(case_study)
        result_output['B'].append('5*beta_age: ' + str(abs(coefficient['AGE_IN_YEARS']) * 5))
        result_output['fs_method'].append(fs_method)
        result_output['cut_off'].append(cut_method)
        result_output['auc'].append(auc)
        result_output['scoreUB'].append(score_UB)

        # append result to df
        df_summary = pd.concat([df_summary, pd.DataFrame(result_output)], ignore_index=True)
        print('df result summary',df_summary)

        # write reulst to csv file
        print('Result output print', result_output)
        df = pd.DataFrame(result_output)
        df.to_csv(output_folder + cut_method + '_Scale_AUC.csv')

        # logistic regression
        coefficient, intercept, auc1, fpr1, tpr1, precision_lg, recall_lg, auprc_lg = common.glm_logreg(x_test, y_test, formular, output_folder)
        plt.plot(fpr1, tpr1, label='Logistic Regression AUROC=%0.3f' % (auc1))

        # Custom settings for the plot
        plt.plot([0, 1], [0, 1], linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel(r'$1-$Specificity')
        plt.ylabel('Sensitivity')
        plt.title('ROC Curve for HFR')
        # plt.title('Receiver Operating Characteristic')
        plt.legend(loc="lower right")
        plt.savefig(output_folder + cut_method + '_AUROC_HFR.pdf')
        plt.close()

        # Plot the Precision-Recall Curve
        plt.figure(2)  # Create a figure with figure number 2
        plt.plot(recall_bstar, precision_bstar, label=r'$B^{*}$, AUPRC=%0.3f' % (auprc_bstar))
        plt.plot(recall_bmin, precision_bmin, label=r'$B_{min}$, AUPRC=%0.3f' % (auprc_bmin))
        plt.plot(recall_beta_age, precision_beta_age, label=r'$5\beta_{age}$, AUPRC=%0.3f' % (auprc_beta_age))
        plt.plot(recall_lg, precision_lg, label=r'Logistic Regression AUPRC=%0.3f' % (auprc_lg))

        # calculate the no skill line (random model) as the proportion of the positive class
        no_skill = len(y_test[y_test == 1]) / len(y_test)
        # plot the no skill precision-recall curve
        plt.plot([0, 1], [no_skill, no_skill], linestyle='--', label=r'Random Model AUPRC=%0.3f' % (no_skill))

        plt.xlabel('Recall')
        plt.ylabel('Precision')
        # Custom settings for the plot
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.title('Precision-Recall Curve for HFR')
        plt.legend(loc="upper right")
        plt.savefig(output_folder + cut_method + '_PRC curve.pdf')
        plt.close()

if __name__ == "__main__":
    #case_study = 'DR' # case name: 'DR' and 'HFX'
    #cut_method = 'even_sample' # cut off method: 1. expert; 2. even_length; 3. even_sample  4. auto_score
    cut_method_list = ['expert','even_length', 'even_sample', 'auto_score']
    case_study_list = ['DR', 'HFX']
    fs_method_list = ['Ensemble']

    columns = ['case_study','B', 'fs_method', 'cut_off', 'auc', 'scoreUB']
    df_summary = pd.DataFrame(columns=columns)
    # The below code is to run all test dataset
    for case_study in case_study_list:
        for cut_method in cut_method_list:
            for fs_method in fs_method_list:
                output_folder = './output/' + case_study + '/' + fs_method + '/' + cut_method + '/'
                # if the folder doesn't exist, create a new folder
                if not os.path.exists(output_folder):
                    os.makedirs(output_folder)
                main(case_study,output_folder, cut_method, fs_method)

    #write result summary
    df_summary.to_csv('./output/result_summary.csv', index=False)