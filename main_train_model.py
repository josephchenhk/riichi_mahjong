# -*- coding: utf-8 -*-
"""
Created on Wed Oct 25 12:03:04 2017

@author: joseph.chen
"""
import time

from train_model.train_model import scores_data_preprocessing
from train_model.train_model import waiting_tiles_data_preprocessing
from train_model.train_model import train_scores_partial_fit

if __name__=="__main__":

    tic = time.time()
    
    
#    is_waiting_data_preprocessing()
    waiting_tiles_data_preprocessing() 
#    scores_data_preprocessing() 
    
#    clf, avg_accuracy_scores, avg_auc_scores = train_is_waiting_partial_fit(load_classifier=False, save_classifier=True)
#    plot_scores(avg_accuracy_scores,
#                avg_auc_scores,
#                save_path="trained_classifiers/plots/",
#                save_name=("Accuracy_is_waiting",
#                           "AUC_is_waiting")
#                )
    
#    for tile in range(34):
#        clf, avg_accuracy_scores, avg_auc_scores = train_waiting_tiles_partial_fit(tile=tile, load_classifier=False, save_classifier=True)     
#        plot_scores(avg_accuracy_scores, 
#                    avg_auc_scores,
#                    save_path="trained_classifiers/plots/",
#                    save_name=("Accuracy_waiting_tile_{}.png".format(tile),
#                               "AUC_waiting_tile_{}.png".format(tile))
#                    )
    
#    clf, avg_mse_scores = train_scores_partial_fit(load_classifier=False, save_classifier=True)
#    
#    waiting_tiles_evaluation = WaitingTilesEvaluation()
#    evaluation = waiting_tiles_evaluation.accuracy_of_prediction()
#    print("Evaluation value: {}".format(evaluation))

    toc = time.time()
    print("Elapsed time: {:.2} seconds.".format(toc-tic)) 