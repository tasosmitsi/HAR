##############################################################
#                                                            #
#    Mark Hoogendoorn and Burkhardt Funk (2017)              #
#    Machine Learning for the Quantified Self                #
#    Springer                                                #
#    Chapter 7                                               #
#                                                            #
##############################################################

import pandas as pd
from pathlib import Path
import time
import numpy as np
start = time.time()


from Chapter7.PrepareDatasetForLearning import PrepareDatasetForLearning
from Chapter7.LearningAlgorithms import ClassificationAlgorithms
from Chapter7.Evaluation import ClassificationEvaluation
from Chapter7.FeatureSelection import FeatureSelectionClassification
from util import util
from util.VisualizeDataset import VisualizeDataset

# Read the result from the previous chapter, and make sure the index is of the type datetime.
GRANULARITIES = [50]
DATA_PATH = Path('./intermediate_datafiles/')

for GRANULARITY in GRANULARITIES:
    DATASET_FNAME = 'HAR_5_' + 'all' + '_g' + str(GRANULARITY) + '_result.csv'
    RESULT_FNAME = 'HAR_7_' + 'all' + '_g' + str(GRANULARITY) + '_result.csv'
    EXPORT_TREE_PATH = Path('./figures/' + 'HAR_5_{}_g{}'.format('all', GRANULARITY) + '/')

    # Next, we declare the parameters we'll use in the algorithms.
    N_FORWARD_SELECTION = 100

    try:
        dataset = pd.read_csv(DATA_PATH / DATASET_FNAME, index_col=0)
    except IOError as e:
        print('File not found, try to run previous crowdsignals scripts first!')
        raise e

    # dataset.index = pd.to_datetime(dataset.index)

    # Let us create our visualization class again.
    DataViz = VisualizeDataset('HAR_7_{}_g{}'.format('all', GRANULARITY))

    # Let us consider our first task, namely the prediction of the label. We consider this as a non-temporal task.

    # We create a single column with the categorical attribute representing our class. Furthermore, we use 70% of our data
    # for training and the remaining 30% as an independent test set. We select the sets based on stratified sampling. We remove
    # cases where we do not know the label.

    # Transform the dataset to a list of datasets one for each participant
    gb = dataset.groupby('subject_name')    
    datasets = [gb.get_group(x) for x in gb.groups]
 
    prepare = PrepareDatasetForLearning()
    train_X, test_X, train_y, test_y = prepare.split_multiple_datasets_classification(datasets, ['label'], 'like', 0.7, filter=True, temporal=False)

    print('Training set length is: ', len(train_X.index))
    print('Test set length is: ', len(test_X.index))

    # Select subsets of the features that we will consider:

    basic_features = ['roll_belt','pitch_belt','yaw_belt','total_accel_belt','gyros_belt_x',
                    'gyros_belt_y','gyros_belt_z','accel_belt_x','accel_belt_y','accel_belt_z',
                    'magnet_belt_x','magnet_belt_y','magnet_belt_z','total_accel_arm',
                    'gyros_arm_x','gyros_arm_y','gyros_arm_z','accel_arm_x','accel_arm_y','accel_arm_z','magnet_arm_x',
                    'magnet_arm_y','magnet_arm_z','roll_dumbbell','pitch_dumbbell','yaw_dumbbell','total_accel_dumbbell',
                    'gyros_dumbbell_x','gyros_dumbbell_y','gyros_dumbbell_z','accel_dumbbell_x','accel_dumbbell_y',
                    'accel_dumbbell_z','magnet_dumbbell_x','magnet_dumbbell_y','magnet_dumbbell_z','roll_forearm','pitch_forearm',
                    'yaw_forearm','total_accel_forearm','gyros_forearm_x','gyros_forearm_y','gyros_forearm_z','accel_forearm_x',
                    'accel_forearm_y','accel_forearm_z','magnet_forearm_x','magnet_forearm_y','magnet_forearm_z']

    pca_features = ['pca_1','pca_2','pca_3','pca_4','pca_5']
    time_features = [name for name in dataset.columns if '_temp_' in name]
    freq_features = [name for name in dataset.columns if (('_freq' in name) or ('_pse' in name))]
    print('#basic features: ', len(basic_features))
    print('#PCA features: ', len(pca_features))
    print('#time features: ', len(time_features))
    print('#frequency features: ', len(freq_features))
    cluster_features = ['cluster']
    print('#cluster features: ', len(cluster_features))
    features_after_chapter_3 = list(set().union(basic_features, pca_features))
    features_after_chapter_4 = list(set().union(basic_features, pca_features, time_features, freq_features))
    features_after_chapter_5 = list(set().union(basic_features, pca_features, time_features, freq_features, cluster_features))


    # # First, let us consider the performance over a selection of features:

    # fs = FeatureSelectionClassification()

    # features, ordered_features, ordered_scores = fs.forward_selection(N_FORWARD_SELECTION,
    #                                                               train_X[features_after_chapter_5],
    #                                                               test_X[features_after_chapter_5],
    #                                                               train_y,
    #                                                               test_y,
    #                                                               gridsearch=False)

    # DataViz.plot_xy(x=[range(1, N_FORWARD_SELECTION+1)], y=[ordered_scores],
    #                 xlabel='number of features', ylabel='accuracy')


    
    # RF_imp = fs.alex_RF_selection(train_X, np.ravel(train_y), n_estimators = 10000, MOST_IMP_FEATURES = 80)
    # ET_imp = fs.alex_ET_selection(train_X, np.ravel(train_y), n_estimators = 10000, MOST_IMP_FEATURES = 80)

    # print(np.intersect1d(RF_imp.index, ET_imp.index))

    
    # based on python2 features, slightly different. 
    selected_features = ['accel_belt_x_temp_entropy_ws_60','accel_belt_z_temp_std_ws_30',
    'accel_forearm_x_temp_entropy_ws_60','gyros_arm_x_temp_mad_ws_30','gyros_arm_x_temp_mad_ws_60',
    'gyros_arm_y_temp_mad_ws_60','gyros_belt_z_temp_max_ws_30','gyros_belt_z_temp_max_ws_60','gyros_belt_z_temp_std_ws_30',
    'gyros_belt_z_temp_std_ws_60','gyros_forearm_y_temp_mad_ws_30','gyros_forearm_y_temp_mad_ws_60','magnet_arm_x_temp_mean_ws_30',
    'magnet_arm_x_temp_median_ws_30','magnet_arm_z_temp_entropy_ws_60','magnet_arm_z_temp_slope_ws_60','magnet_belt_y_temp_entropy_ws_60',
    'magnet_belt_y_temp_std_ws_30','magnet_belt_z_temp_entropy_ws_60','magnet_belt_z_temp_std_ws_30',
    'magnet_dumbbell_x_temp_entropy_ws_60','magnet_dumbbell_y_temp_entropy_ws_60','magnet_forearm_x_temp_entropy_ws_60',
    'pca_2_temp_max_ws_60','total_accel_belt_temp_entropy_ws_60']

    # # # Let us first study the impact of regularization and model complexity: does regularization prevent overfitting?

    learner = ClassificationAlgorithms()
    eval = ClassificationEvaluation()
    start = time.time()


    reg_parameters = [0.0001, 0.001, 0.01, 0.1, 1, 10]
    performance_training = []
    performance_test = []
    ## Due to runtime constraints we run the experiment 3 times, yet if you want even more robust data one should increase the repetitions. 
    N_REPEATS_NN = 3


    for reg_param in reg_parameters:
        performance_tr = 0
        performance_te = 0
        for i in range(0, N_REPEATS_NN):

            class_train_y, class_test_y, class_train_prob_y, class_test_prob_y = learner.feedforward_neural_network(
                train_X, train_y,
                test_X, hidden_layer_sizes=(250, ), alpha=reg_param, max_iter=500,
                gridsearch=False
            )

            performance_tr += eval.accuracy(train_y, class_train_y)
            performance_te += eval.accuracy(test_y, class_test_y)
        performance_training.append(performance_tr/N_REPEATS_NN)
        performance_test.append(performance_te/N_REPEATS_NN)
    DataViz.plot_xy(x=[reg_parameters, reg_parameters], y=[performance_training, performance_test], method='semilogx',
                    xlabel='regularization parameter value', ylabel='accuracy', ylim=[0.95, 1.01],
                    names=['training', 'test'], line_styles=['r-', 'b:'])

    #Second, let us consider the influence of certain parameter settings for the tree model. (very related to the
    #regularization) and study the impact on performance.

    leaf_settings = [1,2,5,10]
    performance_training = []
    performance_test = []

    for no_points_leaf in leaf_settings:

        class_train_y, class_test_y, class_train_prob_y, class_test_prob_y = learner.decision_tree(
            train_X[selected_features], train_y, test_X[selected_features], min_samples_leaf=no_points_leaf,
            gridsearch=False, print_model_details=False)

        performance_training.append(eval.accuracy(train_y, class_train_y))
        performance_test.append(eval.accuracy(test_y, class_test_y))

    DataViz.plot_xy(x=[leaf_settings, leaf_settings], y=[performance_training, performance_test],
                    xlabel='minimum number of points per leaf', ylabel='accuracy',
                    names=['training', 'test'], line_styles=['r-', 'b:'])

    # So yes, it is important :) Therefore we perform grid searches over the most important parameters, and do so by means
    # of cross validation upon the training set.

    possible_feature_sets = [basic_features, features_after_chapter_3, features_after_chapter_4, features_after_chapter_5, selected_features]
    feature_names = ['initial set', 'Chapter 3', 'Chapter 4', 'Chapter 5', 'Selected features']
    N_KCV_REPEATS = 5


    print('Preprocessing took', time.time()-start, 'seconds.')

    scores_over_all_algs = []

    for i in range(0, len(possible_feature_sets)):
        selected_train_X = train_X[possible_feature_sets[i]]
        selected_test_X = test_X[possible_feature_sets[i]]

        # First we run our non deterministic classifiers a number of times to average their score.

        performance_tr_nn = 0
        performance_tr_rf = 0
        performance_tr_svm = 0
        performance_te_nn = 0
        performance_te_rf = 0
        performance_te_svm = 0

        for repeat in range(0, N_KCV_REPEATS):
            print("Training NeuralNetwork run {} / {} ... ".format(repeat, N_KCV_REPEATS, feature_names[i]))
            class_train_y, class_test_y, class_train_prob_y, class_test_prob_y = learner.feedforward_neural_network(
                selected_train_X, train_y, selected_test_X, gridsearch=True
            )
            print("Training RandomForest run {} / {} ... ".format(repeat, N_KCV_REPEATS, feature_names[i]))
            performance_tr_nn += eval.accuracy(train_y, class_train_y)
            performance_te_nn += eval.accuracy(test_y, class_test_y)
            
            class_train_y, class_test_y, class_train_prob_y, class_test_prob_y = learner.random_forest(
                selected_train_X, train_y, selected_test_X, gridsearch=True
            )
            
            performance_tr_rf += eval.accuracy(train_y, class_train_y)
            performance_te_rf += eval.accuracy(test_y, class_test_y)

            print("Training SVM run {} / {}, featureset: {}... ".format(repeat, N_KCV_REPEATS, feature_names[i]))
        
            class_train_y, class_test_y, class_train_prob_y, class_test_prob_y = learner.support_vector_machine_with_kernel(
                selected_train_X, train_y, selected_test_X, gridsearch=True
            )
            performance_tr_svm += eval.accuracy(train_y, class_train_y)
            performance_te_svm += eval.accuracy(test_y, class_test_y)

        
        overall_performance_tr_nn = performance_tr_nn/N_KCV_REPEATS
        overall_performance_te_nn = performance_te_nn/N_KCV_REPEATS
        overall_performance_tr_rf = performance_tr_rf/N_KCV_REPEATS
        overall_performance_te_rf = performance_te_rf/N_KCV_REPEATS
        overall_performance_tr_svm = performance_tr_svm/N_KCV_REPEATS
        overall_performance_te_svm = performance_te_svm/N_KCV_REPEATS

    #     #And we run our deterministic classifiers:
        print("Determenistic Classifiers:")

        print("Training Nearest Neighbor run 1 / 1, featureset {}:".format(feature_names[i]))
        class_train_y, class_test_y, class_train_prob_y, class_test_prob_y = learner.k_nearest_neighbor(
            selected_train_X, train_y, selected_test_X, gridsearch=True
        )
        performance_tr_knn = eval.accuracy(train_y, class_train_y)
        performance_te_knn = eval.accuracy(test_y, class_test_y)
        print("Training Descision Tree run 1 / 1  featureset {}:".format(feature_names[i]))
        class_train_y, class_test_y, class_train_prob_y, class_test_prob_y = learner.decision_tree(
            selected_train_X, train_y, selected_test_X, gridsearch=True
        )
        
        performance_tr_dt = eval.accuracy(train_y, class_train_y)
        performance_te_dt = eval.accuracy(test_y, class_test_y)
        print("Training Naive Bayes run 1/1 featureset {}:".format(feature_names[i]))
        class_train_y, class_test_y, class_train_prob_y, class_test_prob_y = learner.naive_bayes(
            selected_train_X, train_y, selected_test_X
        )
    
        performance_tr_nb = eval.accuracy(train_y, class_train_y)
        performance_te_nb = eval.accuracy(test_y, class_test_y)

        scores_with_sd = util.print_table_row_performances(feature_names[i], len(selected_train_X.index), len(selected_test_X.index), [
                                                                                                    (overall_performance_tr_nn, overall_performance_te_nn),
                                                                                                    (overall_performance_tr_rf, overall_performance_te_rf),
                                                                                                    (overall_performance_tr_svm, overall_performance_te_svm),
                                                                                                    (performance_tr_knn, performance_te_knn),
                                                                                                    (performance_tr_knn, performance_te_knn),
                                                                                                    (performance_tr_dt, performance_te_dt),
                                                                                                    (performance_tr_nb, performance_te_nb)])
        scores_over_all_algs.append(scores_with_sd)

    DataViz.plot_performances_classification(['NN', 'RF','SVM', 'KNN', 'DT', 'NB'], feature_names, scores_over_all_algs)

    # # And we study two promising ones in more detail. First, let us consider the decision tree, which works best with the
    # # selected features.

    class_train_y, class_test_y, class_train_prob_y, class_test_prob_y = learner.decision_tree(train_X[selected_features], train_y, test_X[selected_features],
                                                                                            gridsearch=True,
                                                                                            print_model_details=True, export_tree_path=EXPORT_TREE_PATH)

    class_train_y, class_test_y, class_train_prob_y, class_test_prob_y = learner.random_forest(
        train_X[selected_features], train_y, test_X[selected_features],
        gridsearch=True, print_model_details=True)

    test_cm = eval.confusion_matrix(test_y, class_test_y, class_train_prob_y.columns)

    DataViz.plot_confusion_matrix(test_cm, class_train_prob_y.columns, normalize=False)