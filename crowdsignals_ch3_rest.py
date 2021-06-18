##############################################################
#                                                            #
#    Mark Hoogendoorn and Burkhardt Funk (2017)              #
#    Machine Learning for the Quantified Self                #
#    Springer                                                #
#    Chapter 3                                               #
#                                                            #
##############################################################

from pathlib import Path
import pandas as pd
import argparse
import numpy as np

from util.VisualizeDataset import VisualizeDataset
from Chapter3.DataTransformation import LowPassFilter
from Chapter3.DataTransformation import PrincipalComponentAnalysis
from Chapter3.ImputationMissingValues import ImputationMissingValues
from Chapter3.KalmanFilters import KalmanFilters

# Set up the file names and locations.
GRANULARITIES = [50]
SUBJECT_NAMES = ['jeremy', 'adelmo', 'carlitos', 'charles', 'eurico', 'pedro']
DATA_PATH = Path('./intermediate_datafiles/')    


COLUMNS = ['roll_belt','pitch_belt','yaw_belt','total_accel_belt','gyros_belt_x',
            'gyros_belt_y','gyros_belt_z','accel_belt_x','accel_belt_y','accel_belt_z',
            'magnet_belt_x','magnet_belt_y','magnet_belt_z','total_accel_arm',
            'gyros_arm_x','gyros_arm_y','gyros_arm_z','accel_arm_x','accel_arm_y','accel_arm_z','magnet_arm_x',
            'magnet_arm_y','magnet_arm_z','roll_dumbbell','pitch_dumbbell','yaw_dumbbell','total_accel_dumbbell',
            'gyros_dumbbell_x','gyros_dumbbell_y','gyros_dumbbell_z','accel_dumbbell_x','accel_dumbbell_y',
            'accel_dumbbell_z','magnet_dumbbell_x','magnet_dumbbell_y','magnet_dumbbell_z','roll_forearm','pitch_forearm',
            'yaw_forearm','total_accel_forearm','gyros_forearm_x','gyros_forearm_y','gyros_forearm_z','accel_forearm_x',
            'accel_forearm_y','accel_forearm_z','magnet_forearm_x','magnet_forearm_y','magnet_forearm_z']


def print_flags():
    """
    Prints all entries in FLAGS variable.
    """
    for key, value in vars(FLAGS).items():
        print(key + ' : ' + str(value))


def main():

    print_flags()

    # Next, import the data from the specified location and parse the date index.
    try:
        dataset = pd.read_csv(Path(DATA_PATH / DATASET_FNAME), index_col=0)
        dataset.index = pd.to_datetime(dataset.index)
    except IOError as e:
        print('File not found, try to run previous crowdsignals scripts first!')
        raise e

    # We'll create an instance of our visualization class to plot the results.
    DataViz = VisualizeDataset('HAR_3_final_{}_g{}_{}'.format(SUBJECT_NAME, GRANULARITY, FLAGS.mode))
    # Compute the number of milliseconds covered by an instance based on the first two rows
    milliseconds_per_instance = (dataset.index[1] - dataset.index[0]).to_numpy() / np.timedelta64(1, 's')

    MisVal = ImputationMissingValues()
    LowPass = LowPassFilter()
    PCA = PrincipalComponentAnalysis()

    if FLAGS.mode == 'imputation':
        # Let us impute the missing values and plot an example.
       
        imputed_mean_dataset = MisVal.impute_mean(dataset.copy(), 'hr_watch_rate')       
        imputed_median_dataset = MisVal.impute_median(dataset.copy(), 'hr_watch_rate')
        imputed_interpolation_dataset = MisVal.impute_interpolate(dataset.copy(), 'hr_watch_rate')
        
        DataViz.plot_imputed_values(dataset, ['original', 'mean', 'median', 'interpolation'], 'hr_watch_rate',
                                    imputed_mean_dataset['hr_watch_rate'], 
                                    imputed_median_dataset['hr_watch_rate'],
                                    imputed_interpolation_dataset['hr_watch_rate'])

    elif FLAGS.mode == 'kalman':
        # Using the result from Chapter 2, let us try the Kalman filter on the acc_phone_x attribute and study the result.      # originally 'light_phone_lux' instead of 'acc_phone_x'
        try:
            original_dataset = pd.read_csv(
            DATA_PATH / ORIG_DATASET_FNAME, index_col=0)
            original_dataset.index = pd.to_datetime(original_dataset.index)
        except IOError as e:
            print('File not found, try to run previous crowdsignals scripts first!')
            raise e

        FEATURE_DESIRED = 'hr_watch_rate'   # added by Alex.

        KalFilter = KalmanFilters()
        kalman_dataset = KalFilter.apply_kalman_filter(
            original_dataset, FEATURE_DESIRED)      # originally 'acc_phone_x'
        DataViz.plot_imputed_values(kalman_dataset, [
                                    'original', 'kalman'], FEATURE_DESIRED, kalman_dataset[FEATURE_DESIRED + '_kalman'])     # originally 'acc_phone_x', kalman_dataset['acc_phone_x_kalman'])
        # DataViz.plot_dataset(kalman_dataset, [FEATURE_DESIRED, FEATURE_DESIRED + '_kalman'], [      # originally ['acc_phone_x', 'acc_phone_x_kalman'], [
        #                      'exact', 'exact'], ['line', 'line'])

        # We ignore the Kalman filter output for now...

    elif FLAGS.mode == 'lowpass':
        
        # Let us apply a lowpass filter and reduce the importance of the data above 1.5 Hz

        # Determine the sampling frequency.
        fs = float(1)/milliseconds_per_instance
        cutoff = 1.5
        # Let us study acc_phone_x:
        new_dataset = LowPass.low_pass_filter(dataset.copy(), 'roll_belt', fs, cutoff, order=10)
        DataViz.plot_dataset(new_dataset.iloc[int(0.4*len(new_dataset.index)):int(0.43*len(new_dataset.index)), :],
                             ['roll_belt', 'roll_belt_lowpass'], ['exact', 'exact'], ['line', 'line'])

    elif FLAGS.mode == 'PCA':

        #first impute again, as PCA can not deal with missing values       
        for col in [c for c in dataset.columns if not 'label' in c]:
            dataset = MisVal.impute_interpolate(dataset, col)

       
        selected_predictor_cols = [c for c in dataset.columns if (not ('label' in c)) and (not (dataset.loc[:,c] == 0).all())]
        pc_values = PCA.determine_pc_explained_variance(
            dataset, selected_predictor_cols)

        # Plot the variance explained.
        DataViz.plot_xy(x=[range(1, len(selected_predictor_cols)+1)], y=[pc_values],
                        xlabel='principal component number', ylabel='explained variance',
                        ylim=[0, 1], line_styles=['b-'])

        # We select 7 as the best number of PC's as this explains most of the variance

        n_pcs = 5

        dataset = PCA.apply_pca(dataset.copy(), selected_predictor_cols, n_pcs)

        # And we visualize the result of the PC's
        DataViz.plot_dataset(dataset, ['pca_', 'label'], [
                             'like', 'like'], ['line', 'points'])

    elif FLAGS.mode == 'final':
        # Now, for the final version. 
        # We first start with imputation by interpolation
       
        for col in [c for c in dataset.columns if not 'label' in c]:
            dataset = MisVal.impute_interpolate(dataset, col)

        # And now let us include all LOWPASS measurements that have a form of periodicity (and filter them):
        periodic_measurements = COLUMNS

        
        # Let us apply a lowpass filter and reduce the importance of the data above 1.5 Hz

        # Determine the sampling frequency.
        fs = float(1)/milliseconds_per_instance
        cutoff = 1

        for col in periodic_measurements:
            dataset = LowPass.low_pass_filter(dataset, col, fs, cutoff, order=10)
            dataset[col] = dataset[col + '_lowpass']
            del dataset[col + '_lowpass']

        # We used the optimal found parameter n_pcs = 7, to apply PCA to the final dataset
        selected_predictor_cols = [c for c in dataset.columns if (not ('label' in c)) and (not (dataset.loc[:,c] == 0).all())]
        
        n_pcs = 5
        
        dataset = PCA.apply_pca(dataset.copy(), selected_predictor_cols, n_pcs)

        # And the overall final dataset:
        # DataViz.plot_dataset(dataset, 
        #                     COLUMNS + ['pca_', 'label'],
        #                     ['exact'] * len(COLUMNS) + ['like', 'like'],
        #                     ['line'] * len(COLUMNS) + ['points', 'points'])

        # Store the final outcome.
        dataset.to_csv(DATA_PATH / RESULT_FNAME)


if __name__ == '__main__':
    # Command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='final',
                        help="Select what version to run: final, imputation, lowpass or PCA \
                        'lowpass' applies the lowpass-filter to a single variable \
                        'imputation' is used for the next chapter \
                        'PCA' is to study the effect of PCA and plot the results\
                        'final' is used for the next chapter", choices=['lowpass', 'imputation', 'PCA', 'final', 'kalman'])     # originally no 'kalman' in 'choices=' list

   
    FLAGS, unparsed = parser.parse_known_args()

    for SUBJECT_NAME in SUBJECT_NAMES:
        for GRANULARITY in GRANULARITIES:

            DATASET_FNAME = 'HAR_3_' + SUBJECT_NAME + '_g' + str(GRANULARITY) + '_result_outliers.csv'
            RESULT_FNAME = 'HAR_3_' + SUBJECT_NAME + '_g' + str(GRANULARITY) + '_result_final.csv'
            ORIG_DATASET_FNAME = 'HAR_2_' + SUBJECT_NAME + '_g' + str(GRANULARITY) + '.csv'
            
            main()