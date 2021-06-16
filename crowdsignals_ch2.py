##############################################################
#                                                            #
#    Mark Hoogendoorn and Burkhardt Funk (2017)              #
#    Machine Learning for the Quantified Self                #
#    Springer                                                #
#    Chapter 2                                               #
#                                                            #
##############################################################

# Import the relevant classes.
from scipy.sparse import data
from Chapter2.CreateDataset import CreateDataset
from util.VisualizeDataset import VisualizeDataset
from util import util
from pathlib import Path
import copy
import os
import sys

# Chapter 2: Initial exploration of the dataset.

SUBJECT_NAME = 'jeremy'
DATASET_PATH = Path('./datasets/')
RESULT_PATH = Path('./intermediate_datafiles/')
RESULT_FNAME = 'HAR_2_' + SUBJECT_NAME + '_result.csv'

# Set a granularity (the discrete step size of our time series data). We'll use a course-grained granularity of one
# instance per minute, and a fine-grained one with four instances per second.
GRANULARITIES = [6000, 1000]

# We can call Path.mkdir(exist_ok=True) to make any required directories if they don't already exist.
[path.mkdir(exist_ok=True, parents=True) for path in [DATASET_PATH, RESULT_PATH]]


datasets = []
for milliseconds_per_instance in GRANULARITIES:
    print(f'Creating numerical datasets from files in {DATASET_PATH} using granularity {milliseconds_per_instance}.')

    # Create an initial dataset object with the base directory for our data and a granularity
    dataset = CreateDataset(DATASET_PATH, milliseconds_per_instance)

    # Add the selected measurements to it.

    # We add the accelerometer data (continuous numerical measurements) of the phone and the smartwatch
    # and aggregate the values per timestep by averaging the values
    dataset.add_numerical_dataset(SUBJECT_NAME + '.csv', 'raw_timestamp',
                    ['roll_belt','pitch_belt','yaw_belt','total_accel_belt','gyros_belt_x',
                    'gyros_belt_y','gyros_belt_z','accel_belt_x','accel_belt_y','accel_belt_z',
                    'magnet_belt_x','magnet_belt_y','magnet_belt_z','roll_arm','pitch_arm','yaw_arm','total_accel_arm',
                    'gyros_arm_x','gyros_arm_y','gyros_arm_z','accel_arm_x','accel_arm_y','accel_arm_z','magnet_arm_x',
                    'magnet_arm_y','magnet_arm_z','roll_dumbbell','pitch_dumbbell','yaw_dumbbell','total_accel_dumbbell',
                    'gyros_dumbbell_x','gyros_dumbbell_y','gyros_dumbbell_z','accel_dumbbell_x','accel_dumbbell_y',
                    'accel_dumbbell_z','magnet_dumbbell_x','magnet_dumbbell_y','magnet_dumbbell_z','roll_forearm','pitch_forearm',
                    'yaw_forearm','total_accel_forearm','gyros_forearm_x','gyros_forearm_y','gyros_forearm_z','accel_forearm_x',
                    'accel_forearm_y','accel_forearm_z','magnet_forearm_x','magnet_forearm_y','magnet_forearm_z'],
                    
                    'avg', '')
 
    # We add the labels provided by the users. These are categorical events that might overlap. We add them
    # as binary attributes (i.e. add a one to the attribute representing the specific value for the label if it
    # occurs within an interval).
    dataset.add_event_dataset('labels_{}.csv'.format(SUBJECT_NAME), 'start_time', 'end_time', 'label', 'binary')

    # Get the resulting pandas data table
    dataset = dataset.data_table

    # Plot the data
    DataViz = VisualizeDataset('HAR_2_{}_g{}'.format(SUBJECT_NAME, milliseconds_per_instance))

    # Boxplot
    DataViz.plot_dataset_boxplot(dataset, ['roll_belt','pitch_belt','yaw_belt'])

    # Plot all data
    DataViz.plot_dataset(dataset, ['gyros_belt', 'accel_belt', 'magnet_belt',
                                  'roll_belt','pitch_belt','yaw_belt',
                                  'label'],

                                  ['like', 'like', 'like',
                                  'exact', 'exact', 'exact',
                                  'like'],

                                  ['line', 'line', 'line',
                                  'line', 'line', 'line',
                                  'points'])

    DataViz.plot_dataset(dataset, ['gyros_arm', 'accel_arm', 'magnet_arm',
                                  'roll_arm', 'pitch_arm', 'yaw_arm',
                                  'label'],

                                  ['like', 'like', 'like',
                                  'exact', 'exact', 'exact',
                                  'like'],

                                  ['line', 'line', 'line',
                                  'line', 'line', 'line',
                                  'points'])

    DataViz.plot_dataset(dataset, ['gyros_dumbbell', 'accel_dumbbell', 'magnet_dumbbell',
                                  'roll_dumbbell', 'pitch_dumbbell', 'yaw_dumbbell',
                                  'label'],

                                  ['like', 'like', 'like',
                                  'exact', 'exact', 'exact',
                                  'like'],

                                  ['line', 'line', 'line',
                                  'line', 'line', 'line',
                                  'points'])

    DataViz.plot_dataset(dataset, ['gyros_forearm', 'accel_forearm', 'magnet_forearm',
                                  'roll_forearm', 'pitch_forearm', 'yaw_forearm',
                                  'label'],

                                  ['like', 'like', 'like',
                                  'exact', 'exact', 'exact',
                                  'like'],

                                  ['line', 'line', 'line',
                                  'line', 'line', 'line',
                                  'points'])

    # And print a summary of the dataset.
    util.print_statistics(dataset)
    datasets.append(copy.deepcopy(dataset))

    # If needed, we could save the various versions of the dataset we create in the loop with logical filenames:
    dataset.to_csv(RESULT_PATH / 'HAR_2_{}_g{}.csv'.format(SUBJECT_NAME, milliseconds_per_instance))


# Make a table like the one shown in the book, comparing the two datasets produced.
# util.print_latex_table_statistics_two_datasets(datasets[0], datasets[1])

# Finally, store the last dataset we generated (last granularity).
# dataset.to_csv(RESULT_PATH / RESULT_FNAME)