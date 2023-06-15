from glob import glob
import os
from datetime import datetime
import shutil
from sys import version
import pandas as pd

from utils import connectWithAzure

from dotenv import load_dotenv
from azureml.core import Dataset
from azureml.data.datapath import DataPath
from sklearn.model_selection import train_test_split

# When you work locally, you can use a .env file to store all your environment variables.
# This line read those in.
load_dotenv()

SEED = int(os.environ.get('RANDOM_SEED'))
TRAIN_TEST_SPLIT_FACTOR = float(os.environ.get('TRAIN_TEST_SPLIT_FACTOR'))

RAW_DATASET = os.environ.get('RAW_DATASET_NAME')
CHECKED_DATASET = os.environ.get('CHECKED_DATASET_NAME')
TEST_DATASET = os.environ.get('TEST_DATASET_NAME')
TRAIN_DATASET = os.environ.get('TRAIN_DATASET_NAME')

DATASET_VERSION = os.environ.get('DATASET_VERSION')

GIT_SHA = os.environ.get('GIT_SHA')

# Path to store the downloaded dataset = data
data_folder = os.path.join(os.getcwd(), 'data')

 # Path to store the checked datasheets = data/checked
checked_data_folder = os.path.join(os.getcwd(), 'data', 'checked')

# Path to store the datasheets voor testing = data/test
test_data_folder = os.path.join(os.getcwd(), 'data', 'test')

# Path to store the datasheets voor training = data/train
train_data_folder = os.path.join(os.getcwd(), 'data', 'train')

def checkDatasheets(ws):

    os.makedirs(data_folder, exist_ok=True)
    os.makedirs(checked_data_folder, exist_ok=True)
    
    print(f'Downloading dataset with raw datasheets: {RAW_DATASET}')
    dataset = Dataset.get_by_name(ws, RAW_DATASET, version = DATASET_VERSION)
    dataset.download(data_folder, overwrite=True)

    datasheets = glob(f"{data_folder}/*.csv")

    print(f"Checking {len(datasheets)} datasheets")

    for datasheet in datasheets:
    
        # Read the datasheet contents into a Pandas dataframe.
        df = pd.read_csv(datasheet)
        df.head()
        df.info()

        # check for missing values
        print('****** missing values:')
        print(f'{df.isnull().sum()}')

        # drop missing 
        print(df.shape)
        df = df.dropna(axis=0)
        print(df.shape)

        # check for duplicate rows
        print('****** duplicate rows:')
        print(f'{df.duplicated().sum()}')

        # drop duplicates
        print(df.shape)
        df = df.drop_duplicates()
        print(df.shape)
    
        # check for unique values in each column
        for col in df.columns:
            if df[col].dtype == 'object':
                print(f'{col} : {df[col].unique()}')
            else:
                print(f'{col} : [{df[col].min()}, {df[col].max()}]')

        checked_filename = os.path.basename(datasheet)
        checked_filename = os.path.splitext(checked_filename)[0] + '_checked.csv'
        
        # Save the checked datasheet to the checked_data_folder.
        df.to_csv(os.path.join(checked_data_folder, checked_filename))

    print(f"Finished processing {len(datasheets)} datasheets")

    # Upload the directory with the checked datasheets as a new dataset.
    print(f'Uploading checked datasheets to {CHECKED_DATASET}')
    print(f'checked_data_folder: {checked_data_folder}')
    checked_datasheets = Dataset.File.upload_directory(src_dir = checked_data_folder,
                                                       target = DataPath(datastore=ws.get_default_datastore(), path_on_datastore=f'processed_data/checked'),
                                                       overwrite=True)

    # Register the checked datasheets as a new dataset.
    new_dataset = checked_datasheets.register(ws, name = CHECKED_DATASET,
                                              description =  f'Datasheets that have been checked for missing values, duplicates and unique values.',
                                              tags={'Dataset raw data': RAW_DATASET, 'AI-Model': 'LogisticRegression', 'GIT_SHA': GIT_SHA},
                                              create_new_version=True)
    
    print(f"Dataset id {new_dataset.id} | Dataset version {new_dataset.version}")

    # Remove the data folder and the checked_data_folder.
    shutil.rmtree(data_folder)

def splitDatasheets(ws):

    os.makedirs(data_folder, exist_ok=True)
    os.makedirs(test_data_folder, exist_ok=True)
    os.makedirs(train_data_folder, exist_ok=True)
    
    print(f'Downloading dataset {CHECKED_DATASET}')
    dataset = Dataset.get_by_name(ws, CHECKED_DATASET, version = DATASET_VERSION)
    dataset.download(data_folder, overwrite=True)

    datasheets = glob(f"{data_folder}/*.csv")

    print(f"Splitting {len(datasheets)} datasheets")

    for datasheet in datasheets:

        target = 'diabetes'
        
        # Read the checked datasheet contents into a Pandas dataframe
        df = pd.read_csv(datasheet)
        df.head()
        df.info()
    
        df_train, df_test = train_test_split(df, test_size=TRAIN_TEST_SPLIT_FACTOR, random_state=SEED, shuffle=True, stratify=df[target])

        filename = os.path.basename(datasheet)

        # Remove _checked from the filename
        filename = filename.replace('_checked', '')
        
        # show the target distribution in train and test sets
        print('Train set:')
        print(df_train[target].value_counts(normalize=True))

        # Save the train datasheet to the train_data_folder.
        train_filename = os.path.splitext(filename)[0] + '_train.csv'
        df_train.to_csv(os.path.join(train_data_folder, train_filename))

        print('Test set:')
        print(df_test[target].value_counts(normalize=True))

        # Save the test datasheet to the test_data_folder.
        test_filename = os.path.splitext(filename)[0] + '_test.csv'
        df_test.to_csv(os.path.join(test_data_folder, test_filename))
        
    # Upload the directory with the train datasheets as a new dataset.
    train_datasheets = Dataset.File.upload_directory(src_dir = train_data_folder,
                                                     target = DataPath(datastore=ws.get_default_datastore(),
                                                                       path_on_datastore=f'processed_data/train'), overwrite=True)

    # Register the train datasheets as a new dataset.
    train_dataset = train_datasheets.register(ws, name = TRAIN_DATASET,
                                              description =  f'Datasheets for training the model.',
                                              tags={'Dataset train data': TRAIN_DATASET, 'AI-Model': 'LogisticRegression', 'GIT_SHA': GIT_SHA},
                                              create_new_version=True)
    
    print(f"Dataset id {train_dataset.id} | Dataset version {train_dataset.version}")

     # Upload the directory with the test datasheets as a new dataset.
    test_datasheets = Dataset.File.upload_directory(src_dir = test_data_folder,
                                                     target = DataPath(datastore=ws.get_default_datastore(),
                                                                       path_on_datastore=f'processed_data/test'), overwrite=True)

    # Register the train datasheets as a new dataset.
    test_dataset = test_datasheets.register(ws, name = TEST_DATASET,
                                            description =  f'Datasheets for testing the model.',
                                            tags={'Dataset test data': TEST_DATASET, 'AI-Model': 'LogisticRegression', 'GIT_SHA': GIT_SHA},
                                            create_new_version=True)
    
    print(f"Dataset id {test_dataset.id} | Dataset version {test_dataset.version}")

    # Remove the data folder, the train_data_folder and the test_data_folder.
    shutil.rmtree(data_folder)
    

def main():
    ws = connectWithAzure()

    print('Processing the datasheets')
    checkDatasheets(ws)
   
    # print('Splitting the datasheet entries')
    splitDatasheets(ws)

if __name__ == '__main__':
    main()