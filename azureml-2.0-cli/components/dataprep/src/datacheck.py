from glob import glob
import os
import argparse
import pandas as pd

def main():

    # input and output arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_data", type=str, help="input path to the raw datasheets")
    parser.add_argument("--checked_data", type=str, help="output path for checked datasheets")
    args = parser.parse_args()

    print(" ".join(f"{k}={v}" for k, v in vars(args).items()))

    print("input raw datasheets:", args.raw_data)
    print("output folder checked datasheets:", args.checked_data)

    datasheets = glob(f"{args.raw_data}/*.csv")

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
        df.to_csv(os.path.join(args.checked_data, checked_filename))

    print(f"Finished processing {len(datasheets)} datasheets")

if __name__ == "__main__":
    main()

