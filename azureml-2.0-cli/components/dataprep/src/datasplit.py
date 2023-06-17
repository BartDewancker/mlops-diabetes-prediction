from glob import glob
import os
import argparse
import pandas as pd
from sklearn.model_selection import train_test_split

def main():

    # input and output arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--checked_data", type=str, help="input path to checked datasheets")
    parser.add_argument("--random_seed", type=int, help="random seed factor")
    parser.add_argument("--split_size", type=float, help="train test split factor")
    parser.add_argument("--train_data", type=str, help="output path for train datasheets")
    parser.add_argument("--test_data", type=str, help="output path for test datasheets")
    args = parser.parse_args()

    print(" ".join(f"{k}={v}" for k, v in vars(args).items()))

    print("input checked datasheets:", args.checked_data)
    print("split size:", args.split_size)
    print("random seed:", args.random_seed)
    print("output folder train datasheets:", args.train_data)
    print("output folder test datasheets:", args.test_data)
    
    datasheets = glob(f"{args.checked_data}/*.csv")

    print(f"Splitting {len(datasheets)} datasheets")

    for datasheet in datasheets:

        target = 'diabetes'
        
        # Read the checked datasheet contents into a Pandas dataframe
        df = pd.read_csv(datasheet)
        df.head()
        df.info()
    
        df_train, df_test = train_test_split(df, test_size=args.split_size, random_state=args.random_seed, shuffle=True, stratify=df[target])

        filename = os.path.basename(datasheet)

        # Remove _checked from the filename
        filename = filename.replace('_checked', '')
        
        # show the target distribution in train and test sets
        print('Train set:')
        print(df_train[target].value_counts(normalize=True))

        # Save the train datasheet to the train_data_folder.
        train_filename = os.path.splitext(filename)[0] + '_train.csv'
        df_train.to_csv(os.path.join(args.train_data, train_filename))

        print('Test set:')
        print(df_test[target].value_counts(normalize=True))

        # Save the test datasheet to the test_data_folder.
        test_filename = os.path.splitext(filename)[0] + '_test.csv'
        df_test.to_csv(os.path.join(args.test_data, test_filename))
        
if __name__ == "__main__":
    main()