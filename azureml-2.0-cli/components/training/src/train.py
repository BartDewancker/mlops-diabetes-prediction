from ast import mod
from glob import glob
import os
import argparse
import pandas as pd
import numpy as np
import joblib

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--train_folder', type=str, help='folder with train datasheets.')
    parser.add_argument('--test_folder', type=str, help='folder with test datasheets.')
    parser.add_argument('--train_datasheet', type=str, help='name of train datasheet.')
    parser.add_argument('--test_datasheet', type=str, help='name of test datasheet.')
    parser.add_argument('--model_name', type=str, help='name of the model to use.')
    parser.add_argument("--output_folder", type=str, help="output path for model")
    args = parser.parse_args()

    print(" ".join(f"{k}={v}" for k, v in vars(args).items()))

    print("input train datasheets:", args.train_folder)
    print("input test datasheets:", args.test_folder)
    print("train datasheet:", args.train_datasheet)
    print("test datasheet", args.test_datasheet)
    print("model name:", args.model_name)
    print("output folder for model:", args.output_folder)

    # Read the train datasheet contents into a Pandas dataframe.
    datasheets = glob(f"{args.train_folder}/*.csv")
    
    df_train = None

    for datasheet in datasheets:
        filename = os.path.basename(datasheet)
        if (args.train_datasheet in filename):
            print(f"Found train datasheet: {datasheet}")

            df_train = pd.read_csv(datasheet)
            print(df_train.head())
            continue

    # Read the test datasheet contents into a Pandas dataframe.
    datasheets = glob(f"{args.test_folder}/*.csv")
    df_test = None

    for datasheet in datasheets:
        filename = os.path.basename(datasheet)
        if (args.test_datasheet in filename):
            print(f"Found test datasheet: {datasheet}")

            df_test = pd.read_csv(datasheet)
            print(df_test.head())
            continue
    
    # create model as pipeline

    if(not df_train.empty and not df_test.empty):

        print("Datasheets loaded successfully.")

        # 1. create a pipeline for numeric features
        numeric_features = ['age', 'bmi', 'HbA1c_level', 'blood_glucose_level']
        numeric_transformer = Pipeline(steps=[
            ('scaler', StandardScaler())
        ])

        # 2. create a pipeline for categorical features
        categoricalFeatures = ['gender', 'smoking_history']
        categorical_transformer = Pipeline(steps=[
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])

        # 3. define boolean features, which will be passed through the pipeline
        booleanFeatures = ['hypertension', 'heart_disease']

        # 4. create a column transformer to combine both pipelines
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, categoricalFeatures),
                ('bool', 'passthrough', booleanFeatures),
            ],
            remainder='drop'
        )

        # 5. create a pipeline for the model
        model = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', LogisticRegression())
        ])

        # # configure model
        # model.set_params(
        #     classifier__C=10, 
        #     classifier__solver='liblinear', 
        #     classifier__max_iter=1000, 
        #     classifier__class_weight='balanced',
        #     preprocessor__num__scaler__with_mean=True)

        # print
        model

        c = 0.5
        solver = 'liblinear'
        # class_weight = {'class_label': 'balanced'}

        # configure model
        model.set_params(
            classifier__C=c,
            classifier__solver=solver,
            classifier__max_iter=1000,
            classifier__class_weight='balanced',
            preprocessor__num__scaler__with_mean=False)

        target = 'diabetes'

        # fit the model
        X_train = df_train.drop(target, axis=1)
        y_train = df_train[target]
        model.fit(X_train, y_train)

        # make predictions on the test set
        X_test = df_test.drop(target, axis=1)
        y_test = df_test[target]
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]

        # evaluate the model

        # 1. accuracy score
        print(f'Accuracy score: {accuracy_score(y_test, y_pred):.4f}')

        # 2. confusion matrix
        print(f'Confusion matrix: \n{confusion_matrix(y_test, y_pred)}')

        # 3. ROC AUC score
        print(f'ROC AUC score: {roc_auc_score(y_test, y_pred_proba):.4f}')

        # 4. classification report
        print(f'Classification report: \n{classification_report(y_test, y_pred)}')


    # Create an output directory where our AI model will be saved to.
    # Everything inside the `outputs` directory will be logged and kept aside for later usage.
    
    model_path = os.path.join(args.output_folder, 'outputs', args.model_name)
    os.makedirs(model_path, exist_ok=True)

    # save model in the outputs folder so it automatically get uploaded

    model_filename = os.path.join(model_path, f'pipeline_logistic_regression.pkl')
   
    joblib.dump(value=model, filename=model_filename)
    print(f"Model will be saved at {model_filename}")

    print("DONE TRAINING")

if __name__ == "__main__":
    main()
