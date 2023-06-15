import os
import argparse
import pandas as pd
from glob import glob
import joblib

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# This AzureML package will allow to log our metrics etc.
from azureml.core import Run

parser = argparse.ArgumentParser()
parser.add_argument('--train-folder', type=str, dest='train_folder', help='Train folder mounting point.')
parser.add_argument('--test-folder', type=str, dest='test_folder', help='Test folder mounting point.')
parser.add_argument('--train-datasheet', type=str, dest='train_datasheet', help='Name of train datasheet.')
parser.add_argument('--test-datasheet', type=str, dest='test_datasheet', help='Name of test datasheet.')
parser.add_argument('--model-name', type=str, dest='model_name', help='The name of the model to use.')
args = parser.parse_args()

# As we're mounting the train_folder and test_folder onto the `/mnt/data` directories, we can load in the datasheets by using glob.
# A folder is like /mnt/azureml/cr/j/ff157465348a45d9a582392b62d83ca9/exe/wd/./processed_data/train

# Read the train datasheet contents into a Pandas dataframe.
datasheets = glob("./processed_data/train/*.csv")
df_train = None

for datasheet in datasheets:
    filename = os.path.basename(datasheet)
    if (args.train_datasheet in filename):
        print(f"Found train datasheet: {datasheet}")

        df_train = pd.read_csv(datasheet)
        print(df_train.head())
        continue

# Read the test datasheet contents into a Pandas dataframe.
datasheets = glob("./processed_data/test/*.csv")
df_test = None

for datasheet in datasheets:
    filename = os.path.basename(datasheet)
    if (args.test_datasheet in filename):
        print(f"Found test datasheet: {datasheet}")

        df_test = pd.read_csv(datasheet)
        print(df_test.head())
        continue
 

## START OUR RUN context.
## We can now log interesting information to Azure, by using these methods.
run = Run.get_context()

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

    accuracy =accuracy_score(y_test, y_pred)
    run.log('Accuracy score', accuracy)
    print(f'Accuracy score: {accuracy_score(y_test, y_pred):.4f}')

    # 2. confusion matrix
    print(f'Confusion matrix: \n{confusion_matrix(y_test, y_pred)}')

    # 3. ROC AUC score
    print(f'ROC AUC score: {roc_auc_score(y_test, y_pred_proba):.4f}')

    # 4. classification report
    print(f'Classification report: \n{classification_report(y_test, y_pred)}')


# Create an output directory where our AI model will be saved to.
# Everything inside the `outputs` directory will be logged and kept aside for later usage.
model_path = os.path.join('outputs', args.model_name)
os.makedirs(model_path, exist_ok=True)

# save model in the outputs folder so it automatically get uploaded

model_filename = os.path.join(model_path, 
    f'pipeline_logistic_regression.pkl')
joblib.dump(value=model, filename=model_filename)
print(f"Model saved at {model_filename}")

print("DONE TRAINING")
