import os

from utils import connectWithAzure
from azureml.core import Model

from dotenv import load_dotenv

# When you work locally, you can use a .env file to store all your environment variables.
# This line read those in.
load_dotenv()

MODEL_NAME = os.environ.get('MODEL_NAME')
LOCAL_MODEL_PATH = os.environ.get('LOCAL_MODEL_PATH')

def downloadLatestModel(ws):

    model_name = MODEL_NAME
    local_model_path = LOCAL_MODEL_PATH

    model = Model(ws, name = model_name)
    
    model.download(local_model_path, exist_ok=True)
    return model

def main():
    ws = connectWithAzure()

    print(os.environ)

    model = downloadLatestModel(ws)
    print(model)

if __name__ == '__main__':
    main()