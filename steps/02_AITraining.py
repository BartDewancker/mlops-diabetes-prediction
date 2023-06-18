from glob import glob
import os
from typing import Tuple, List
import shutil
import pandas as pd

from utils import connectWithAzure

from azureml.core import ScriptRunConfig, Experiment, Dataset
from azureml.core.compute import ComputeTarget, AmlCompute
from azureml.core.environment import Environment
from azureml.core.conda_dependencies import CondaDependencies
from dotenv import load_dotenv

# Read the .env file and store the values as environment variables.
load_dotenv()

MODEL_NAME = os.environ.get('MODEL_NAME')
MODEL_C_VALUE = float(os.environ.get('MODEL_C_VALUE'))
MODEL_MAX_ITER = int(os.environ.get('MODEL_MAX_ITER'))
TRAIN_DATASET = os.environ.get('TRAIN_DATASET_NAME')
TEST_DATASET = os.environ.get('TEST_DATASET_NAME')
TRAIN_DATASHEET = os.environ.get('TRAIN_DATASHEET_NAME')
TEST_DATASHEET = os.environ.get('TEST_DATASHEET_NAME')
DATASET_VERSION = os.environ.get('DATASET_VERSION')

CONDA_DEPENDENCIES = os.environ.get('CONDA_DEPENDENCIES_PATH')

EXPERIMENT = os.environ.get('EXPERIMENT_NAME')
SCRIPTFOLDER = os.environ.get('SCRIPT_FOLDER')
TRAINING_ENV = os.environ.get('TRAINING_ENV_NAME')
    
COMPUTE_NAME = os.environ.get("AML_COMPUTE_CLUSTER_NAME", "cpu-cluster")
COMPUTE_MIN_NODES = int(os.environ.get("AML_COMPUTE_CLUSTER_MIN_NODES", 0))
COMPUTE_MAX_NODES = int(os.environ.get("AML_COMPUTE_CLUSTER_MAX_NODES", 4))

# This example uses CPU VM. For using GPU VM, set SKU to STANDARD_NC6
VM_SIZE = os.environ.get("AML_COMPUTE_CLUSTER_SKU", "STANDARD_D2_V2")

GIT_SHA = os.environ.get('GIT_SHA')

def prepareComputeCluster(ws):
    
    if COMPUTE_NAME in ws.compute_targets:
        compute_target = ws.compute_targets[COMPUTE_NAME]
        if compute_target and type(compute_target) is AmlCompute:
            print("found compute target: " + COMPUTE_NAME)
    else:
        print("creating new compute target...")
        provisioning_config = AmlCompute.provisioning_configuration(vm_size = VM_SIZE,
                                                                    min_nodes = COMPUTE_MIN_NODES, 
                                                                    max_nodes = COMPUTE_MAX_NODES)

        # create the cluster
        compute_target = ComputeTarget.create(ws, COMPUTE_NAME, provisioning_config)
        
        # can poll for a minimum number of nodes and for a specific timeout. 
        # if no min node count is provided it will use the scale settings for the cluster
        compute_target.wait_for_completion(show_output=True, min_node_count=None, timeout_in_minutes=20)
        
        # For a more detailed view of current AmlCompute status, use get_status()
        print(compute_target.get_status().serialize())

    return compute_target

def prepareEnvironment(ws):

    # Create an Environment name for later use
    environment_name = TRAINING_ENV
   
    env = Environment.from_conda_specification(environment_name, file_path = CONDA_DEPENDENCIES)
    #env.python.user_managed_dependencies = False # False when training on local machine, otherwise True.
    
    # Register environment to re-use later
    env.register(workspace = ws)

    return env

def prepareTraining(ws, env, compute_target) -> Tuple[Experiment, ScriptRunConfig]:
    
    # Create a new experiment.
    exp = Experiment(workspace=ws, name=EXPERIMENT)

    train_dataset = Dataset.get_by_name(ws, TRAIN_DATASET, version = DATASET_VERSION)
    test_dataset = Dataset.get_by_name(ws, TEST_DATASET, version = DATASET_VERSION)

    args = [
        # You can set these to .as_mount() when not training on local machines, but this should also work.
    '--train-folder', train_dataset.as_download('./processed_data/train'),
    '--test-folder', test_dataset.as_download('./processed_data/test'),
    '--train-datasheet', TRAIN_DATASHEET,
    '--test-datasheet', TEST_DATASHEET,
    '--model-name', MODEL_NAME,
    '--classifier-c', MODEL_C_VALUE,
    '--classifier-max-iter', MODEL_MAX_ITER,
    '--git-sha', GIT_SHA]

    script_run_config = ScriptRunConfig(source_directory=SCRIPTFOLDER, script='train.py', arguments=args,
                                        compute_target=compute_target, environment=env)

    print('Run started!')

    return exp, script_run_config

def downloadAndRegisterModel(ws, run):

    model_path = 'outputs/' + MODEL_NAME

    test_dataset = Dataset.get_by_name(ws, TEST_DATASET, version = DATASET_VERSION)
      
    run.download_files(prefix=model_path)
    run.register_model(MODEL_NAME, model_path=model_path,
                       tags={'Patiens health data': TRAIN_DATASET, 'AI-Model': 'LogisticRegression', 'GIT_SHA': GIT_SHA},
                       description="Diabetes prediction",
                       sample_input_dataset=test_dataset)

def main():

    ws = connectWithAzure()

    compute_target = prepareComputeCluster(ws)

    # We can also run on the local machine if we set the compute_target to None. We specify this in an ENV variable as TRAIN_ON_LOCAL.
    # If you don't give this parameter, we are defaulting to False, which means we will not train on local
    environment = prepareEnvironment(ws)
    exp, config = prepareTraining(ws, environment, compute_target)
    #exp, config = prepareTraining(ws, environment, None)

    run = exp.submit(config=config)
    run.wait_for_completion(show_output=True) # We aren't going to show the training output, you can follow that on the Azure logs if you want to.
    print(f"Run {run.id} has finished.")

    downloadAndRegisterModel(ws, run)

if __name__ == '__main__':
    main()