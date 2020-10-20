# tutorial/06-run-pytorch-data.py
from azureml.core import Workspace
from azureml.core import Experiment
from azureml.core import Environment
from azureml.core import ScriptRunConfig
from azureml.core import Dataset
from azureml.core.runconfig import DEFAULT_GPU_IMAGE

if __name__ == "__main__":
    ws = Workspace.from_config()
    
    datastore = ws.get_default_datastore()
    dataset = Dataset.File.from_files(path=(datastore, 'data'))

    experiment = Experiment(workspace=ws, name='trainmodel')

    config = ScriptRunConfig(
        source_directory='./src',
        script='train.py',
        compute_target='gpu-cluster',
        arguments=[
            '--data_path', dataset.as_named_input('input').as_mount()
            ],)
    # set up pytorch environment
    env = Environment.from_conda_specification(name='pytorch-env',file_path='.azureml/pytorch-env.yml')
    config.run_config.environment = env
    env.docker.base_image = DEFAULT_GPU_IMAGE

    run = experiment.submit(config)
    run.wait_for_completion(show_output=True)
    aml_url = run.get_portal_url()
    
    print("Submitted to an Azure Machine Learning compute cluster. Click on the link below")
    print("")
    print(aml_url)