from azureml.core import Workspace, Model, Environment
from azureml.core.compute import ComputeTarget
from azureml.core.model import InferenceConfig
from azureml.core.webservice import AksWebservice


ws = Workspace(
    subscription_id='f2f70602-65d9-479b-8014-a1c92a06ef0e',
    resource_group='Learn_MLOps',
    workspace_name='MLOps'
)

models = [
    Model(ws, 'support-vector-classifier'),
    Model(ws, 'scaler'),
]

env = Environment.from_conda_specification(name='weather-governance-env', file_path='conda_env.yml')
inference_config = InferenceConfig(entry_script='score.py', environment=env)

aks_target = ComputeTarget(workspace=ws, name='port-aks')
deploy_config = AksWebservice.deploy_configuration(
    cpu_cores=0.5,
    memory_gb=0.5,
    auth_enabled=True,
    enable_app_insights=True,
    autoscale_enabled=True,
    autoscale_min_replicas=1,
    autoscale_max_replicas=3,
    autoscale_target_utilization=70
)

print('Deploying to port-aks...')
service = Model.deploy(
    workspace=ws,
    name='weather-governance-prediction',
    models=models,
    inference_config=inference_config,
    deployment_config=deploy_config,
    deployment_target=aks_target,
    overwrite=True
)

service.wait_for_deployment(show_output=True)
print(f'Service state:  {service.state}')
print(f'Scoring URI:    {service.scoring_uri}')
print(f'Swagger URI:    {service.swagger_uri}')
