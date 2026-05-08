import azureml.core
from azureml.core import Workspace
from azureml.core.compute import ComputeTarget
from azureml.core.compute_target import ComputeTargetException

ws = Workspace(
    subscription_id='f2f70602-65d9-479b-8014-a1c92a06ef0e',
    resource_group='Learn_MLOps',
    workspace_name='MLOps'
)
print(ws.name, ws.resource_group, ws.location, sep='\n')

aks_name = 'port-aks'

try:
    aks_target = ComputeTarget(workspace=ws, name=aks_name)
    print(f'Found existing cluster: {aks_name}')
except ComputeTargetException:
    print(f'Cluster {aks_name} not found.')

if aks_target.get_status() != "Succeeded":
    aks_target.wait_for_completion(show_output=True)

print(f'Cluster status: {aks_target.get_status()}')
print(f'Cluster ready for production deployments.')