import argparse
import azureml.core
from azureml.core import Workspace, Experiment, Run
from azureml.core.model import Model
from azureml.core import Environment
from azureml.core.model import InferenceConfig
from azureml.core.webservice import AciWebservice, Webservice
from azureml.core.authentication import AzureCliAuthentication
import json
import os
import sys

print("In test_deployment.py")
print("Azure Python SDK version: ", azureml.core.VERSION)

parser = argparse.ArgumentParser("test_deployment")
parser.add_argument("--model_name", type=str, help="model name", dest="model_name", required=True)
parser.add_argument("--service_name", type=str, help="service name", dest="service_name", required=True)
parser.add_argument("--description", type=str, help="description", dest="description", required=True)
args = parser.parse_args()

print("Argument 1: %s" % args.model_name)
print("Argument 2: %s" % args.service_name)
print("Argument 3: %s" % args.description)

script_dir = os.path.join('aml_service', 'image_files')
conda_file_path = os.path.join('aml_service', 'dependencies.yml')

print('creating AzureCliAuthentication...')
cli_auth = AzureCliAuthentication()
print('done creating AzureCliAuthentication!')

print('get workspace...')
ws = Workspace.from_config(auth=cli_auth)
print('done getting workspace!')

print('get model...')
model_name = args.model_name
model = Model(ws, name=model_name)
train_run_id = model.tags['run_id']
print('Model training run id:', train_run_id)

print('get training dataset...')
ds = model.datasets['training data'][0].to_pandas_dataframe()

print('save the training dataset...')
ds.to_csv(os.path.join(script_dir, 'training_data.csv'), index=False)

print('Updating scoring file with the correct model name')
with open(os.path.join(script_dir, 'score.py')) as f:
    data = f.read()
with open(os.path.join(script_dir, 'score_fixed.py'), "w") as f:
    f.write(data.replace('MODEL-NAME', model_name)) #replace the placeholder MODEL-NAME
    print('score_fixed.py saved')

aci_service_name = args.service_name
description = args.description

print("Creating new test ACI webservice")
myEnv = Environment.from_conda_specification(aci_service_name + '-env', conda_file_path)
myEnv.register(workspace=ws)
inference_config = InferenceConfig(entry_script='score_fixed.py', source_directory=script_dir, environment=myEnv)

aci_config = AciWebservice.deploy_configuration(
                        cpu_cores=3, 
                        memory_gb=15, 
                        location='eastus', 
                        description=description, 
                        auth_enabled=True, 
						tags = {'name': 'ACI container', 
                                'model_name': model.name, 
                                'model_version': model.version, 
                                'run_id': train_run_id
                                }
                        )

service = Model.deploy(workspace=ws,
                       name=aci_service_name,
                       models=[model],
                       inference_config=inference_config,
                       deployment_config= aci_config, 
                       overwrite=True)
service.wait_for_deployment(show_output=True)
print(service.state)

api_key, _ = service.get_keys()
print("Deployed ACI test Webservice: {} \nWebservice Uri: {} \nWebservice API Key: {}".
      format(service.name, service.scoring_uri, api_key))

aci_webservice = {}
aci_webservice["aci_service_name"] = service.name
aci_webservice["aci_service_url"] = service.scoring_uri
aci_webservice["aci_service_api_key"] = api_key
print("ACI test Webservice Info")
print(aci_webservice)

print("Saving aci_test_webservice.json...")
os.makedirs('./outputs', exist_ok=True)
aci_webservice_filepath = os.path.join('./outputs', 'aci_test_webservice.json')
with open(aci_webservice_filepath, "w") as f:
    json.dump(aci_webservice, f)
print("Done saving aci_test_webservice.json!")

# Single test data
test_data =[['manufactured in 2016 made of plastic in good condition']]

# Call the webservice to make predictions on the test data
prediction = service.run(json.dumps(test_data))
print('Test data prediction: ', prediction)

