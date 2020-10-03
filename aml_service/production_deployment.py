import argparse
import azureml.core
from azureml.core import Workspace, Experiment, Run
from azureml.core.model import Model
from azureml.core import Environment
from azureml.core.model import InferenceConfig
from azureml.core.compute import AksCompute, ComputeTarget
from azureml.core.webservice import Webservice, AksWebservice
from azureml.core.authentication import AzureCliAuthentication
import json
import os
import sys

print("In production_deployment.py")
print("Azure Python SDK version: ", azureml.core.VERSION)

print('Opening eval_info.json...')
eval_filepath = os.path.join('outputs', 'eval_info.json')

try:
    with open(eval_filepath) as f:
        eval_info = json.load(f)
        print('eval_info.json loaded')
        print(eval_info)
except:
    print("Cannot open: ", eval_filepath)
    print("Exiting...")
    sys.exit(0)

deploy_model = eval_info["deploy_model"]
if deploy_model == False:
    print('Model metric did not meet the metric threshold criteria and will not be deployed!')
    print('Exiting')
    sys.exit(0)

model_name = eval_info["model_name"]
model_description = eval_info['model_description']
model_version = eval_info["model_version"]
model_path = eval_info["model_path"]
model_acc = eval_info["model_acc"]
deployed_model_acc = eval_info["deployed_model_acc"]
train_run_id = eval_info['train_run_id']
eval_run_id = eval_info['eval_run_id']

print('Moving forward with deployment...')

parser = argparse.ArgumentParser("deploy")
parser.add_argument("--service_name", type=str, help="service name", dest="service_name", required=True)
parser.add_argument("--aks_name", type=str, help="aks name", dest="aks_name", required=True)
parser.add_argument("--aks_region", type=str, help="aks region", dest="aks_region", required=True)
parser.add_argument("--description", type=str, help="description", dest="description", required=True)
args = parser.parse_args()

print("Argument 1: %s" % args.service_name)
print("Argument 2: %s" % args.aks_name)
print("Argument 3: %s" % args.aks_region)
print("Argument 4: %s" % args.description)

script_dir = os.path.join('aml_service', 'image_files')
conda_file_path = os.path.join('aml_service', 'dependencies.yml')

print('creating AzureCliAuthentication...')
cli_auth = AzureCliAuthentication()
print('done creating AzureCliAuthentication!')

print('get workspace...')
ws = Workspace.from_config(auth=cli_auth)
print('done getting workspace!')

print('get model...')
model = Model(ws, name=model_name, version=model_version)
print('save the training dataset...')
model.datasets['training data'][0].to_pandas_dataframe().to_csv(os.path.join(script_dir, 'training_data.csv'), index=False)

print('Updating scoring file with the correct model name')
with open(os.path.join(script_dir, 'score.py')) as f:
    data = f.read()
with open(os.path.join(script_dir, 'score_fixed.py'), "w") as f:
    f.write(data.replace('MODEL-NAME', model_name)) #replace the placeholder MODEL-NAME
    print('score_fixed.py saved')

aks_name = args.aks_name 
aks_region = args.aks_region
aks_service_name = args.service_name
description = args.description

try:
    service = Webservice(name=aks_service_name, workspace=ws)
    print("Deleting AKS service {}".format(aks_service_name))
    service.delete()
except:
    print("No existing webservice found: ", aks_service_name)

compute_list = ws.compute_targets
aks_target = None
if aks_name in compute_list:
    aks_target = compute_list[aks_name]
    
if aks_target == None:
    print("No AKS found. Creating new Aks: {} and AKS Webservice: {}".format(aks_name, aks_service_name))
    prov_config = AksCompute.provisioning_configuration(location=aks_region)
    # Create the cluster
    aks_target = ComputeTarget.create(workspace=ws, name=aks_name, provisioning_configuration=prov_config)
    aks_target.wait_for_completion(show_output=True)
    print(aks_target.provisioning_state)
    print(aks_target.provisioning_errors)

print("Creating new webservice")
myEnv = Environment.from_conda_specification(aks_service_name + '-env', conda_file_path)
myEnv.register(workspace=ws)
inference_config = InferenceConfig(entry_script='score_fixed.py', source_directory=script_dir, environment=myEnv)

aks_config = AksWebservice.deploy_configuration(description = description, 
                                                tags = {'name': aks_name, 
                                                        'model_name': model.name, 
                                                        'model_version': model.version, 
                                                        'run_id': train_run_id})
service = Model.deploy(workspace=ws,
                       name=aks_service_name,
                       models=[model],
                       inference_config=inference_config,
                       deployment_config=aks_config, 
                       deployment_target=aks_target, 
                       overwrite=True)
service.wait_for_deployment(show_output=True)
print(service.state)

api_key, _ = service.get_keys()
print("Deployed AKS Webservice: {} \nWebservice Uri: {} \nWebservice API Key: {}".
      format(service.name, service.scoring_uri, api_key))

aks_webservice = {}
aks_webservice["aks_service_name"] = service.name
aks_webservice["aks_service_url"] = service.scoring_uri
aks_webservice["aks_service_api_key"] = api_key
print("AKS Webservice Info")
print(aks_webservice)

print("Saving aks_webservice.json...")
os.makedirs('./outputs', exist_ok=True)
aks_webservice_filepath = os.path.join('./outputs', 'aks_webservice.json')
with open(aks_webservice_filepath, "w") as f:
    json.dump(aks_webservice, f)
print("Done saving aks_webservice.json!")

# Single test data
test_data = ['manufactured in 2016 made of plastic in good condition']

# Call the webservice to make predictions on the test data
prediction = service.run(json.dumps(test_data))
print('Test data prediction: ', prediction)

