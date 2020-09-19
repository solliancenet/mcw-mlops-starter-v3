import argparse
import os, json, sys
import azureml.core
from azureml.core import Workspace
from azureml.core import Experiment
from azureml.core.model import Model
from azureml.core.dataset import Dataset
from azureml.core.datastore import Datastore
from azureml.core import Run
from azureml.pipeline.core import PipelineRun
from azureml.core.webservice import AciWebservice, Webservice

print("In evaluate.py")

parser = argparse.ArgumentParser("evaluate")

parser.add_argument("--model_name", type=str, help="model name", dest="model_name", required=True)
parser.add_argument("--build_number", type=str, help="build number", dest="build_number", required=True)
parser.add_argument("--input", type=str, help="directory for saved model", dest="input", required=True)
parser.add_argument("--output", type=str, help="eval output directory", dest="output", required=True)

args = parser.parse_args()

print("Argument 1: %s" % args.model_name)
print("Argument 2: %s" % args.build_number)
print("Argument 3: %s" % args.input)
print("Argument 4: %s" % args.output)

run = Run.get_context()
ws = run.experiment.workspace

print('Workspace configuration succeeded')

train_filepath = os.path.join(args.input, 'train_info.json')
with open(train_filepath) as f:
    train_info = json.load(f)

#pipeline_run = PipelineRun(run.experiment, run_id = run.properties['azureml.pipelinerunid'])
#latest_model_run = Run(run.experiment, run_id = pipeline_run.find_step_run('train')[0].id)
latest_model_run_id = train_info['train_run_id']
latest_model_run = Run(run.experiment, run_id=latest_model_run_id)
print('Latest model run id: ', latest_model_run.id)
latest_model_accuracy = latest_model_run.get_metrics().get("acc")
print('Latest model accuracy: ', latest_model_accuracy)

ws_list = Webservice.list(ws, model_name = args.model_name)
print('webservice list')
print(ws_list)

deploy_model = False
current_model_run_id = None

if(len(ws_list) > 0):
    webservice = None
    for i in range(len(ws_list)):
        if ws_list[i].compute_type != 'ACI':
            webservice = ws_list[i]
    if webservice != None:
        current_model_run_id = webservice.tags.get("run_id")
        print('Found current deployed model run id:', current_model_run_id)
    else:
        deploy_model = True
        print('No deployed production webservice for model: ', args.model_name)
else:
    deploy_model = True
    print('No deployed webservice for model: ', args.model_name)

current_model_accuracy = -1 # undefined
if current_model_run_id != None:
    current_model_run = Run(run.experiment, run_id = current_model_run_id)
    current_model_accuracy = current_model_run.get_metrics().get("acc")
    print('accuracies')
    print(latest_model_accuracy, current_model_accuracy)
    if latest_model_accuracy > current_model_accuracy:
        deploy_model = True
        print('Current model performs better and will be deployed!')
    else:
        print('Current model does NOT perform better and thus will NOT be deployed!')

eval_info = {}
eval_info["model_acc"] = latest_model_accuracy
eval_info["deployed_model_acc"] = current_model_accuracy
eval_info["deploy_model"] = deploy_model
eval_info['train_run_id'] = latest_model_run_id
eval_info['eval_run_id'] = run.id

if deploy_model:
    os.chdir(args.input)
    cardata_ds_name = 'connected_car_components'
    cardata_ds  = Dataset.get_by_name(workspace=ws, name=cardata_ds_name)
    glove_ds_name = 'glove_6B_100d'
    glove_ds = Dataset.get_by_name(workspace=ws, name=glove_ds_name)
    
    model_description = 'Deep learning model to classify the descriptions of car components as compliant or non-compliant.'
    model = Model.register(
        model_path='model.onnx',  # this points to a local file
        model_name=args.model_name,  # this is the name the model is registered as
        tags={"type": "classification", "run_id": latest_model_run_id, "build_number": args.build_number},
        description=model_description,
        workspace=run.experiment.workspace,
        datasets=[('training data', cardata_ds), ('embedding data', glove_ds)])
    
    print("Model registered: {} \nModel Description: {} \nModel Version: {}".format(model.name, 
                                                                                model.description, model.version))
    eval_info["model_name"] = model.name
    eval_info["model_description"] = model.description
    eval_info["model_version"] = model.version
    eval_info["model_path"] = Model.get_model_path(model.name, version=model.version, _workspace=ws)

os.makedirs(args.output, exist_ok=True)
eval_filepath = os.path.join(args.output, 'eval_info.json')
with open(eval_filepath, "w") as f:
    json.dump(eval_info, f)
print('eval_info.json saved!')
