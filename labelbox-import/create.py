from labelbox import Client
from labelbox import Project
from labelbox import Dataset
import json
import os

# from: https://labelbox.com/docs/python-api/model-assisted-labeling-python-script

print("Connecting to Labelbox...")
client = Client(os.environ.get("LABELBOX_API_KEY"))

print("Creating Project: {}...".format(os.environ.get("LABELBOX_PROJECT_NAME")))
new_project = client.create_project(name=os.environ.get("LABELBOX_PROJECT_NAME"))

print("Creating Dataset: {}...".format(os.environ.get("LABELBOX_DATASET_NAME")))
new_dataset = client.create_dataset(name=os.environ.get("LABELBOX_DATASET_NAME"), projects = new_project)
new_dataset.create_data_row(row_data="./sample.jpg")

all_frontends = list(client.get_labeling_frontends())
for frontend in all_frontends:
    if frontend.name == 'Editor':
        new_project_frontend = frontend
        break

new_project.labeling_frontend.connect(new_project_frontend)

print("Configuring Editor...")
ontology = {
    "classifications": [
        {"required": False,
        "instructions": "Model",
        "name": "model",
        "type": "text",
        "options":[]},
        {"required": False,
        "instructions": "Operator",
        "name": "operator",
        "type": "text",
        "options":[]},
        {"required": False,
        "instructions": "Manufacturer",
        "name": "manufacturer",
        "type": "text",
        "options":[]},
        {"required": False,
        "instructions": "ICAO24",
        "name": "icao24",
        "type": "text",
        "options":[]}                        
    ]
}
new_project.setup(new_project_frontend, ontology)
new_project.datasets.connect(new_dataset)

print("\n\tSetup Complete!\n--------------------------")
print(f"The project id is: {new_project.uid}")
print(f"The dataset id is: {new_dataset.uid}")