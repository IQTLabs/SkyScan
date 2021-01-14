from labelbox import Client, Project, schema, Dataset
from labelbox.schema.bulk_import_request import BulkImportRequest
from labelbox.schema.enums import BulkImportRequestState
import requests
import json
import re
import os, sys, time, uuid
import argparse
from multiprocessing.dummy import Pool as ThreadPool
import pandas as pd
import ndjson


def get_project_ontology(project_id: str) -> dict:
    """
    Gets the ontology of the given project

    Args:
        project_id (str): The id of the project
    Returns:
        The ontology of the project in a dict format
    """
    res_str = client.execute("""
                    query get_ontology($proj_id: ID!) {
                        project(where: {id: $proj_id}) {
                            ontology {
                                normalized
                            }
                        }
                    }
                """, {"proj_id": project_id})
    return res_str['project']['ontology']['normalized']

def get_schema_ids(ontology: dict) -> dict:
    """
    Gets the schema id's of each tool given an ontology
    
    Args:
        ontology (dict): The ontology that we are looking to parse the schema id's from
    Returns:
        A dict containing the tool name and the schema information
    """
    schemas = {}
    for tool in ontology['tools']:
        schema = {
            'schemaNodeId': tool['featureSchemaId'],
            'color': tool['color'],
            'tooltype':tool['tool']
                    }
        schemas[tool['name']] = schema
    return schemas

def set_metadata(client: Client, data_row_id: str, metadata: str): 
    result = client.execute("""
        mutation addAssetInfo($data_row_id: ID!, $metadata: String!) {
        createAssetMetadata(
            data: {
            dataRowId: $data_row_id,
            metaValue:$metadata,
            metaType: TEXT
            }
        ) {
            id
        }
        }
    """, {"data_row_id": data_row_id, "metadata": metadata})
    print(result)
        #return result['project']['ontology']['normalized']

# from: https://labelbox.com/docs/automation/mal-import-formats
def generateClassification(schemaId, dataRowId, answer):
    my_uuid = uuid.uuid4()
    item = { "schemaId":schemaId,
            "uuid": str(my_uuid),
            "dataRow": {
                "id": dataRowId
            },
            "answer": answer }
    return item


def main():
    global client
    parser = argparse.ArgumentParser()
    parser.add_argument('--apiKey', help='LabelBox API Key to use',
                        default=None)
    parser.add_argument('--filePath', help='files to upload and archive',
                        default=None)
    args = parser.parse_args()

    if args.filePath is None:
        print("you gotta tell us where the files you want to upload are. use --filePath")
        sys.exit()
    else:
        file_path = args.filePath
    print("\n\tLoading Planes\n---------------------------------")
    planes = pd.read_csv("../data/aircraftDatabase.csv") #,index_col='icao24')
    print("Printing table")
    print(planes)
    print("Connecting to LabelBox......\n")
    client = Client(args.apiKey)
    projectName = "SkyScan"
    datasetName = "high alt planes"
    importName = str(time.time())
    file_upload_thread_count = 20


    print("\n\tLoading Project\n---------------------------------")
    projects = client.get_projects(where=Project.name == projectName)
    projects = list(projects)
    if len(projects) != 1:
        print("Expect a single project named: {} but found {} projects".format(projectName, len(projects)))
        exit(0)
    project = projects[0]
    print("Working with Project {}\n\"{}\"\nID: {}\n ".format(project.name, project.description, project.uid))

    print("\n\tLoading Dataset\n---------------------------------")
    datasets = project.datasets(where=Dataset.name == datasetName)
    datasets = list(datasets)
    if len(datasets) != 1:
        print("Expect a single dataset named: {} but found {} projects".format(datasetName, len(datasets)))
        exit(0)
    dataset = datasets[0]

    ontology = get_project_ontology(project.uid)
    modelSchemaId = next(item for item in ontology["classifications"] if item["name"] == "model")["featureSchemaId"]
    operatorSchemaId = next(item for item in ontology["classifications"] if item["name"] == "operator")["featureSchemaId"]
    manufacturerSchemaId = next(item for item in ontology["classifications"] if item["name"] == "manufacturer")["featureSchemaId"]
    icao24SchemaId = next(item for item in ontology["classifications"] if item["name"] == "icao24")["featureSchemaId"]
    
    
    print("Working with Dataset {}\n\"{}\"\nID: {} \n".format(dataset.name, dataset.description, dataset.uid))
    data_rows = list(dataset.data_rows())
    print("There are {} rows in the dataset".format(len(data_rows)))
    print(data_rows)
    print("\n\tImporting Images\n---------------------------------") 
    labelbox_import = []

    def upload_image(item):
        image_url = client.upload_file(image_path)
        new_item = {"row_data": image_url, "external_id": item["external_id"]}
        return new_item

    for file in os.listdir(file_path):
        if file.endswith(".jpg"):
            image_filename = os.path.basename(file)
            external_id = os.path.splitext(image_filename)[0]
            image_path = os.path.join(file_path, file)
            if any(row.external_id == external_id for row in data_rows):
                print("Image already uploaded: {}".format(image_path))
            else:
                item = {"file_path": image_path,
                    "external_id": external_id}
                labelbox_import.append(item)

    if len(labelbox_import) > 0:
        with ThreadPool(file_upload_thread_count) as thread_pool:
            labelbox_import = thread_pool.map(upload_image, labelbox_import)
        print("Found {} images".format(len(labelbox_import)))
        upload_job = dataset.create_data_rows(labelbox_import)
        print("The image upload is: {}".format(upload_job.status))
        annotations = []
        for image in labelbox_import:
            data_row = dataset.data_row_for_external_id(image["external_id"])
            plane_id = image["external_id"].split("_")[0]
            plane = planes.loc[planes['icao24'] == plane_id.lower()]
            
            if plane.size == 27:
                print("Adding metadata for plane {} onto image {}".format(plane["icao24"].values[0], image["external_id"]))
                metadata = {"operator": plane["operator"].values[0], "manufacturer": plane["manufacturername"].values[0], "icao24": plane["icao24"].values[0], "model": plane["model"].values[0], "registration": plane["registration"].values[0]}
                set_metadata(client, data_row.uid, json.dumps(metadata))
                annotations.append(generateClassification(modelSchemaId, data_row.uid, plane["model"].values[0] ))    
                annotations.append(generateClassification(manufacturerSchemaId, data_row.uid, plane["manufacturername"].values[0] ))    
                annotations.append(generateClassification(operatorSchemaId, data_row.uid, plane["operator"].values[0] ))    
                annotations.append(generateClassification(icao24SchemaId, data_row.uid, plane["icao24"].values[0] ))    
        
        # from https://labelbox.com/docs/python-api/model-assisted-labeling-python-script
        print("Importing {} annotations for {} images".format(len(annotations), len(labelbox_import)))
        project.upload_annotations(annotations = annotations, name = importName)
        upload_job = BulkImportRequest.from_name(client, project_id = project.uid, name = importName)
        upload_job.wait_until_done()

        print(f"The annotation import is: {upload_job.state}")

        if upload_job.error_file_url:
            res = requests.get(upload_job.error_file_url)
            errors = ndjson.loads(res.text)
            print("\nErrors:")
            for error in errors:
                print(
                    "An annotation failed to import for "
                    f"datarow: {error['dataRow']} due to: "
                    f"{error['errors']}")
    else:
        print("No new files to upload")
    


                    





if __name__ == '__main__':
    main()