from labelbox import Client, Project, schema, Dataset
from labelbox.schema.bulk_import_request import BulkImportRequest
from labelbox.schema.enums import BulkImportRequestState
import requests
import json
import math
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

    # you will get an Annotation Import error if there is a nan for an answer. better to pass in a none string instead.
    if isinstance(answer, str) == False:
        answer = "none"
    item = { "schemaId":schemaId,
            "uuid": str(my_uuid),
            "dataRow": {
                "id": dataRowId
            },
            "answer": answer }
    return item

def upload_image(item):
    
    while True:
        try:
            image_url = client.upload_file(item["file_path"])
        except Exception as e:
            print("Exception uploading {} \n{}".format(item["file_path"],e))
            continue
        break

    new_item = {"row_data": image_url, "external_id": item["external_id"]}
    return new_item



def buildImageList(filePath):
    labelbox_import = []
    for folder, subfolders, files in os.walk(filePath):
        for file in files:
            if file.endswith(".jpg"):
                image_filename = os.path.basename(file)
                external_id = os.path.splitext(image_filename)[0]
                image_path = os.path.abspath(os.path.join(folder, file))
                if any(row.external_id == external_id for row in data_rows):
                    print("Image already uploaded: {}".format(image_path))
                else:
                    item = {"file_path": image_path,
                        "external_id": external_id}
                    labelbox_import.append(item)
    return labelbox_import


def importImageList(fileList):
    importName = str(time.time())
    file_upload_thread_count = 5

    print("Uploading images...")
    with ThreadPool(file_upload_thread_count) as thread_pool:
        fileList = thread_pool.map(upload_image, fileList)
    print("Importing images...")
    upload_job = dataset.create_data_rows(fileList)
    print("The image import is: {}".format(upload_job.status))
    annotations = []
    updated_data_rows = list(dataset.data_rows())
    for image in fileList:
        #data_row = filter(lambda row: row['external_id'] == image["external_id"], updated_data_rows) #dataset.data_row_for_external_id(image["external_id"])
        plane_id = image["external_id"].split("_")[0]
        plane = planes.loc[planes['icao24'] == plane_id.lower()]
        
        if plane.size == 27:
            print("Adding metadata for plane {} onto image {}".format(plane["icao24"].values[0], image["external_id"]))
            uid = None
            for row in updated_data_rows:
                if row.external_id == image["external_id"]:
                    uid = row.uid
                    break

            #metadata = {"operator": plane["operator"].values[0], "manufacturer": plane["manufacturername"].values[0], "icao24": plane["icao24"].values[0], "model": plane["model"].values[0], "registration": plane["registration"].values[0]}
            #set_metadata(client, data_row.uid, json.dumps(metadata))
            if uid != None:
                annotations.append(generateClassification(modelSchemaId, uid, plane["model"].values[0] ))    
                annotations.append(generateClassification(manufacturerSchemaId, uid, plane["manufacturername"].values[0] ))    
                annotations.append(generateClassification(operatorSchemaId, uid, plane["operator"].values[0] ))    
                annotations.append(generateClassification(icao24SchemaId, uid, plane["icao24"].values[0] ))    
    
    # from https://labelbox.com/docs/python-api/model-assisted-labeling-python-script
    print(annotations)
    print("Importing {} annotations for {} images".format(len(annotations), len(fileList)))
    try:
        project.upload_annotations(annotations = annotations, name = importName)
        upload_job = BulkImportRequest.from_name(client, project_id = project.uid, name = importName)
        upload_job.wait_until_done()
    except Exception as e:
        print(e)
        print(annotations)
    print("The annotation import is: {upload_job.state}")
    print(upload_job)
    if upload_job.error_file_url:
        res = requests.get(upload_job.error_file_url)
        print(res)
        #errors = ndjson.loads(res.text)
        #print("\nErrors:")
        #for error in errors:
        #    print(
        #        "An annotation failed to import for "
        #        f"datarow: {error['dataRow']} due to: "
        #        f"{error['errors']}")

def main():
    global client
    global project
    global planes
    global data_rows
    global dataset
    global modelSchemaId
    global operatorSchemaId
    global manufacturerSchemaId
    global icao24SchemaId

    parser = argparse.ArgumentParser()

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
    batch_size = 10
    client = Client(os.environ.get("LABELBOX_API_KEY"))
    projectName = os.environ.get("LABELBOX_PROJECT_NAME")
    datasetName = os.environ.get("LABELBOX_DATASET_NAME")



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
    print("\n\tImporting Images\n---------------------------------") 
    labelbox_import = []
    for folder, subfolders, files in os.walk(file_path):
        for file in files:
            if file.endswith(".jpg"):

                if len(labelbox_import) > 10:
                    break
                image_filename = os.path.basename(file)
                external_id = os.path.splitext(image_filename)[0]
                image_path = os.path.abspath(os.path.join(folder, file))
                if any(row.external_id == external_id for row in data_rows):
                    print("Image already uploaded: {}".format(image_path))
                else:
                    item = {"file_path": image_path,
                        "external_id": external_id}
                    labelbox_import.append(item)

                if len(labelbox_import) > 10:
                    break

    print(labelbox_import)
    if len(labelbox_import) > 0:
        print("Found {} images, processing in batches of: {}".format(len(labelbox_import), batch_size))

        importName = str(time.time())
        file_upload_thread_count = 5

        print("Uploading images...")
        with ThreadPool(file_upload_thread_count) as thread_pool:
            labelbox_import = thread_pool.map(upload_image, labelbox_import)
        print("Importing images...")
        upload_job = dataset.create_data_rows(labelbox_import)
        print("The image import is: {}".format(upload_job.status))
        annotations = []
        updated_data_rows = list(dataset.data_rows())
        for image in labelbox_import:
            #data_row = filter(lambda row: row['external_id'] == image["external_id"], updated_data_rows) #dataset.data_row_for_external_id(image["external_id"])
            plane_id = image["external_id"].split("_")[0]
            plane = planes.loc[planes['icao24'] == plane_id.lower()]
            
            if plane.size == 27:
                print("Adding metadata for plane {} onto image {}".format(plane["icao24"].values[0], image["external_id"]))
                uid = None
                for row in updated_data_rows:
                    if row.external_id == image["external_id"]:
                        uid = row.uid
                        break

                #metadata = {"operator": plane["operator"].values[0], "manufacturer": plane["manufacturername"].values[0], "icao24": plane["icao24"].values[0], "model": plane["model"].values[0], "registration": plane["registration"].values[0]}
                #set_metadata(client, data_row.uid, json.dumps(metadata))
                if uid != None:
                    annotations.append(generateClassification(modelSchemaId, uid, plane["model"].values[0] ))    
                    annotations.append(generateClassification(manufacturerSchemaId, uid, plane["manufacturername"].values[0] ))    
                    annotations.append(generateClassification(operatorSchemaId, uid, plane["operator"].values[0] ))    
                    annotations.append(generateClassification(icao24SchemaId, uid, plane["icao24"].values[0] ))    
        
        # from https://labelbox.com/docs/python-api/model-assisted-labeling-python-script
        print(annotations)
        print("Importing {} annotations for {} images".format(len(annotations), len(labelbox_import)))
        try:
            project.upload_annotations(annotations = annotations, name = importName)
            upload_job = BulkImportRequest.from_name(client, project_id = project.uid, name = importName)
            upload_job.wait_until_done()
        except Exception as e:
            print(e)
            print(annotations)
        print("The annotation import is: {upload_job.state}")
        print(upload_job)
        if upload_job.error_file_url:
            res = requests.get(upload_job.error_file_url)
            print(res)
            #errors = ndjson.loads(res.text)
            #print("\nErrors:")
            #for error in errors:
            #    print(
            #        "An annotation failed to import for "
            #        f"datarow: {error['dataRow']} due to: "
            #        f"{error['errors']}")

        

    else:
        print("No new files to upload")
    


                    





if __name__ == '__main__':
    main()