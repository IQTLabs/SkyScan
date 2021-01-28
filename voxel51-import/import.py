import random
import fiftyone as fo
import requests
import json
import math
import re
import os, sys, time, uuid
import argparse
from multiprocessing.dummy import Pool as ThreadPool
import pandas as pd





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

                item = {"file_path": image_path,
                    "external_id": external_id}
                labelbox_import.append(item)
    return labelbox_import


def importImageList(fileList):
    importName = str(time.time())

    samples = []
    for image in fileList:
        plane_id = image["external_id"].split("_")[0]
        plane = planes.loc[planes['icao24'] == plane_id.lower()]
        
        sample = fo.Sample(filepath=image["file_path"])
        if plane.size == 27:
            print("Adding metadata for plane {} onto image {}".format(plane["icao24"].values[0], image["external_id"]))
            sample["plane"] = fo.Classification(label="True")
            sample["model"] = fo.Classification(label=plane["model"].values[0] )
            sample["manufacturer"] = fo.Classification(label=plane["manufacturername"].values[0])
            sample.tags.append("plane")
        samples.append(sample)

    # Create the dataset
    dataset = fo.Dataset(name="plane-dataset")
    dataset.add_samples(samples)
    dataset.persistent = True
    # View summary info about the dataset
    print(dataset)

    # Print the first few samples in the dataset
    print(dataset.head())


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

    print("\n\tImporting Images\n---------------------------------") 
    image_list = buildImageList(file_path)

    batch_size = 1000
    try:
        dataset = fo.load_dataset("plane-dataset")
        dataset.delete()
        print("Removed the old version of the plane-dataset")
    except fo.core.dataset.DoesNotExistError:
        print("Dataset does not currently exist - creating it!")
    if len(image_list) > 0:
        print("Found {} images, processing in batches of: {}".format(len(image_list), batch_size))

        for i in range(0, len(image_list), batch_size):
            chunk = image_list[i:i + batch_size]
            importImageList(chunk)

    else:
        print("No new files to upload")
    


                    





if __name__ == '__main__':
    main()