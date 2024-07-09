#!/usr/bin/env python
# coding: utf-8

"""
SkyScan Edge AI
"""

import os
import sys
import json
import glob
import shutil
import subprocess
import numpy as np
import pandas as pd
import IPython.display

from utils.metrics2 import ConfusionMatrix

display_width = 120
default_seed = 1903

class Dataset:

    def create_from_images(self, path, suffix='jpg'):
        """
        Initialize dataframe using a directory of images
        """
        image_paths = sorted(glob.glob(path + '/**/*.'+suffix, recursive=True))
        self.df = pd.DataFrame(image_paths, columns=['path'])
        self.df['filename'] = self.df['path'].apply(
            lambda x: os.path.basename(x))
        self.df['icao24'] = self.df['path'].apply(
            lambda x: os.path.basename(x).split('_')[0].lower())

    def create_from_dirs(self, paths, suffix='jpg'):
        """
        Initialize dataframe using a list of directories of images
        """
        image_paths = []
        for path in paths:
            more_image_paths = sorted(glob.glob(path + '/**/*.'+suffix,
                                                recursive=True))
            image_paths.extend(more_image_paths)
        self.df = pd.DataFrame(image_paths, columns=['path'])
        self.df['filename'] = self.df['path'].apply(
            lambda x: os.path.basename(x))
        self.df['icao24'] = self.df['path'].apply(
            lambda x: os.path.basename(x).split('_')[0].lower())

    def create_from_paths_file(self, path, suffix='jpg'):
        """
        Initialize dataframe using a file of paths
        Note: Assumes paths are relative to the folder from which code is run
        """
        #path_file = open(path, 'r')
        with open(path, 'r') as path_file:
            image_paths = path_file.readlines()
        self.df = pd.DataFrame(image_paths, columns=['path'])
        self.df['filename'] = self.df['path'].apply(
            lambda x: os.path.basename(x))
        self.df['icao24'] = self.df['path'].apply(
            lambda x: os.path.basename(x).split('_')[0].lower())

    def concat(self, second_dataset):
        """
        Return a new dataset which is the concatenation of this one
        and the provided `second_dataset`
        """
        new_dataset = Dataset()
        new_dataset.df = pd.concat((self.df, second_dataset.df),
                                   ignore_index=True)
        return new_dataset

    def load_makemodel_string(self, database_path=None, method=1):
        """
        Add raw make+model string from 3rd-party database
        """
        if method == 0:
            # Assumes images are already sorted into folders named after
            # make+model string
            df['string'] = df['path'].apply(lambda x: x.split('/')[-2])
        else:
            # Convert icao24 (ICAO 24-bit address) to make+model string with
            # OpenSky Network database.  Source:
            # https://opensky-network.org/datasets/metadata/aircraftDatabase.csv

            # Step 1: Load and format lookup table
            database_dataframe = pd.read_csv(database_path, usecols=[
                'icao24', 'manufacturername', 'model'])
            database_dataframe['string'] = database_dataframe[
                'manufacturername'] + ' ' + database_dataframe['model']
            database_dataframe = database_dataframe[['icao24', 'string']]

            # Step 2: Join tables
            self.df = pd.merge(self.df, database_dataframe,
                               how='left', on='icao24')
            database_dataframe = None

    def load_makemodel_mm(self, hierarchy_path,
                          print_lookup_table=False, warn_missing=True):
        """
        Convert raw make+model string to sanitized version with a lookup table
        """

        # Convert class hierarchy to table of make/model/string combinations
        # Note: 'string' is unsanitized make+model from database;
        # 'mm' is make+model after data cleaning
        class_hierarchy = json.load(open(hierarchy_path))
        class_array = np.concatenate([np.concatenate([np.stack([
            np.array([make + ' ' + model, make, model, string])
            for string in class_hierarchy['make_model_strings'][make][model]], axis=0)
            for model in class_hierarchy['make_model_strings'][make]], axis=0)
            for make in class_hierarchy['make_model_strings']], axis=0)
        class_dataframe = pd.DataFrame(class_array, columns=['mm', 'make', 'model', 'string'])

        if print_lookup_table:
            with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.width', display_width):
                print(class_dataframe[['mm', 'string']])

        # Join class information to dataset item information
        self.df = pd.merge(self.df, class_dataframe, how='left', on='string')

        # Check for items not found in class hierarchy
        if warn_missing:
            missing = sorted(self.df[self.df['mm'].isna() & ~self.df['string'].isna()]['string'].unique())
            if len(missing) > 0:
                print('! Make+model strings missing from class hierarchy!')
                print(missing)

        # Return class hierarchy
        return class_hierarchy

    def print_classes(self, group_field='mm', title=None, condition=None):
        """
        Print count of each class
        """
        if title is not None:
            print('-' * display_width)
            print(title)
        if condition is not None:
            df = self.df[condition]
        else:
            df = self.df
        image_count = df.groupby(group_field)[group_field].count()
        plane_count = df.groupby('icao24').head(1).groupby(group_field)[group_field].count()
        count_dataframe = pd.concat([image_count, plane_count], axis=1)
        count_dataframe.columns=['image_count', 'aircraft_count']
        with pd.option_context('display.max_rows', None,
                               'display.width', display_width):
            print(count_dataframe)
        print()
        print('Identified images:', image_count.sum())
        print('Identified aircraft:', plane_count.sum())
        print('Total images:', df['path'].count())
        print('Total aircraft:', df.groupby('icao24')['icao24'].head(1).count())

    def add_usable(self, detector_classes, block_makes=None):
        """
        Mark usable items for localizer and detector
        """
        # Identified (and unidentified, if not overwritten below)
        if block_makes is not None:
            self.df['localizer_usable'] = ~self.df['make'].isin(block_makes)
        else:
            self.df['localizer_usable'] = True
        self.df['detector_usable'] = self.df['mm'].isin(detector_classes)

        # Unidentified (if different from above)
        # self.df.loc[self.df['mm'].isna(), 'localizer_usable'] = True
        # self.df.loc[self.df['mm'].isna(), 'detector_usable'] = False

    def print(self, title=None, condition=None):
        """
        Print dataframe
        """
        if title is not None:
            print('-' * display_width)
            print(title)
        with pd.option_context('display.max_columns', None,
                               'display.width', display_width):
            if condition is None:
                print(self.df)
            else:
                print(self.df[condition])

    def write_suggested_labels(self, output_path,
                               max_planes_per_class=3,
                               max_images_per_plane=5,
                               seed1=default_seed,
                               seed2=default_seed):
        """
        Write a file with a list of suggested images to hand-label
        with bounding boxes.  Include N random images per plane from
        up to M planes of each class, to emphasize less common classes.
        """
        # Step 1: Select planes
        planes_to_label = self.df.groupby('icao24').head(1)[['icao24', 'mm']].sample(frac=1, random_state=seed1).groupby('mm').head(max_planes_per_class).sort_values(by='mm')

        # Step 2: Select images
        images_to_label = self.df[self.df['icao24'].isin(planes_to_label['icao24'])]
        images_to_label = images_to_label[images_to_label['localizer_usable']]
        images_to_label = images_to_label.sample(frac=1, random_state=seed2).groupby('icao24').head(max_images_per_plane).sort_values(by=['mm', 'icao24'])

        # Step 3: Save
        images_to_label['path'].to_csv(output_path, header=False, index=False)

    def write_suggested_labels_prop(self, output_path,
                                    num_planes=10,
                                    max_images_per_plane=5,
                                    seed1=default_seed,
                                    seed2=default_seed):
        """
        Write a file with a list of suggested images to hand-label
        with bounding boxes.  Include N random images per plane from
        each of M planes, to follow the observed distribution of classes.
        """
        # Step 1: Select planes
        planes_to_label = self.df.groupby('icao24').head(1)[['icao24', 'mm']].sample(frac=1, random_state=seed1).head(num_planes).sort_values(by=['mm', 'icao24'])

        # Step 2: Select images
        images_to_label = self.df[self.df['icao24'].isin(planes_to_label['icao24'])]
        images_to_label = images_to_label[images_to_label['localizer_usable']]
        images_to_label = images_to_label.sample(frac=1, random_state=seed2).groupby('icao24').head(max_images_per_plane).sort_values(by=['mm', 'icao24'])

        # Step 3: Save
        images_to_label['path'].to_csv(output_path, header=False, index=False)

    def load_labelbox_labels(self, input_path,
                             img_height=1080, img_width=1920, min_area=100):
        """
        Load manual bounding boxes from LabelBox export file into dataframe
        """
        label_struct = json.load(open(input_path))

        self.df['manual_bbox_flag'] = False
        self.df['manual_bbox_count'] = 0
        self.df['manual_bbox_desc'] = ''
        filename_index = pd.Index(self.df['filename'])
        for item in label_struct:
            # Find this item in dataframe
            filename = item['External ID']
            try:
                loc = filename_index.get_loc(filename)
            except KeyError:
                print('! Did not find entry for', filename)
                continue

            # Parse hand-labeled bounding box
            desc = ''
            valid_boxes = 0
            for obj in item['Label']['objects']:
                bbox = obj['bbox']
                top = bbox['top']
                left = bbox['left']
                height = bbox['height']
                width = bbox['width']
                if height * width >= min_area:
                    if valid_boxes > 0:
                        desc += '\n'
                    desc += '0 %f %f %f %f' % (
                        (left + 0.5 * width) / img_width,
                        (top + 0.5 * height) / img_height,
                        width / img_width,
                        height / img_height
                    )
                    valid_boxes += 1
            self.df.at[loc, 'manual_bbox_flag'] = True
            self.df.at[loc, 'manual_bbox_count'] = valid_boxes
            self.df.at[loc, 'manual_bbox_desc'] = desc

    def load_yolov7_labels(self, name,
                           img_height=1080, img_width=1920,
                           min_area=100,
                           always_flag=['detector_train', 'detector_test']):
        """
        Load YOLOv7-generated bounding boxes
        """
        path = os.path.join('runs/test', name, 'best_predictions.json')
        label_struct = json.load(open(path))

        self.df['auto_bbox_flag'] = False
        self.df['auto_bbox_count'] = 0
        self.df['auto_bbox_desc'] = ''
        filename_index = pd.Index(self.df['filename'].str[:-4])
        for entry in label_struct:
            # Find this item in dataframe
            filename = entry['image_id']
            loc = filename_index.get_loc(filename)

            # Parse auto-labeled bounding box
            left, top, width, height = tuple(entry['bbox'])
            if height * width >= min_area:
                if self.df.loc[loc, 'auto_bbox_count'] > 0:
                    self.df.loc[loc, 'auto_bbox_desc'] += '\n'
                self.df.loc[loc, 'auto_bbox_desc'] += '0 %f %f %f %f' % (
                    (left + 0.5 * width) / img_width,
                    (top + 0.5 * height) / img_height,
                    width / img_width,
                    height / img_height
                )
                self.df.loc[loc, 'auto_bbox_count'] += 1

        # Flag all entries for which bboxes were sought
        # (e.g., detector train/test), even those with no bboxes found
        for colname in always_flag:
            self.df.loc[self.df[colname], 'auto_bbox_flag'] = True

    def split_train_test(self, detector_classes,
                         localizer_train_count = 2,
                         detector_test_count = 3,
                         seed3 = default_seed,
                         seed4 = default_seed,
                         verbose = True):
        """
        Divides images into four categories: localizer training, localizer
        testing, detector training, and detector testing.  The division is
        such that no one plane's images appear in more than one category.
        Not all images are assigned, because localizer categories can only
        use a plane's hand-labeled bounding boxes.
        """

        # Assign planes to use: train/test for localizer/detector
        planes = self.df[['icao24', 'mm', 'manual_bbox_flag']].groupby('icao24', as_index=False).max()
        planes_manual = planes[planes['manual_bbox_flag']]
        planes_notman = planes[~planes['manual_bbox_flag']]
        planes_localizer = planes_manual
        planes_localizer_train = planes_localizer.sample(frac=1, random_state=seed3).groupby('mm', as_index=False).head(localizer_train_count)
        planes_localizer_test = planes_localizer[~planes_localizer['icao24'].isin(planes_localizer_train['icao24'])]
        planes_detector = planes_notman[planes_notman['mm'].isin(detector_classes)]
        planes_detector_test = planes_detector.sample(frac=1, random_state=seed4).groupby('mm', as_index=False).head(detector_test_count)
        planes_detector_train = planes_detector[~planes_detector['icao24'].isin(planes_detector_test['icao24'])]

        # Print plane counts
        if verbose:
            print('planes', len(planes))
            print('planes_manual', len(planes_manual))
            print('planes_notman', len(planes_notman))
            print('planes_localizer_train', len(planes_localizer_train))
            print('planes_localizer_test', len(planes_localizer_test))
            print('planes_detector_train', len(planes_detector_train))
            print('planes_detector_test', len(planes_detector_test))

        # Assign dataset items to use: train/test for localizer/detector
        items_localizer_train = self.df[self.df['icao24'].isin(planes_localizer_train['icao24'])]
        items_localizer_train = items_localizer_train[items_localizer_train['manual_bbox_flag']]
        items_localizer_test = self.df[self.df['icao24'].isin(planes_localizer_test['icao24'])]
        items_localizer_test = items_localizer_test[items_localizer_test['manual_bbox_flag']]
        items_detector_train = self.df[self.df['icao24'].isin(planes_detector_train['icao24'])]
        items_detector_test = self.df[self.df['icao24'].isin(planes_detector_test['icao24'])]

        # Save to main dataframe
        self.df['localizer_train'] = self.df['path'].isin(items_localizer_train['path'])
        self.df['localizer_test'] = self.df['path'].isin(items_localizer_test['path'])
        self.df['detector_train'] = self.df['path'].isin(items_detector_train['path'])
        self.df['detector_test'] = self.df['path'].isin(items_detector_test['path'])

        # Print item counts
        if verbose:
            print('items', self.df.shape[0])
            print('items_localizer_train', self.df['localizer_train'].sum())
            print('items_localizer_test', self.df['localizer_test'].sum())
            print('items_detector_train', self.df['detector_train'].sum())
            print('items_detector_test', self.df['detector_test'].sum())

    def split_train_val_shared(self, detector_classes,
                               val_fraction = 0.3,
                               seed3 = default_seed,
                               verbose = True):
        """
        Assigns images to four categories: localizer training, localizer
        validation, detector training, and detector validation.  The localizer
        images are a proper subset of the detector images, but no image is used
        for both training and validation.
        """

        # Assign planes to use: either train or validation
        planes = self.df[['icao24', 'mm', 'manual_bbox_flag']].groupby('icao24', as_index=False).max()
        val_count = int(val_fraction * len(planes))
        planes_randomized = planes.sample(frac=1, random_state=seed3)
        planes_train = planes_randomized.iloc[val_count:]
        planes_val = planes_randomized.iloc[:val_count]

        # Assign dataset items to use: train/validation for localizer/detector
        items_train = self.df[self.df['icao24'].isin(planes_train['icao24'])]
        items_val = self.df[self.df['icao24'].isin(planes_val['icao24'])]
        items_localizer_train = items_train[items_train['manual_bbox_flag']]
        items_localizer_val = items_val[items_val['manual_bbox_flag']]
        items_detector_train = items_train[items_train['mm'].isin(detector_classes)]
        items_detector_val = items_val[items_val['mm'].isin(detector_classes)]

        # Save to main dataframe
        self.df['localizer_train'] = self.df['path'].isin(
            items_localizer_train['path'])
        self.df['localizer_val'] = self.df['path'].isin(
            items_localizer_val['path'])
        self.df['detector_train'] = self.df['path'].isin(
            items_detector_train['path'])
        self.df['detector_val'] = self.df['path'].isin(
            items_detector_val['path'])

        # Print plane counts
        if verbose:
            print('planes', len(planes))
            print('planes_train', len(planes_train))
            print('planes_val', len(planes_val))

        # Print item counts
        if verbose:
            print('items', self.df.shape[0])
            print('items_train', len(items_train))
            print('items_val', len(items_val))
            print('items_localizer_train', self.df['localizer_train'].sum())
            print('items_localizer_val', self.df['localizer_val'].sum())
            print('items_detector_train', self.df['detector_train'].sum())
            print('items_detector_val', self.df['detector_val'].sum())

    def check_categories_nonintersecting(self, categories):
        """
        Confirm that no plane's images appear in multiple usage categories
        """
        for cat1idx in range(len(categories)):
            cat1 = set(self.df.loc[self.df[categories[cat1idx]], 'icao24'])
            for cat2idx in range(cat1idx + 1, len(categories)):
                cat2 = set(self.df.loc[self.df[categories[cat2idx]], 'icao24'])
                inter = cat1.intersection(cat2)
                if len(inter) > 0:
                    print('! Warning: Category Overlap!',
                          categories[cat1idx], categories[cat2idx])

    def write_folders(self, base_path, categories):
        """
        Create folders to hold dataset files
        """
        for category in categories:
            os.makedirs(os.path.join(base_path, category, 'images'),
                        exist_ok=True)
            os.makedirs(os.path.join(base_path, category, 'labels'),
                        exist_ok=True)

    def write_images(self, base_path, categories):
        """
        Write dataset images
        """
        for category in categories:
            dest_dir = os.path.join(base_path, category, 'images')
            for src_path in self.df.loc[self.df[category], 'path']:
                shutil.copy2(src_path, dest_dir)

    def write_labels(self, base_path, categories, col, condition=None):
        """
        Write dataset labels
        """
        for category in categories:
            for idx, row in self.df[self.df[category]].iterrows():
                if condition is None or condition[idx]:
                    dest_name = os.path.splitext(row['filename'])[0] + '.txt'
                    dest_path = os.path.join(base_path, category,
                                             'labels', dest_name)
                    dest_file = open(dest_path, 'w')
                    print(row[col], file=dest_file)
                    dest_file.close()

    def write_image_lists(self, base_path, categories, condition):
        """
        Write image lists, using only images that meet condition
        """
        for category in categories:
            list_content = './' + category + '/images/' + self.df.loc[
                self.df[category] & condition, 'filename']
            list_content.to_csv(os.path.join(base_path, category + '.txt'),
                                header=False, index=False)

    def add_class_labels(self, detector_classes, condition=None,
                         input_col='auto_bbox_desc',
                         output_col='auto_bboxclass_desc'):
        """
        Given localizer bounding boxes (which are all of class zero),
        generate detector bounding boxes (which indicate class of plane)
        """
        detector_classes_dict = {x:y for y, x in enumerate(detector_classes)}
        self.df[output_col] = ''
        for idx, row in self.df.iterrows():
            if condition is None or condition[idx]:
                dest_cont = str(detector_classes_dict[row['mm']]) + row[input_col][1:].replace('\n0', '\n' + str(detector_classes_dict[row['mm']])) # Assumes input class is zero
                self.df.loc[idx, output_col] = dest_cont

    def train(self,
              weights='yolov7-tiny.pt',
              cfg='cfg/training/yolov7-tiny.yaml',
              data=None, #e.g., /code/localizer.yaml
              hyp='data/hyp.scratch.tiny.yaml',
              epochs=None, #e.g., 500
              batchsize=4,
              workers=4,
              name=None, #e.g., yolov7_localizer_01_1
    ):
        """
        Train YOLOv7 model
        """
        cmd = 'python train.py'
        cmd += ' --weights ' + weights
        cmd += ' --cfg ' + cfg
        cmd += ' --data ' + data
        cmd += ' --hyp ' + hyp
        cmd += ' --epochs ' + str(epochs)
        cmd += ' --batch-size ' + str(batchsize)
        cmd += ' --workers ' + str(workers)
        cmd += ' --name ' + name
        print(cmd)
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True, bufsize=1)
        for line in p.stdout:
            print(line)

    def test(self,
             train_name, #e.g., yolov7_localizer_01_1
             test_name, #e.g., yolov7_localizer_01_1
             data, #e.g., /code/localizer.yaml
             batchsize=32,
             confthres=0.25,
             iouthres=0.5,
             task='test',
    ):
        """
        Run YOLOv7 model to generate labels on categories listed in yaml file
        """
        weights = os.path.join('runs/train/', train_name, 'weights/best.pt')
        cmd = 'python test2.py'
        cmd += ' --weights ' + weights
        cmd += ' --data ' + data
        cmd += ' --batch-size ' + str(batchsize)
        cmd += ' --conf-thres ' + str(confthres)
        cmd += ' --iou-thres ' + str(iouthres)
        cmd += ' --task ' + task
        cmd += ' --name ' + test_name
        cmd += ' --save-json'
        print(cmd)
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True, bufsize=1)
        for line in p.stdout:
            print(line)

    def detect(self,
               train_name, # e.g., yolov7_localizer_01_1
               detect_name, # e.g., yolov7_localizer_01_1
               source, # e.g., path/filename.jpg
               imgsize=640,
               confthres=0.25
    ):
        """
        Run YOLOv7 model on specified file(s)
        """
        weights = os.path.join('runs/train/', train_name, 'weights/best.pt')
        cmd = 'python detect.py'
        cmd += ' --weights ' + weights
        cmd += ' --source "' + source + '"'
        cmd += ' --img-size ' + str(imgsize)
        cmd += ' --conf-thres ' + str(confthres)
        cmd += ' --save-txt'
        cmd += ' --save-conf'
        cmd += ' --agnostic-nms'
        cmd += ' --name ' + detect_name
        print(cmd)
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True, bufsize=1)
        for line in p.stdout:
            print(line)

    def sort(self,
             sourcedir,
             planedir,
             noplanedir,
             logdir,
             weights='../code/localizer.pt',
             imgsize=640,
             confthres=0.25,
             device='cpu',
             savejson=False,
    ):
        """
        Runs daemon to sort folder contents into plane and noplane folders
        Enter Ctrl+c to end (otherwise, continues indefinitely)
        """
        cmd = 'python sort.py'
        cmd += ' --weights ' + weights
        cmd += ' --source-dir ' + sourcedir
        cmd += ' --plane-dir ' + planedir
        cmd += ' --noplane-dir ' + noplanedir
        cmd += ' --log-dir ' + logdir
        cmd += ' --img-size ' + str(imgsize)
        cmd += ' --conf-thres ' + str(confthres)
        cmd += ' --device ' + device
        cmd += ' --nosave'
        cmd += ' --agnostic-nms'
        if savejson:
            cmd += ' --save-json'
        print(cmd)
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True)
        for line in p.stdout:
            print(line)

    def sort_tflite(self,
                    sourcedir,
                    planedir,
                    noplanedir,
                    logdir,
                    weights='../data/tflite/localizer.tflite',
                    savejson=False,
    ):
        """
        Runs daemon to sort folder contents into plane and noplane folders
        using the Tensorflow Lite version of the model.
        Enter Ctrl+c to end (otherwise, continues indefinitely).
        """
        cmd = 'python sort_tflite.py'
        cmd += ' --weights ' + weights
        cmd += ' --source-dir ' + sourcedir
        cmd += ' --plane-dir ' + planedir
        cmd += ' --noplane-dir ' + noplanedir
        cmd += ' --log-dir ' + logdir
        if savejson:
            cmd += ' --save-json'
        print(cmd)
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True)
        for line in p.stdout:
            print(line)

    def visualize_train(self, name, batch):
        """
        Show some graphics generated during model training
        Currently, just prints paths to graphics
        """
        results_dir = os.path.join('runs/train/', name)
        file_names = ['results.png',
                      'test_batch' + str(batch) + '_labels.jpg',
                      'test_batch' + str(batch) + '_pred.jpg']
        for file_name in file_names:
            file_path = os.path.join(results_dir, file_name)
            print(file_path)

    def visualize_detect(self, name, image_path):
        """
        Visualize and print detector inference example
        Currently, just prints path to graphic
        """
        results_dir = os.path.join('runs/detect', name)
        idstring = os.path.splitext(os.path.split(image_path)[1])[0]
        image_path = os.path.join(results_dir, idstring + '.jpg')
        text_path = os.path.join(results_dir, 'labels', idstring + '.txt')

        file = open(text_path)
        print(file.readlines())
        file.close()
        print(image_path)

    def check_planes_per_image(self, flag_col='auto_bbox_flag',
                               count_col='auto_bbox_count'):
        """
        Count images with various numbers of planes found
        """
        print('0  Planes:',
              (self.df[flag_col] & (self.df[count_col] == 0)).sum())
        print('1  Plane :',
              (self.df[flag_col] & (self.df[count_col] == 1)).sum())
        print('2+ Planes:',
              (self.df[flag_col] & (self.df[count_col] >= 2)).sum())

    def check_top_global_mm_strings(self, database_path, count=25):
        """
        Print the most common unsanitized make+model strings in the
        aircraft database file
        """
        database_dataframe = pd.read_csv(database_path, usecols=['icao24', 'manufacturername', 'model'])
        database_dataframe['string'] = database_dataframe['manufacturername'] + ' ' + database_dataframe['model']
        database_dataframe = database_dataframe[['icao24', 'string']]
        database_dataframe = database_dataframe.groupby('string', as_index=False).count().sort_values(by='icao24', ascending=False)
        database_dataframe = database_dataframe.reset_index(drop=True)
        with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.width', display_width):
            print(database_dataframe.iloc[:count])

    def check_confusion_matrix(self, cmpath, save=False, names=()):
        """
        Show a confusion matrix previously saved to disk
        """
        cm = ConfusionMatrix(0)
        cm.read(cmpath)
        print(cm.matrix)
        print()
        with np.printoptions(precision=3):
            print(cm.norm(0).matrix)
            print()
            print(cm.norm(1).matrix)
        if save:
            savedir = os.path.split(cmpath)[0]
            cm.plot(savedir, names, normalize=False, transpose=False,
                    round_to_int=True,
                    filename='confusion_matrix_counts')
            cm.norm(0).plot(savedir, names, normalize=False, transpose=False,
                    filename='confusion_matrix_bytrue')
            cm.norm(1).plot(savedir, names, normalize=False, transpose=False,
                    filename='confusion_matrix_bypred')

    def torch2onnx(self,
                   weights, # e.g., 'runs/train/xyz/yolov7_localizer_01_01/weights/best.pt'
                   imgsize=640,
                   confthres=0.25,
                   iouthres=0.5,
                   nms=True
    ):
        """
        Convert PyTorch YOLOv7 model to ONNX
        """
        cmd = 'python export.py'
        cmd += ' --weights ' + weights
        cmd += ' --grid'
        if nms:
            cmd += ' --end2end'
        cmd += ' --simplify'
        cmd += ' --topk-all 100'
        cmd += ' --iou-thres ' + str(iouthres)
        cmd += ' --conf-thres ' + str(confthres)
        cmd += ' --img-size ' + str(imgsize) + ' ' + str(imgsize)
        cmd += ' --max-wh ' + str(imgsize)
        print(cmd)
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True)
        for line in p.stdout:
            print(line)

    def onnx2tf(self,
                input_model, # e.g., 'runs/train/xyz/yolov7_localizer_01_01/weights/best.onnx'
                output_model # e.g., '../data/tf/localizer'
    ):
        """
        Convert ONNX YOLOv7 model to TensorFlow
        """
        cmd = 'onnx-tf convert'
        cmd += ' -i ' + input_model
        cmd += ' -o ' + output_model
        print(cmd)
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True)
        for line in p.stdout:
            print(line)

    def tf2tflite(self,
                  input_model, # e.g., '../data/tf/localizer'
                  output_model # e.g., '../data/tflite/localizer.tflite'
    ):
        """
        Convert TensorFlow YOLOv7 model to TensorFlow Lite
        """
        import tensorflow as tf
        tflite_model = tf.lite.TFLiteConverter\
                              .from_saved_model(input_model)\
                              .convert()
        with open(output_model, 'wb') as output_file:
            output_file.write(tflite_model)


def main_sequence():
    """
    Original model-building sequence, using six classes
    and previously-acquired close-view data
    Note: Methods train, test, detect, sort, and write_* write output to disk
    """
    # Images
    ds = Dataset()
    ds.create_from_images('../data/2021/skyscan-datasets/dataset/close-view')
    ds.load_makemodel_string('../data/databases/aircraftDatabase.csv')
    ch = ds.load_makemodel_mm('../code/taxon.json', True)
    ds.print_classes()
    detector_classes = ['Airbus A319', 'Airbus A320', 'Airbus A321',
                        'Boeing 737', 'Bombardier CL-600', 'Embraer ERJ-170']
    ds.add_usable(detector_classes, ch['helicopters'])
    ds.print(title='Initial Dataset')

    # Bounding boxes and data split
    ds.write_suggested_labels('../data/label_suggestions/label_suggestions.csv')
    ds.load_labelbox_labels('../data/labelbox_export/export-2022-10-09T04_14_18.580Z.json')
    ds.print(title='Manual Labels', condition=ds.df['manual_bbox_flag'])
    ds.split_train_test(detector_classes)
    categories = ['localizer_train', 'localizer_test', 'detector_train', 'detector_test']
    ds.check_categories_nonintersecting(categories)
    ds.print(title='Split')

    # Write dataset to disk
    # ds.write_folders('../data/dataset', categories)
    # ds.write_images('../data/dataset', categories)
    # ds.write_labels('../data/dataset', categories[:2], 'manual_bbox_desc')

    # Localizer
    loc_name_trn = 'yolov7_localizer_01_2' # Change before retraining
    loc_name_tst = 'yolov7_localizer_01_23' # Change before retesting
    loc_data = '/code/localizer.yaml'
    # ds.train(data=loc_data, epochs=500, name=loc_name_trn)
    ds.visualize_train(loc_name_trn, 0)
    # ds.test(train_name=loc_name_trn, test_name=loc_name_tst, data=loc_data)
    ds.load_yolov7_labels(name=loc_name_tst)
    ds.check_planes_per_image()
    ds.write_image_lists('../data/dataset', categories[2:],
                         ds.df['auto_bbox_count']==1)
    ds.add_class_labels(detector_classes, ds.df['auto_bbox_count']==1,
                        'auto_bbox_desc', 'auto_bboxclass_desc')
    # ds.write_labels('../data/dataset', categories[2:], 'auto_bboxclass_desc',
    #                 ds.df['auto_bbox_count']==1)
    ds.print(title='Auto Labels', condition=ds.df['auto_bbox_flag'])

    # Detector
    det_name_trn = 'yolov7_detector_01_4' # Change before retraining
    det_name_det = 'yolov7_detector_01_2' # Change before re-detecting
    det_data = '/code/detector.yaml'
    # ds.train(data=det_data, epochs=50, name=det_name_trn)
    ds.visualize_train(det_name_trn, 0)
    example_file = '../data/skyscan-datasets/dataset/close-view/4-23-DCA/Airbus A319 112/AA0AAB_2021-04-23-14-43-05.jpg'
    # ds.detect(train_name=det_name_trn, detect_name=det_name_det,
    #           source=example_file)
    ds.visualize_detect(det_name_det, example_file)

    # Other
    # Sorting daemon
    # ds.sort('../data/edge_test/tosort', '../data/edge_test/plane',
    #         '../data/edge_test/noplane', '../data/edge_test/log')
    # Common make+models
    # ds.check_top_global_mm_strings('../data/databases/aircraftDatabase.csv', 100)
    # Confusion matrix
    det_name_tst = 'yolov7_detector_01_3' # Change before retesting
    # ds.test(train_name=det_name_trn, test_name=det_name_tst, data=det_data,
    #         task='val')
    cmpath = os.path.join('runs/test', det_name_tst, 'confusion_matrix_counts.txt')
    ds.check_confusion_matrix(cmpath, save=True, names=detector_classes)


def daemon():
    """
    Runs sorting daemon
    """
    ds = Dataset()
    ds.sort('../data/edge_test/tosort',
            '../data/edge_test/plane',
            '../data/edge_test/noplane',
            '../data/edge_test/log',
            savejson=True)


def compare_collections():
    """
    Look at contents of collections of images
    Note: The first set of paths includes many images multiple times
    """
    paths = ['../data/2021/skyscan-datasets/dataset/cruising-view',
             '../data/2021/skyscan-datasets/dataset/close-view',
             '../data/2022/airshow',
             '../data/2022/multi/2022-11-16_field-test',
             '../data/2022/multi/2022-11-18--21_weekend-home',
             '../data/2021/tf/dataset-export/multi_class_train_by_aircraft',
             '../data/2021/tf/dataset-export/multi_class_eval_by_aircraft',
             '../data/2021/322images-training',
             '../data/2021/500image-train']
    paths = ['../data/2022/airshow',
             '../data/2021/skyscan-datasets/dataset/close-view',
             '../data/2022/multi/2022-11-16_field-test']
    # paths = ['../data/classification/test/%02d' % (classnum,)
    #          for classnum in range(23)]
    for path in paths:
        ds = Dataset()
        ds.create_from_images(path)
        ds.load_makemodel_string('../data/databases/aircraftDatabase.csv')
        ds.load_makemodel_mm('../code/taxon.json')
        ds.print_classes(title=path)
    if True:
        ds = Dataset()
        ds.create_from_dirs(paths)
        ds.load_makemodel_string('../data/databases/aircraftDatabase.csv')
        ds.load_makemodel_mm('../code/taxon.json')
        ds.print_classes(title='all')
        print('Identified make+models:', len(ds.df['mm'].unique()))


def entry_search():
    """
    Print dataset items matching command line argument
    """
    paths = ['../data/2022/airshow',
             '../data/2021/skyscan-datasets/dataset/close-view',
             '../data/2022/multi/2022-11-16_field-test']
    col = 'make'
    ds = Dataset()
    ds.create_from_dirs(paths)
    ds.load_makemodel_string('../data/databases/aircraftDatabase.csv')
    ds.load_makemodel_mm('../code/taxon.json')
    print(ds.df[ds.df[col]==sys.argv[1]])
    print(ds.df[ds.df[col]==sys.argv[1]]['icao24'].unique())


def revised_sequence():
    """
    Revised sequence of steps to build two models: a "localizer" to identify
    bounding boxes, and a "detector" to identify bounding boxes as well as
    class of each aircraft
    """

    # FILE NAMES
    # Commercial air transport images for training/validation
    paths_cat = ['../data/2021/skyscan-datasets/dataset/close-view',
             '../data/2022/multi/2022-11-16_field-test']
    # General aviation images for training/validation
    paths_ga = ['../data/2022/airshow']
    # Images for testing
    paths_test = ['../data/2023/dulles/raw']
    # Path to database of aircraft from OpenSky Network
    database_path = '../data/databases/aircraftDatabase.csv'
    # Lookup table with hierarchy of aircraft makes, models, and descriptions
    hierarchy_path = '../code/taxon.json'
    # Make/models to train detector on
    detector_classes = ['Airbus A319', 'Airbus A320', 'Airbus A321',
                        'Beechcraft Bonanza',
                        'Boeing 737',
                        'Bombardier CL-600',
                        'Cessna Citation', 'Cessna Skyhawk', 'Cessna Skylane',
                        'Embraer ERJ-170',
                        'Piper PA-28 Cherokee', 'Piper PA-32 Cherokee Six']
    # Files in which to save lists of suggested images to manually label
    ls_cat = '../data/label_suggestions/label_suggestions_revised_cat.csv'
    ls_ga = '../data/label_suggestions/label_suggestions_revised_ga.csv'
    ls = '../data/label_suggestions/label_suggestions_revised.csv'
    ls_test = '../data/label_suggestions/label_suggestions_revised_test.csv'
    # Folder in which to store dataset
    dataset_dir = '../data/dataset'
    # Files of hand-drawn bounding boxes, exported from LabelBox
    labelbox_export_trainval = '../data/labelbox_export/export-2022-12-07T07_41_59.984Z.json'
    labelbox_export_test = '../data/labelbox_export/export-2023-03-02T07_38_51.807Z.json'

    # YOLOv7 OUTPUT NAMES
    # Folder name for training the localizer
    loc_name_trn = 'yolov7_localizer_02_01' # Change before retraining
    # Folder name for testing the localizer
    loc_name_tst = 'yolov7_localizer_02_04' # Change before retesting
    # Folder name for training the detector
    det_name_trn = 'yolov7_detector_02_01' # Change before retraining
    # Folder name for testing the detector
    det_name_tst = 'yolov7_detector_02_01' # Change before re-detecting
    # Folder name for testing on new data
    test_name = 'yolov7_test_02_14' # Change before retesting

    # Open CAT and GA collects separately, to label different fractions thereof
    ds_cat = Dataset()
    ds_ga = Dataset()
    ds_cat.create_from_dirs(paths_cat)
    ds_ga.create_from_dirs(paths_ga)
    ds_cat.load_makemodel_string(database_path)
    ds_ga.load_makemodel_string(database_path)
    ds_cat.load_makemodel_mm(hierarchy_path)
    ch = ds_ga.load_makemodel_mm(hierarchy_path)
    ds_cat.print_classes(title='Field tests')
    ds_ga.print_classes(title='Airshow')
    ds_cat.add_usable(detector_classes, ch['helicopters']) # + ch['homebuilt'])
    ds_ga.add_usable(detector_classes, ch['helicopters']) # + ch['homebuilt'])
    ds_cat.write_suggested_labels(
        ls_cat, max_planes_per_class=12, max_images_per_plane=2)
    ds_ga.write_suggested_labels(
        ls_ga, max_planes_per_class=12, max_images_per_plane=1)
    subprocess.check_output("cat %s %s > %s" % (ls_cat, ls_ga, ls), shell=True)
    ds = ds_cat.concat(ds_ga)
    ds.print_classes(title='Combined dataset')

    # Check that suggested files to hand-label are as expected
    ds_suggest = Dataset()
    ds_suggest.create_from_paths_file(ls)
    ds_suggest.load_makemodel_string(database_path)
    ds_suggest.load_makemodel_mm(hierarchy_path)
    ds_suggest.print_classes(title='Hand-labeled')

    # Data split
    if not os.path.exists(labelbox_export_trainval):
        raise Exception('Use LabelBox to label bounding boxes for the files in ' + ls + ' and save the exported LabelBox file to ' + labelbox_export_trainval + ' (or specify the correct path in the code), then re-run this function.')
    ds.load_labelbox_labels(labelbox_export_trainval, min_area=400)
    ds.print(title='Manual Labels', condition=ds.df['manual_bbox_flag'])
    ds.split_train_val_shared(detector_classes, val_fraction=0.25)
    ds.print_classes(title='Detector Train', condition=ds.df['detector_train'])
    ds.print_classes(title='Detector Val', condition=ds.df['detector_val'])

    # Write dataset to disk
    categories = ['localizer_train', 'localizer_val',
                  'detector_train', 'detector_val']
    ds.write_folders(dataset_dir, categories)
    ds.write_images(dataset_dir, categories) ##
    ds.write_labels(dataset_dir, categories[:2], 'manual_bbox_desc')

    # Localizer
    loc_data = 'localizer_rev2.yaml'
    ds.train(data=loc_data, epochs=500, name=loc_name_trn) ##
    ds.visualize_train(loc_name_trn, 0)

    # Test on localizer_val data, to generate localizer confusion matrix
    ds.test(train_name=loc_name_trn, test_name=loc_name_tst,
            data=loc_data, task='val') ##
    ds.load_yolov7_labels(name=loc_name_tst, min_area=400,
                          always_flag=['localizer_val']) # Used?
    ds.check_planes_per_image()
    cmpath = os.path.join('runs/test', loc_name_tst,
                          'confusion_matrix_counts.txt')
    ds.check_confusion_matrix(cmpath, save=True, names=['aircraft'])

    # Test on detector data, to generate detector training data labels
    ds.df.drop(['auto_bbox_flag', 'auto_bbox_count', 'auto_bbox_desc'],
               axis='columns')
    ds.test(train_name=loc_name_trn, test_name=loc_name_tst+'b',
            data=loc_data) ##
    ds.load_yolov7_labels(name=loc_name_tst+'b', min_area=400,
                          always_flag=['detector_train', 'detector_val'])
    ds.check_planes_per_image()
    ds.write_image_lists(dataset_dir, categories[2:],
                         ds.df['auto_bbox_count']==1)
    ds.add_class_labels(detector_classes, ds.df['auto_bbox_count']==1,
                        'auto_bbox_desc', 'auto_bboxclass_desc')
    ds.write_labels(dataset_dir, categories[2:], 'auto_bboxclass_desc',
                    ds.df['auto_bbox_count']==1) ##

    # Detector
    det_data = 'detector_rev2.yaml'
    ds.train(data=det_data, epochs=50, name=det_name_trn) ##
    ds.visualize_train(det_name_trn, 0)

    # Test on detector_val data, to generate detector confusion matrix
    ds.test(train_name=det_name_trn, test_name=det_name_tst,
            data=det_data, task='val') ##
    ds.load_yolov7_labels(name=det_name_tst, min_area=400,
                          always_flag=['detector_val'])
    ds.check_planes_per_image()
    cmpath = os.path.join('runs/test', det_name_tst,
                          'confusion_matrix_counts.txt')
    ds.check_confusion_matrix(cmpath, save=True, names=detector_classes)

    # Testing the model on a new dataset
    ds = Dataset()
    ds.create_from_dirs(paths_test)
    ds.load_makemodel_string(database_path)
    ch = ds.load_makemodel_mm(hierarchy_path)
    ds.add_usable(detector_classes, ch['helicopters'])
    ds.print_classes(title='Testing Data')
    # Manual bounding boxes
    ds.write_suggested_labels_prop(ls_test, num_planes=100,
                                   max_images_per_plane=1)
    if not os.path.exists(labelbox_export_trainval):
        raise Exception('Use LabelBox to label bounding boxes for the files in ' + ls_test + ' and save the exported LabelBox file to ' + labelbox_export_test + ' (or specify the correct path in the code), then re-run this function.')
    ds.load_labelbox_labels(labelbox_export_test, min_area=400)
    # Write imagery and manual bounding boxes (for test of localizer)
    categories = ['localizer_test', 'detector_test']
    ds.df['localizer_test'] = ds.df['manual_bbox_flag']
    ds.df['detector_test'] = ds.df['detector_usable']
    ds.write_folders(dataset_dir, categories)
    ds.write_images(dataset_dir, categories) ##
    ds.write_labels(dataset_dir, categories[:1], 'manual_bbox_desc')
    # Generate and write auto bounding boxes (for test of detector)
    ds.test(train_name=loc_name_trn, test_name=test_name+'a',
            data=loc_data, task='test3') ##
    ds.load_yolov7_labels(name=test_name+'a', min_area=400,
                          always_flag=['detector_test'])
    ds.check_planes_per_image()
    ds.write_image_lists(dataset_dir, categories[1:],
                         ds.df['auto_bbox_count']==1)
    ds.add_class_labels(detector_classes, ds.df['auto_bbox_count']==1,
                        'auto_bbox_desc', 'auto_bboxclass_desc')
    ds.write_labels(dataset_dir, categories[1:], 'auto_bboxclass_desc',
                    ds.df['auto_bbox_count']==1) ##
    ds.print(title='Test data', condition=ds.df['manual_bbox_count']>=1)
    # Test of localizer with testing data
    ds.test(train_name=loc_name_trn, test_name=test_name+'b',
            data=loc_data, task='test2') ##
    cmpath = os.path.join('runs/test', test_name+'b',
                          'confusion_matrix_counts.txt')
    ds.check_confusion_matrix(cmpath, save=True, names=['aircraft'])
    # Test of detector with testing data
    ds.test(train_name=det_name_trn, test_name=test_name+'c',
            data=det_data, task='test') ##
    cmpath = os.path.join('runs/test', test_name+'c',
                          'confusion_matrix_counts.txt')
    ds.check_confusion_matrix(cmpath, save=True, names=detector_classes)


def tflite_conversion():
    """
    Convert the model:
    PyTorch --> ONNX --> TensorFlow --> TensorFlow Lite
    """
    train_name = 'yolov7_localizer_02_01'
    model_name = 'localizer'
    ds = Dataset()

    torch_model = os.path.join('runs/train/', train_name, 'weights/best.pt')
    onnx_model = os.path.join('runs/train/', train_name, 'weights/best.onnx')
    tf_model = os.path.join('../data/model/tf', model_name)
    tflite_model = os.path.join('../data/model/tflite', model_name + '.tflite')

    ds.torch2onnx(torch_model)
    ds.onnx2tf(onnx_model, tf_model)
    ds.tf2tflite(tf_model, tflite_model)


def daemon_tflite():
    """
    Runs TFLite version of the sorting daemon
    """
    ds = Dataset()
    ds.sort_tflite('../data/edge_test/tosort',
                   '../data/edge_test/plane',
                   '../data/edge_test/noplane',
                   '../data/edge_test/log',
                   savejson=True)


if __name__ == '__main__':
    #main_sequence()
    #daemon()
    #compare_collections()
    #entry_search()
    revised_sequence()
    #tflite_conversion()
    #daemon_tflite()
