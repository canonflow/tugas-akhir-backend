import os
import pandas as pd
import numpy as np
import random
import shutil
from fastapi import UploadFile
from api.helper import helper

def make_base_dir(dir_type, reference_name):
    dir_type = dir_type.lower()
    if dir_type not in ['train', 'valid', 'test']:
        raise Exception("Dir Type must be either train, valid, or test!")
    base_train_path = f"data/dataset/{dir_type}"
    new_reference_path = f"{base_train_path}/{reference_name}/ref1"
    new_sketches_path = f"{new_reference_path}/sketches"

    os.makedirs(new_reference_path, exist_ok=True)
    os.makedirs(new_sketches_path, exist_ok=True)

    return [new_reference_path, new_sketches_path]

def make_temp_dir():
    slug = helper.generate_slug()
    temp_path = f"data/dataset/temp/{slug}"
    os.makedirs(temp_path, exist_ok=True)
    return slug

def copy_temp_into_original(dataframe, dir_type, temp_slug, reference_name):
    dir_type = dir_type.lower()

    if dir_type not in ['train', 'valid', 'test']:
        raise Exception("`dir_type` must be either train, valid, or test!")
    
    # TODO: Setup the directories
    temp_dir = f"data/dataset/temp/{temp_slug}"
    original_ref_dir = f"data/dataset/{dir_type}/{reference_name}/ref1"
    original_sketches_dir = f"data/dataset/{dir_type}/{reference_name}/ref1/sketches"
    

    # TODO: Move the reference image
    shutil.copy(temp_dir + "/ref.png", original_ref_dir + "/ref.png")

    # TODO: Iterates the dataframe
    folder_names = []
    scores = []
    for index, row in dataframe.iterrows():
        name = row['name']
        score = row['score']

        # TODO: Make folder for each sketch
        slug = helper.generate_slug()
        current_dir = f"{original_sketches_dir}/{slug}"
        os.makedirs(current_dir, exist_ok=True)

        # TODO: Copy the sketch
        shutil.copy(f"{temp_dir}/{name}", f"{current_dir}/{name}")

        # TODO: Insert the scores
        folder_names.append(f"{slug}")
        scores.append(score)

        print("---", name, score)
    
    # TODO: Save the dataframe into .csv file
    scores_df = pd.DataFrame({
        "folder_name": folder_names,
        "score": scores
    });
    
    scores_df.to_csv(f"{original_ref_dir}/scores-full.csv", header=False, index=False)
    print(f"--- Copy temp into original: {dir_type} DONE")


    
