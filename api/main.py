from typing import Annotated
from fastapi import FastAPI, UploadFile, File, HTTPException, Form, BackgroundTasks
from fastapi.responses import JSONResponse
# from api.core.preprocessing import preprocessing_pipeline
# from api.core.siamese import SiameseModel
from api.core.retrain import move_uploaded_file
from api.core.retrain import augment
from api.core.retrain import csv
from api.helper import helper
import shutil
import pandas as pd
import math
import os

app = FastAPI();

ALLOWED_TYPES = ["image/jpeg", "image/jpg", "image/png"];

# @app.post("/api/calculate-similarity")
# async def calculate_similarity(
#     reference_image: Annotated[UploadFile, File()],
#     sketch_image: Annotated[UploadFile, File()]
# ):
#     # TODO: Check the extension
#     print(reference_image.content_type)
#     print(sketch_image.content_type)
#     if reference_image.content_type not in ALLOWED_TYPES or sketch_image.content_type not in ALLOWED_TYPES:
#         raise HTTPException(
#             status_code=400,
#             detail="Only JPEG / JPG / PNG images are allowed!"
#         )

#     # TODO: Read image bytes
#     reference_bytes = await reference_image.read()
#     sketch_bytes = await sketch_image.read()
#     print("Images read successfully")

#     # TODO: Convert bytes to PIL Image
#     reference_img = preprocessing_pipeline(reference_bytes)
#     sketch_img = preprocessing_pipeline(sketch_bytes)
#     print("Images preprocessed successfully")

#     # TODO: Calculate the similarity between reference image and sketch image
#     siamese_model = SiameseModel(reference_img, sketch_img)
#     similarity = float(siamese_model.calculate_similarity())
#     print(f"Similarity: {similarity}")

#     # TODO: Send the response
#     return JSONResponse({
#         "message": "Images received successfully",
#         "similarity": similarity
#     })

@app.post("/api/re-train")
def retrain(
    new_reference_name: Annotated[str, Form()],
    reference_image: Annotated[UploadFile, File()],
    sketches: Annotated[list[UploadFile], File()],
    scores: Annotated[list[int], Form()],
    background_tasks: BackgroundTasks
):
    
    # TODO: Cek apakah new_reference_name sudah pernah ada sebelumnya
    #! Cek di Supabase (huruf kecil semua)

    background_tasks.add_task(
        retrain_background, 
        new_reference_name, 
        reference_image,
        sketches,
        scores
    )

    return JSONResponse({
        "message": "OK",
    })

@app.get("/")
def root():
    return {"message": "Welcome to the root route!"}

@app.get("/health")
def health():
    return {
        "status": 'OK'
    }

def retrain_background(
    new_reference_name: str,
    reference_image: UploadFile,
    sketches: list[UploadFile],
    scores: list[int]      
):
    # TODO: Check length of sketches and scores
    if len(sketches) != len(scores):
        return HTTPException(
            status_code=400,
            detail="length of sketches and scores must be equal"
        )
    
    sketches_data = []
    scores_data = []

    # TODO: Make Base Dir based on 'new_reference_name'
    print("\n============================ MAKING BASE DIR FOR NEW REFERENCE NAME IN DATASET ============================")
    new_train_reference_path, new_train_sketches_path = move_uploaded_file.make_base_dir('train', new_reference_name)
    print(">> TRAIN DIR DONE")
    new_valid_reference_path, new_valid_sketches_path = move_uploaded_file.make_base_dir('valid', new_reference_name)
    print(">> VALID DIR DONE")
    new_valid_reference_path, new_valid_sketches_path = move_uploaded_file.make_base_dir('test', new_reference_name)
    print(">> TEST DIR DONE")

    # TODO: Make Temp Dir for this context
    slug_temp_dir = move_uploaded_file.make_temp_dir()
    temp_path = f"data/dataset/temp/{slug_temp_dir}"

    # TODO: Iterate the sketches
    print("\n============================ ITERATE THE SKETCHES ============================")
    for idx, sketch_file in enumerate(sketches):
        sketch_name = f"{helper.generate_slug()}_{sketch_file.filename}"
        sketches_data.append(sketch_name)
        scores_data.append(scores[idx])

        curr_sketch_temp_path = f"{temp_path}/{sketch_name}"
        with open(curr_sketch_temp_path, 'wb') as buffer:
            shutil.copyfileobj(sketch_file.file, buffer)
        print(f">> Copy `{sketch_file.filename}` into `{curr_sketch_temp_path}`")

    # TODO: Move the reference image
    print("\n============================ MOVE THE REFERENCE IMAGE ============================")
    curr_reference_temp_path = f"{temp_path}/ref.png"
    with open(curr_reference_temp_path, 'wb') as buffer:
        shutil.copyfileobj(reference_image.file, buffer)
        print(f">> Copy `{reference_image.filename}` into `{curr_reference_temp_path}`")

    new_reference_df = pd.DataFrame({
        "name": sketches_data,
        "score": scores_data
    })

    print("\n============================ MOVE THE REFERENCE AND THE SKETCHES INTO ORIGINAL DIRECTORY ============================")
    print(">> Generate count split (train, valid, test)")
    total_sketches = len(new_reference_df['name'])
    count_train = math.floor(total_sketches * 0.5)
    count_valid = math.floor((total_sketches - count_train) * 0.5)
    count_test = total_sketches - count_train - count_valid

    print(f">> Train: {count_train}\n>> Valid: {count_valid}\n>> Test: {count_test}")

    # Shuffle the DataFrame randomly
    print(">> Shuffle the DataFrame randomly")
    shuffled_df = new_reference_df.sample(frac=1).reset_index(drop=True)

    # Split using your computed counts
    train_df = shuffled_df.iloc[:count_train]
    valid_df = shuffled_df.iloc[count_train:count_train + count_valid]
    test_df = shuffled_df.iloc[count_train + count_valid:]

    print(">> Move into train directory")
    move_uploaded_file.copy_temp_into_original(train_df, 'train', slug_temp_dir, new_reference_name)
    print(">> Move into valid directory")
    move_uploaded_file.copy_temp_into_original(valid_df, 'valid', slug_temp_dir, new_reference_name)
    print(">> Move into test directory")
    move_uploaded_file.copy_temp_into_original(test_df, 'test', slug_temp_dir, new_reference_name)

    shutil.rmtree(temp_path)
    print(f">> Remove temp directory: {temp_path}")

    print("\n============================ IMAGE AUGMENTATION ============================")
    new_sketches_train_dirs = [
        d for d in os.listdir(f"data/dataset/train/{new_reference_name}/ref1/sketches")
    ]
    augment.augment_pipeline(new_sketches_train_dirs, new_reference_name)
    print(">> Image Augmentation Finished Successfully")

    print("\n============================ GENERATES CSV PAIRS (BOTH OF POSITIVE AND NEGATIVE) ============================")
    positive_train_df = csv.create_sketch_dataframe('data/dataset', 'train', True)
    positive_train_df.to_csv("data/csv/positive_train_pairs.csv", index=False)

    negative_train_df = csv.create_negative_pairs_from_positive_df(positive_train_df)
    negative_train_df.to_csv('data/csv/negative_train_pairs.csv', index=False)
    print(f">> Successfully created train pairs (both positive and negative)\n")

    positive_valid_df = csv.create_sketch_dataframe('data/dataset', 'valid', True)
    positive_valid_df.to_csv("data/csv/positive_valid_pairs.csv", index=False)

    negative_valid_df = csv.create_negative_pairs_from_positive_df(positive_train_df)
    negative_valid_df.to_csv('data/csv/negative_valid_pairs.csv', index=False)
    print(f">> Successfully created valid pairs (both positive and negative)\n")

    # TODO: Test
    positive_test_df = csv.create_sketch_dataframe('data/dataset', 'test', True)
    positive_test_df.to_csv("data/csv/positive_test_pairs.csv", index=False)

    print(f">> Successfully created test pairs (only positive)")
    