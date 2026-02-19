from typing import Annotated
from fastapi import FastAPI, UploadFile, File, HTTPException, Form, BackgroundTasks, Depends, status
from fastapi.responses import JSONResponse
# from api.core.preprocessing import preprocessing_pipeline
from api.core.siamese import SiameseModel, build_siamese, build_callbacks
from api.core.retrain import move_uploaded_file
from api.core.retrain import augment
from api.core.retrain import csv
from api.helper import helper
import shutil
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error
import math
import os
import random
from api.core.preprocessing import preprocessing_pipeline, create_processed_dataframe_from_csv
from api.core.siamese import (
    ReduceMeanLayer, 
    ReduceMaxLayer, 
    L2NormalizationLayer, 
    CosineSimilarityLayer,
    pearson_correlation_metric
)
from dotenv import load_dotenv
from api.repository.supabase_repository import get_supabase_client
from supabase._sync.client import Client
from PIL import Image
import io
load_dotenv()

# print(os.environ.get("SUPABASE_URL"))

app = FastAPI();

ALLOWED_TYPES = ["image/jpeg", "image/jpg", "image/png"];

@app.post("/api/calculate-similarity")
async def calculate_similarity(
    reference_image: Annotated[UploadFile, File()],
    sketch_image: Annotated[UploadFile, File()]
):
    # TODO: Check the extension
    print(reference_image.content_type)
    print(sketch_image.content_type)
    if reference_image.content_type not in ALLOWED_TYPES or sketch_image.content_type not in ALLOWED_TYPES:
        raise HTTPException(
            status_code=400,
            detail="Only JPEG / JPG / PNG images are allowed!"
        )

    # TODO: Read image bytes
    reference_bytes = await reference_image.read()
    sketch_bytes = await sketch_image.read()
    print("Images read successfully")

    # TODO: Convert bytes to PIL Image
    reference_img = preprocessing_pipeline(reference_bytes, image_size=224)
    sketch_img = preprocessing_pipeline(sketch_bytes, image_size=224)
    print("Images preprocessed successfully")

    # TODO: Calculate the similarity between reference image and sketch image
    siamese_model = SiameseModel(reference_img, sketch_img)
    similarity = round(float(siamese_model.calculate_similarity()), 2)
    print(f"Similarity: {similarity}")

    # TODO: Send the response
    return JSONResponse({
        "message": "Images received successfully",
        "similarity": similarity
    })

@app.post("/api/re-train")
def retrain(
    new_reference_name: Annotated[str, Form()],
    reference_image: Annotated[UploadFile, File()],
    sketches: Annotated[list[UploadFile], File()],
    scores: Annotated[list[int], Form()],
    background_tasks: BackgroundTasks,
    supabase_client: Client = Depends(get_supabase_client)
):
    
    if reference_image.content_type not in ALLOWED_TYPES:
        raise HTTPException(
            status_code=400,
            detail="Only JPEG / JPG / PNG images are allowed!"
        )
    # TODO: Cek apakah new_reference_name sudah pernah ada sebelumnya
    #! Cek di Supabase (huruf kecil semua)
    print(f"LEN SCORES: {len(scores)}")
    print(f"LEN SKETCHES: {len(sketches)}")
    print(f"NEW REFERENCE NAME: {new_reference_name}")
    print(f"REFERENCE IMAGE FILENAME: {reference_image.filename}")

    if len(sketches) != len(scores):
        print("!!! ERROR LEN SKETCHES != SCORES !!!")
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={
                "message": "ERROR",
            }
        )
        # return HTTPException(
        #     status_code=400,
        #     detail="length of sketches and scores must be equal"
        # )

    background_tasks.add_task(
        retrain_background, 
        new_reference_name, 
        reference_image,
        sketches,
        scores,
        supabase_client
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

def set_seeds(seed=42):
    np.random.seed(seed)
    random.seed(seed)
    tf.random.set_seed(seed)
    # For GPU determinism (may slow down training)
    tf.config.experimental.enable_op_determinism()
    tf.config.optimizer.set_jit(False)

async def retrain_background(
    new_reference_name: str,
    reference_image: UploadFile,
    sketches: list[UploadFile],
    scores: list[int],
    supabase_client: Client
):
    # TODO: Check length of sketches and scores
    if len(sketches) != len(scores):
        print("!!! ERROR LEN SKETCHES != SCORES !!!")
        return HTTPException(
            status_code=400,
            detail="length of sketches and scores must be equal"
        )
    
    try:
        # TODO: Upload the new reference and sketches into Bucket
        print("\n============================ UPLOADING NEW REFERENCE INTO SUPABASE BUCKET ============================")
        file_extension = reference_image.filename.split('.')[-1].lower()  # Normalize to lowercase
        print(">> File Extension:", file_extension)
        print(">> Client-Provided Content Type:", reference_image.content_type)

        # Map extensions to valid MIME types for images
        mime_types = {
            'jpg': 'image/jpeg',
            'jpeg': 'image/jpeg',
            'png': 'image/png',
            'gif': 'image/gif',
            'bmp': 'image/bmp',
            'webp': 'image/webp'
        }

        # Validate extension and get MIME type
        if file_extension not in mime_types:
            raise ValueError(f"Unsupported file extension: .{file_extension}. Allowed: {list(mime_types.keys())}")
        content_type = mime_types[file_extension]
        print(">> Using Content Type:", content_type)

        new_reference_filename = f"{new_reference_name.lower()}.{file_extension}"
        reference_image_copy = reference_image.file

        # Read file content
        contents = await reference_image.read()
        print(">> Read reference image contents")

        #* Upload to Supabase Bucket with explicit content-type
        file_options = {"content-type": content_type}  # Critical fix: set correct MIME type
        bucket_response = supabase_client.storage.from_('anchor').upload(
            path=new_reference_filename,
            file=contents,
            file_options=file_options  # Pass headers here
        )
        print(f">> Upload reference image into Supabase Bucket: {bucket_response}")
        public_url = supabase_client.storage.from_('anchor').get_public_url(new_reference_filename)

        # TODO: Bikin Record Baru di Supabase
        data = {
            "name": new_reference_name.lower(),
            "status": "on training",
            "image": public_url
        }
        supabase_response = supabase_client.table('anchors').insert(data).execute()
        anchor_id = supabase_response.data[0]['id']
        print(f">> Insert new record into Supabase Anchors Table: {supabase_response}")
        print(f">> New Anchor ID: {anchor_id}")
        # TODO: Tambah Try-Catch

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
        # curr_reference_temp_path = f"{temp_path}/ref.png"
        # with open(curr_reference_temp_path, 'wb') as buffer:
        #     shutil.copyfileobj(reference_image_copy, buffer)
        #     print(f">> Copy `{reference_image.filename}` into `{curr_reference_temp_path}`")

        curr_reference_temp_path = f"{temp_path}/ref.png"  # Always use PNG for dataset consistency
        # CRITICAL FIX: Convert to PNG if needed and save using our bytes
        if file_extension == 'png':
            # Directly use the bytes if it's already PNG
            with open(curr_reference_temp_path, 'wb') as buffer:
                buffer.write(contents)
        else:
            # Convert to PNG format for dataset consistency
            img = Image.open(io.BytesIO(contents))
            
            # Handle transparency if present
            if img.mode in ('RGBA', 'LA') or (img.mode == 'P' and 'transparency' in img.info):
                img = img.convert('RGB')
            
            img.save(curr_reference_temp_path, 'PNG')
            print(f">> Converted reference image to PNG format")

        print(f">> Saved reference image to `{curr_reference_temp_path}`")

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


        print("\n============================ PROCESS AND LOAD DATASET ============================")
        X_train, y_train = create_processed_dataframe_from_csv('train', 224)
        print(">> Processing Train Pairs")

        X_valid, y_valid = create_processed_dataframe_from_csv("valid", 224)
        print(">> Processing Valid Pairs")

        print("\n============================ SETUP CALLBACK ============================")
        callbacks = build_callbacks()

        print("\n============================ SETUP MODEL ============================")
        siamese_model = build_siamese(input_shape=224)
        history = siamese_model.fit(
            x=X_train,
            y=y_train,
            validation_data=(X_valid, y_valid),
            batch_size=32,
            callbacks=callbacks,
            epochs=10,
            verbose=1
        )
        
        print("\n============================ GET TRAINING METRICS ============================")
        '''
        {
            'loss': 0.421,
            'mse': 0.421,
            'pearson_correlation_metric': 0.8,
            'r2_score': 0.532,
            'val_loss': 0.321,
            'val_mse': 0.321,
            'val_pearson_correlation_metric': 0.5,
            'val_r2_score': 0.232,
        }
        '''
        latest_metrics = {
            metric: values[-1] for metric, values in history.history.items()
        }
        print(f">> Training Metrics: {latest_metrics}")

        print("\n============================ PREDICT TEST SET ============================")
        X_test, y_test = create_processed_dataframe_from_csv('test', 224)
        predictions = siamese_model.predict(X_test, batch_size=32, verbose=1)

        if predictions.ndim > 1 and predictions.shape[1] == 1:
            predictions = predictions.flatten()


        print("\n============================ GET PREDICTION METRIC ============================")
        test_r2 = r2_score(y_test, predictions)
        print(f">> Test R2 Score: {test_r2}")
        test_pearson_correlation_metric = np.corrcoef(y_test, predictions)[0][1]
        print(f">> Test Pearson: {test_pearson_correlation_metric}")
        test_mse = mean_squared_error(y_test, predictions)
        print(f">> Test MSE: {test_mse}")

        latest_metrics['test_r2'] = test_r2
        latest_metrics['test_pearson_correlation_metric'] = test_pearson_correlation_metric
        latest_metrics['test_mse'] = test_mse

        print(f"\n\nFINAL METRICS:\n{latest_metrics}")

        print("\n============================ SAVE MODEL ============================")
        siamese_model.save("model/siamese_model.keras")
        
        print("\n============================ CALL SUPABASE ============================")
        ''' NOTE
        1. Update data kalau status training untuk referensi baru sudah berhasil 'success'
        2. Masukkin Metrics-nya untuk model terbaru ke Supabase
        3. Jangan Lupa Try-Catch, kalau ada error, update status 'failed'
        '''
        update_data = {
            "status": "active",
        }
        update_response = supabase_client.table('anchors').update(update_data).eq('id', anchor_id).execute()
    except Exception as e:
        print(f"ERROR DURING RETRAINING: {e}")
        update_data = {
            "status": "inactive",
        }
        update_response = supabase_client.table('anchors').update(update_data).eq('id', anchor_id).execute()