from typing import Annotated
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from api.core.preprocessing import preprocessing_pipeline
from api.core.siamese import SiameseModel

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
    reference_img = preprocessing_pipeline(reference_bytes)
    sketch_img = preprocessing_pipeline(sketch_bytes)
    print("Images preprocessed successfully")

    # TODO: Calculate the similarity between reference image and sketch image
    siamese_model = SiameseModel(reference_img, sketch_img)
    similarity = float(siamese_model.calculate_similarity())
    print(f"Similarity: {similarity}")

    # TODO: Send the response
    return JSONResponse({
        "message": "Images received successfully",
        "similarity": similarity
    })

@app.get("/")
def root():
    return {"message": "Welcome to the root route!"}

@app.get("/health")
def health():
    return {
        "status": 'OK'
    }