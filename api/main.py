from typing import Annotated
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from mangum import Mangum
import random

app = FastAPI();

ALLOWED_TYPES = ["image/jpeg", "image/jpg", "image/png"];

@app.post("/calculate-similarity")
async def calculate_similarity(
    reference_image: Annotated[UploadFile, File()],
    sketch_image: Annotated[UploadFile, File()]
):
    # TODO: Check the extension
    print(reference_image.content_type);
    print(sketch_image.content_type);
    if reference_image.content_type not in ALLOWED_TYPES or sketch_image.content_type not in ALLOWED_TYPES:
        raise HTTPException(
            status_code=400,
            detail="Only JPEG / JPG / PNG images are allowed!"
        )

    # TODO: Read image bytes
    reference_bytes = await reference_image.read();
    sketch_bytes = await sketch_image.read();

    # TODO: Calculate the similarity between reference image and sketch image
    similarity = round(random.uniform(80.0, 100.0), 3)

    # TODO: Send the response
    return JSONResponse({
        "message": "Images received successfully",
        "reference": reference_image.filename,
        "sketch": sketch_image.filename,
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

handler = Mangum(app, lifespan="off")