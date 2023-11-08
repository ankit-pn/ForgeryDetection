from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import StreamingResponse
from typing import Optional
from x2 import error_level_analysis
import cv2
import io
from x1 import process_image_with_scribbles
app = FastAPI()

@app.post("/process_image")
async def process_image(
    image: UploadFile = File(...), 
    quality: int = Form(...), 
    contour_value: float = Form(...)
):
    print(quality)
    # Read image as bytes
    image_bytes = await image.read()

    # Process the image
    processed_image = error_level_analysis(image_bytes, quality, contour_value)

    # Encode the processed image to bytes
    _, buffer = cv2.imencode('.jpg', processed_image)
    processed_image_bytes = buffer.tobytes()

    # Convert the processed image bytes to a file-like object
    image_stream = io.BytesIO(processed_image_bytes)

    # Create a StreamingResponse, set media type to image/jpeg
    response = StreamingResponse(image_stream, media_type="image/jpeg")

    # Set custom headers (optional)
    response.headers["Content-Disposition"] = f"attachment; filename=processed_{image.filename}"

    return response



@app.post("/process_scribble_images")
async def process_scribble_images(
    image1: UploadFile = File(...), 
    threshold_area: int = Form(...)
):
    # Read image as bytes
    image1_bytes = await image1.read()
    # image2_bytes = await image2.read()

    # Process the images
    processed_image1 = process_image_with_scribbles(image1_bytes, threshold_area)
    # processed_image2 = process_image_with_scribbles(image2_bytes, threshold_value, threshold_area)

    # TODO: You might want to combine these images or handle them separately
    # For this example, we'll just return the first processed image

    # Encode the processed image to bytes
    _, buffer = cv2.imencode('.jpg', processed_image1)
    processed_image_bytes = buffer.tobytes()

    # Convert the processed image bytes to a file-like object
    image_stream = io.BytesIO(processed_image_bytes)

    # Create a StreamingResponse, set media type to image/jpeg
    response = StreamingResponse(image_stream, media_type="image/jpeg")

    # Set custom headers (optional)
    response.headers["Content-Disposition"] = f"attachment; filename=processed_{image1.filename}"

    return response
# Run the app with: uvicorn your_file_name:app --reload
