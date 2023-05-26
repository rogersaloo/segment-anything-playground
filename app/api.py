import uvicorn
from fastapi import FastAPI, File, UploadFile
from io import BytesIO
from tkinter import Image

from fine_tuning.main import predict_on_tuned_sam

app = FastAPI()


def read_image_file(file) -> Image.Image:
    """Reads image and loads it to memory

    Args:
        file (_type_): the file name

    Returns:
        Image.Image: Image loaded from dir
    """
    image = Image.open(BytesIO(file))
    return image


@app.get("/")
def root():
    """
    Root page dispay item on homepage

    """
    return {"Welcome to fine tuned sam model please insert bottle image to mask it"}


@app.post("/mask/image")
async def predict_api(file: UploadFile = File(...)):
    """
    Args:
        file (UploadFile, optional): Image to mask with the SAM model

    Returns:
        _type_: masked image using the SAM model
    """
    extension = file.filename.split(".")[-1] in ("jpg", "jpeg", "png")
    if not extension:
        return "Image must be jpg or png format!"
    image = read_image_file(await file.read())
    masked_image = predict_on_tuned_sam(image)
    return masked_image


#change the host to local host when running locally
if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=8000)
