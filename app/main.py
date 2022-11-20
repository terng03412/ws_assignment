
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, UploadFile, File, BackgroundTasks
from fastapi.responses import JSONResponse
from os import getcwd
from PIL import Image

from .train_model import train
from .utils import train_labels, predict
from .preprocess import create_test, create_train

import numpy as np


def resize_image(filename: str):
    sizes = [{
        "width": 1280,
        "height": 720
    }, {
        "width": 640,
        "height": 480
    }]

    for size in sizes:
        size_defined = size['width'], size['height']

        image = Image.open(PATH_FILES + filename, mode="r")
        image.thumbnail(size_defined)
        image.save(PATH_FILES + str(size['height']) + "_" + filename)
    print("success")


origins = [
    "http://localhost:8080",
    "http://127.0.0.1:8080/send",
    "http://localhost:4200",
    "http://localhost:8789"

]


# from pydantic import BaseModel
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
PATH_FILES = getcwd() + "/"


@app.post("/recognize")
async def upload_file(background_tasks: BackgroundTasks, file: UploadFile = File(...)):

    # SAVE FILE ORIGINAL
    with open(PATH_FILES + file.filename, "wb") as myfile:
        content = await file.read()
        myfile.write(content)
        myfile.close()

    test_image_path = PATH_FILES + file.filename
    # RESIZE IMAGES
    # background_tasks.add_task(resize_image, filename=file.filename)

    # image = Image.open(PATH_FILES + file.filename, mode="r")
    # w, h = image.size
    # print('width: ', w)
    # print('height:', h)
    embedded_path = '/code/app/files/'
    try:

        ans = predict(embedded_path, test_image_path)[1]

        return {
            "success": True,
            "account": str(ans)
        }
    except Exception:
        return {
            "success": False,
            "account": ""
        }


@app.post("/preprocess")
async def preprocess():
    create_train()
    create_test()
    return {
        "success": True
    }


@app.post("/train")
async def train():
    train_labels()
    return {
        "success": True
    }


@ app.get("/")
async def root():
    return {"message": "Welcome"}
