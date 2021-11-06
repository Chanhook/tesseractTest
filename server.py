###########pythoncode.py###############
import numpy as np
import sys
import os
from fastapi import FastAPI, UploadFile, File
from starlette.requests import Request
import io
import cv2
import pytesseract
import re
from pydantic import BaseModel


def read_img(img):
    pytesseract.pytesseract.tesseract_cmd = '/app/.apt/usr/bin/tesseract'
    # pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
    custom_config = '--psm 1 -c preserve_interword_spaces=1'
    text = pytesseract.image_to_string(img, lang='eng', config=custom_config)
    return(text)


app = FastAPI()


class ImageType(BaseModel):
    url: str


@app.post("/predict/")
def prediction(request: Request, file: bytes = File(...)):
    if request.method == "POST":
        image_stream = io.BytesIO(file)
        image_stream.seek(0)
        file_bytes = np.asarray(bytearray(image_stream.read()), dtype=np.uint8)
        frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        label = read_img(frame)
        return label
    else:
        return "No post request found"
