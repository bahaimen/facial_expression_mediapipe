import os
import glob

import cv2
import cvlib as cv

import numpy as np
import pandas as pd

import json

# ai hub 한국인 감정인식을 위한 복합영상csv

class DataHandle():
    def __init__(self):
        self.db= []
        self.X_position= (0,0)
        self.Y_position= (0,0)

        self.emotion= {
            'angry': 0,
            'neutral': 1,
            'sad': 2
        }

    def _save_crop_img(self):
        try:
            img = self.img.copy()
            roi = img[
            self.Y_position[0]:

            ]



