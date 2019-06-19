import numpy as np
from PIL import Image, ImageFont, ImageDraw

FONT_NAME = '../dataset/NotoSansCJKjp-Regular.otf'
FONT_SIZE = 40
FONT = ImageFont.truetype(
    font=FONT_NAME, size=int(FONT_SIZE * 0.85), encoding="utf-8")


def trans_to_img(word):
    imgs = [char_to_img(c) for c in word]
    imgs = np.array([resize_img(img) for img in imgs])
    return normlize(imgs).astype(np.float32)


def char_to_img(char):
    img_size = np.ceil(np.array(FONT.getsize(char)) * 1.1).astype(np.int32)
    img = Image.new('L', tuple(img_size), "black")
    text_offset = (img_size - FONT.getsize(char)) // 2
    draw = ImageDraw.Draw(img)
    draw.text(text_offset, char, font=FONT, fill="#fff")
    return img


def resize_img(img):
    arr = np.array(img)
    row, column = np.where(arr != 0)
    row.sort()
    column.sort()
    if len(row) == 0:
        b = np.zeros((FONT_SIZE, FONT_SIZE))
    else:
        top = row[0]
        bottom = row[-1]
        left = column[0]
        right = column[-1]

        c_arr = arr[top:bottom, left:right]
        b = np.zeros((FONT_SIZE, FONT_SIZE))
        r_offset = int((b.shape[0] - c_arr.shape[0]) / 2)
        c_offset = int((b.shape[1] - c_arr.shape[1]) / 2)
        b[r_offset:r_offset + c_arr.shape[0],
          c_offset:c_offset + c_arr.shape[1]] = c_arr
    return b


def normlize(arr):
    return arr * (1.0 / 255.0)
