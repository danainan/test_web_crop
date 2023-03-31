import cv2
import numpy as np
import streamlit as st
import keyboard
from PIL import Image
from tesserocr import PyTessBaseAPI
import streamlit.components.v1 as components
from transformers import AutoTokenizer , AutoModelForTokenClassification
from pythainlp.tokenize import word_tokenize
import torch
import os

def take_img():
    global frame

    cap = cv2.VideoCapture(2)
    frame = np.zeros((1280, 720, 3), dtype=np.uint8)
    st.title('Webcam Crop')
    x, y, w, h = 100, 100, 450, 250

    w = st.slider('Width', 0, 640, 450)
    h = st.slider('Height', 0, 480, 250)

    stframe = st.empty()
    stop = False
    st.write('Press q to crop')

    while stop == False:
       ret, frame = cap.read()
       cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), thickness=2)
       stframe.image(frame, channels='BGR')
       if keyboard.is_pressed('q'):
           stop = True
           crop = frame[y:y+h, x:x+w]
           cv2.imwrite('cropped/crop.jpg', crop)
           break


    cap.release()
    cv2.destroyAllWindows()



if st.button ('Take Img'):
    take_img()
    


image_path = 'cropped/crop.jpg'

if os.path.exists(image_path):
    image = Image.open(image_path)

    def original_img():
        st.write("Original Image")
        original = np.array(image)
        Image.fromarray(original).save("cropped/result.jpg")
        images = [image,original]

        image_on_row = st.columns(len(images))
        for i in range(len(images)):
            image_on_row[i].image(images[i], width=350)

    def binary_img():
        st.write("Binary")
        im_gray = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2GRAY)
        # kernel = np.ones((3, 3), np.uint8)
        # #threshold
        ret, thres = cv2.threshold(im_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        Image.fromarray(thres).save("cropped/result.jpg")
        images = [image,thres]

        image_on_row = st.columns(len(images))
        for i in range(len(images)):
            image_on_row[i].image(images[i], width=350)

    def dilation_img():
        st.write("Dilation")
        im_gray = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2GRAY)
        kernel = np.array([[1, 0, 0],
                           [0, 1, 0]],np.uint8)

        # #threshold
        ret, thres = cv2.threshold(im_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        new_img = cv2.dilate(thres, kernel, iterations=1)

        Image.fromarray(new_img).save("cropped/result.jpg")
        images = [image,new_img]

        image_on_row = st.columns(len(images))
        for i in range(len(images)):
            image_on_row[i].image(images[i], width=350)

    def erosion_img():
        st.write("Erosion")
        im_gray = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2GRAY)
        kernel = np.array([[1, 0, 0],
                           [0, 1, 0]],np.uint8)

        # #threshold
        ret, thres = cv2.threshold(im_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        new_img = cv2.erode(thres, kernel, iterations=1)

        Image.fromarray(new_img).save("cropped/result.jpg")
        images = [image,new_img]

        image_on_row = st.columns(len(images))
        for i in range(len(images)):
            image_on_row[i].image(images[i], width=350)

    def opening_img():
        st.write("Opening")
        im_gray = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2GRAY)
        kernel = np.array([[1, 0, 0],
                           [0, 1, 0]],np.uint8)
        # #threshold
        ret, thres = cv2.threshold(im_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        new_img = cv2.morphologyEx(thres, cv2.MORPH_OPEN, kernel, iterations=1)

        Image.fromarray(new_img).save("cropped/result.jpg")
        images = [image,new_img]

        image_on_row = st.columns(len(images))
        for i in range(len(images)):
            image_on_row[i].image(images[i], width=350)

    def closing_img():
        st.write("Closing")
        im_gray = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2GRAY)
        kernel = np.array([[1, 0, 0],
                           [0, 1, 0]],np.uint8)

        # #threshold
        ret, thres = cv2.threshold(im_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        new_img = cv2.morphologyEx(thres, cv2.MORPH_CLOSE, kernel, iterations=1)

        Image.fromarray(new_img).save("cropped/result.jpg")
        images = [image,new_img]

        image_on_row = st.columns(len(images))
        for i in range(len(images)):
            image_on_row[i].image(images[i], width=350)
        


    option = st.selectbox(
    'Select Pre-Processing',
    ('Original Image','Binary', 'Dilation', 'Erosion', 'Opening', 'Closing'))

    if option == 'Original Image':
        original_img()
    elif option == 'Binary':
        binary_img()
    elif option == 'Dilation':
        dilation_img()
    elif option == 'Erosion':
        erosion_img()
    elif option == 'Opening':
        opening_img()
    elif option == 'Closing':
        closing_img()

def ocr_core(img):
   if st.button('OCR'):
    with PyTessBaseAPI(path='C:/Users/User/anaconda3/share/tessdata_best-main',lang="tha+eng") as api:
        api.SetImageFile(img)
        text = api.GetUTF8Text()
        text_array = []
        text_array.append(text.replace("\n", " "))
        return text_array

st.write("OCR Result")
st.write(ocr_core(os.path.join("cropped/result.jpg")))



    

