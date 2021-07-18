import streamlit as st
import cv2
import numpy as np
from PIL import Image
import mediapipe as mp
import face_recognition


st.title("This is a web application made for OpevCv Operation :sunglasses:") 

add_selectbox = st.sidebar.selectbox(
    "What Operation woul you like to perform",
    ("About","Colour Change","Background Change","Image Blending","Face Recogination"))

#About
if add_selectbox == "About":
    st.write("**This is a web application made for OpevCv Operation**")
    st.write("Operations are:")
    st.write("Colour Change")
    st.write("Background Change")
    st.write("Image Blending")
    st.write("Face Recogination")
#Background Colour Change

elif add_selectbox == "Colour Change" :
    st.write("**Colour Changing Options will help you to see image in different colour(Blue,Green,Red). As there are three Channel r,g,b this app remove two channel and show photo in one channel which you will select. :astonished: **")
    image = None
    image_file_path=st.sidebar.file_uploader("Upload Image")
    color_schemes = st.sidebar.radio("Choose your Colour",
                                   ("Blue","Green","Red"))
    if image_file_path is not None:
        image =np.array(Image.open(image_file_path))
        st.sidebar.image(image)
        if color_schemes == "Blue" :
            zeros = np.zeros(image.shape[:2], dtype= "uint8")
            b,g,r = cv2.split(image)
            blue_image = cv2.merge([zeros,zeros,b])
            st.image(blue_image)
        elif color_schemes == "Green" :
            zeros = np.zeros(image.shape[:2], dtype= "uint8")
            b,g,r = cv2.split(image)
            green_image = cv2.merge([zeros,g,zeros])
            st.image(green_image)
        elif color_schemes == "Red" :
            zeros = np.zeros(image.shape[:2], dtype= "uint8")
            b,g,r = cv2.split(image)
            green_image = cv2.merge([r,zeros,zeros])
            st.image(green_image)
            

#Background Changing


elif add_selectbox == "Background Change" :
    mp_drawing = mp.solutions.drawing_utils
    mp_selfie_segmentation = mp.solutions.selfie_segmentation
    segment = mp_selfie_segmentation.SelfieSegmentation()
    st.write("**Background Changing options will help you to change background of any image :heart_eyes:**")
    image = None
    image_file_path=st.sidebar.file_uploader("Upload Image whose backgroung you whant to change")
    image_file_path1=st.sidebar.file_uploader("Upload Image of background")
    if image_file_path and image_file_path1 is not None:
        Images=[]
        image1 =np.array(Image.open(image_file_path))
        st.sidebar.image(image1) 
        RGB_sample_img = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
        result= segment.process(RGB_sample_img)
        binary_mask = result.segmentation_mask > 0.9
        binary_mask_3 = np.dstack((binary_mask,binary_mask,binary_mask))
        output_image = np.where(binary_mask_3, image1, 255) 
        image2 =np.array(Image.open(image_file_path1))
        image2 = cv2.resize(image2, (image1.shape[1], image1.shape[0]))
        st.sidebar.image(image2) 
        output_image = np.where(binary_mask_3,image1,image2)
        st.image(output_image)


#Image Blending
elif add_selectbox == "Image Blending" :
    st.write("Image Blending will help you to blend two images and see the output of it :kissing_smiling_eyes:")
    image = None
    image_file_path=st.sidebar.file_uploader("Upload Image1")
    image_file_path1=st.sidebar.file_uploader("Upload Image2")
    if image_file_path and image_file_path1 is not None:
        Images=[]
        image1 =np.array(Image.open(image_file_path))
        st.sidebar.image(image1) 
        image2 =np.array(Image.open(image_file_path1))
        st.sidebar.image(image2) 
        image2 = cv2.resize(image2, (image1.shape[1], image1.shape[0]))
        value1 = st.sidebar.slider('Select Opacity of Image1' ,0.0, 1.0,value=0.7 ,step=0.1)
        value2 = st.sidebar.slider('Select Opacity of Image2',0.0, 1.0, value=0.3,step=0.1)
        blended_image = cv2.addWeighted(image1, value1, image2,value2, gamma=0.5)
        st.image(blended_image)

#Face Recogination
elif add_selectbox == "Face Recogination" :
    st.write("Face Recogination will help you to recoginized both the uploaded photos is of same person or not :anguished:")
    image = None
    image_file_path=st.sidebar.file_uploader("Upload Image1")
    image_file_path1=st.sidebar.file_uploader("Upload Image2")
    if image_file_path and image_file_path1 is not None:
        st.sidebar.image(image_file_path) 
        
        st.sidebar.image(image_file_path1) 

        face1 = face_recognition.load_image_file(image_file_path)
        face1_encodings = face_recognition.face_encodings(face1)[0]

        face2 = face_recognition.load_image_file(image_file_path1)
        face2_encodings = face_recognition.face_encodings(face2)[0]
        
        matches = face_recognition.compare_faces([face1_encodings], face2_encodings)
        if matches[0]:
            st.write("**Both image is of same person**")
            
        else:
            st.write("**Both image is of different person**")
		    