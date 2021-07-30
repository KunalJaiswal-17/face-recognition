import streamlit as st
import cv2
import numpy as np
from PIL import Image
import mediapipe as mp
import face_recognition

mp_drawing = mp.solutions.drawing_utils
mp_selfie_segmentation = mp.solutions.selfie_segmentation

mp_face_detection = mp.solutions.face_detection


st.set_page_config(
    page_title="Fun With Images",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("<h1 style='text-align: center; color: black;'><u>Fun With Images</u></h1>", unsafe_allow_html=True)
st.markdown("---")
st.markdown("<h2 style='text-align: center; color: black;'>Perform different operations on images.</h2>", unsafe_allow_html=True)
st.markdown("---")

add_selectbox = st.sidebar.selectbox(
    "Which operation you want to perform out of these?",
    ('-SELECT-', 'About', 'Selfie Segmentation', 'Face Detection', 'Face Recognition'),
)

if add_selectbox == 'SELECT':
    st.markdown("<h3 style='text-align: center; color: black;'>Open the sidebar and start the fun.</h3>", unsafe_allow_html=True)

if add_selectbox =='About':
    image1 = Image.open('1.jpg')
    st.write('This application performs various operations with OpenCV')
    st.write('This application contain total of 4 operations:')
    st.markdown('1. __Selfie Segmentation__ : *Selfie Segmentation* allows developers to easily separate the background from users within a scene and focus on what matters.')
    st.markdown('*For Example:*')
    image2 = Image.open('2.jpg')
    image3 = Image.open('3.jpg')
    col1, col2, col3 = st.beta_columns(3)
    col1.image(image1, caption = 'Original Image')
    col2.image(image2, caption = 'Segmented Image - 1')
    col3.image(image3, caption = 'Segmented Image - 2')
    st.markdown('2. __Face Detection__ : *Face Detection* is the first and essential step for face recognition, and it is used to detect faces in the images.')
    st.markdown('*For Example:*')
    image4 = Image.open('4.jpg')
    col4, col5 = st.beta_columns(2)
    col4.image(image1, caption = 'Original Image')
    col5.image(image4, caption = 'Face Detected Image')
    st.markdown('3. __Face Recognition__ : *Face Recognition* system is a technology capable of matching a human face from a digital image or a video frame against a database of faces,' 
    'typically employed to authenticate users through ID verification services, works by pinpointing and measuring facial features from a given image')
    st.markdown('*For Example:*') 
    st.markdown('Here we are comparing two image 1 and 2 and the third image is showing whether it is same person or not, if it is same then it will show true and by how much distance.' 
    'Here distance means how similar the faces are.  Lower is more strict. 0.6 is typical best performance. ')
    image5 = Image.open('5.jpg')
    image6 = Image.open('6.jpg')
    col6, col7, col8 = st.beta_columns(3)
    col6.image(image1, caption = 'Image 1')
    col7.image(image5, caption = 'Image 2')
    col8.image(image6, caption = 'Recognized Face')

elif add_selectbox == 'Selfie Segmentation':
    image_path = st.sidebar.file_uploader('Upload File', type = ['png', 'jpg', 'jpeg'])
    if image_path is not None:
        st.sidebar.write('Image uploaded sucessfully.')
    st.sidebar.markdown('---')
    if image_path is not None:
        image = np.array(Image.open(image_path))
        radio_option = st.sidebar.radio(
            'Select 1st two, to change background color and the next two, to change background image',
            ['Selct below: ', 'Cyan', 'Magenta', 'Mountain Scenery', 'City View']
        )

        if radio_option == 'Select below:':
            None
        
        elif radio_option == 'Cyan':
            bg_color1 = None
            model = mp_selfie_segmentation.SelfieSegmentation(model_selection = 1)
            results= model.process(image)
            condition = np.stack((results.segmentation_mask,) * 3, axis = -1) > 0.1
            if bg_color1 is None:
                bg_color1 = np.zeros(image.shape, dtype=np.uint8)
                bg_color1[:]= (0, 255, 255)
            bg_color = cv2.resize(bg_color1, (image.shape[1], image.shape[0]) )
            output_image = np.where(condition, image, bg_color)
            col1, col2 = st.beta_columns(2)
            col1.image(image, caption = 'Original File')
            col2.image(output_image, caption = 'With Cyan Background')

        elif radio_option == 'Magenta':
            bg_color2 = None
            model = mp_selfie_segmentation.SelfieSegmentation(model_selection = 1)
            results= model.process(image)
            condition = np.stack((results.segmentation_mask,) * 3, axis = -1) > 0.1
            if bg_color2 is None:
                bg_color2 = np.zeros(image.shape, dtype=np.uint8)
                bg_color2[:]= (255, 0, 106)
            bg_color = cv2.resize(bg_color2, (image.shape[1], image.shape[0]) )
            output_image = np.where(condition, image, bg_color)
            col1, col2 = st.beta_columns(2)
            col1.image(image, caption = 'Original File')
            col2.image(output_image, caption = 'With Magenta Background')

        elif radio_option == 'Mountain Scenery':
            bg_image1 = np.array(Image.open('Mountain.jpg'))
            model = mp_selfie_segmentation.SelfieSegmentation(model_selection = 1)
            results= model.process(image)
            condition = np.stack((results.segmentation_mask,) * 3, axis = -1) > 0.1
            bg_image = cv2.resize(bg_image1, (image.shape[1], image.shape[0]) )
            output_image = np.where(condition, image, bg_image)
            col1, col2 = st.beta_columns(2)
            col1.image(image, caption = 'Original File')
            col2.image(output_image, caption = 'With Mountain in Background')

        elif radio_option == 'City View':
            bg_image2 = np.array(Image.open('city.jpg'))
            model = mp_selfie_segmentation.SelfieSegmentation(model_selection = 1)
            results= model.process(image)
            condition = np.stack((results.segmentation_mask,) * 3, axis = -1) > 0.1
            bg_image = cv2.resize(bg_image2, (image.shape[1], image.shape[0]) )
            output_image = np.where(condition, image, bg_image)
            col1, col2 = st.beta_columns(2)
            col1.image(image, caption = 'Original File')
            col2.image(output_image, caption = 'With City in Background')

elif add_selectbox == 'Face Detection':
    image_path = st.sidebar.file_uploader('Upload File', type = ['png', 'jpg', 'jpeg'])
    if image_path is not None:
        st.sidebar.write('Image uploaded sucessfully.')
    st.sidebar.markdown('---')
    if image_path is not None:
        image = np.array(Image.open(image_path))
        col1, col2 = st.beta_columns(2)
        col1.image(image, caption = 'Original File')

        # Model for detecting face
        model_detection = mp_face_detection.FaceDetection()
        results = model_detection.process(image)
        for landmark in results.detections:
            mp_drawing.draw_detection(image, landmark)
        col2.image(image, caption = 'Detected Face')

elif add_selectbox == 'Face Recognition':
    image_paths = st.sidebar.file_uploader('Upload File', type = ['png', 'jpg', 'jpeg'])
    
    if image_paths is not None:
        st.sidebar.write('Image uploaded sucessfully.')
    st.sidebar.markdown('---')
    image_paths1 = st.sidebar.file_uploader('Upload 2nd File', type = ['png', 'jpg', 'jpeg'])        
    if image_paths is not None:
        col, col1 = st.beta_columns(2)
        image_1 = np.array(Image.open(image_paths))
        col.image(image_1, caption = 'Image 1')
        if image_paths1 is not None:
            st.sidebar.write('Image uploaded sucessfully.')         
            image_2 = np.array(Image.open(image_paths1))       
            col1.image(image_2, caption = 'Image 2')

            st.markdown('---')
            st.markdown("<h2 style='text-align: center; color: black;'><u>Face Recognition</u></h2>", unsafe_allow_html=True)
            st.write(' ')

            col, col1, col2 = st.beta_columns(3)
            col.image(image_1, caption = 'Image 1')
            col1.image(image_2, caption = 'Image 2')
            im_train = image_1
            im_encoding_train = face_recognition.face_encodings(im_train)[0]
            im_location_train = face_recognition.face_locations(im_train)[0]

            im_test = image_2
            im_encoding_test = face_recognition.face_encodings(im_test)[0]

            results = face_recognition.compare_faces([im_encoding_test], im_encoding_train)[0]
            distance = face_recognition.face_distance([im_encoding_train], im_encoding_test)

            if results:
                cv2.rectangle(im_train, (im_location_train[3], im_location_train[0]),
                    (im_location_train[1], im_location_train[2]), (255, 0, 0), 2)
                cv2.putText(im_train,f"{results} {distance}",
                    (95, 350), cv2.FONT_HERSHEY_DUPLEX, 1, (66, 245, 215), 2)
                col2.image(im_train, caption = 'Result')
                st.markdown(f'Result was __{results}__ and distance was __{distance}__')
            else:
                st.warning(f"Could not recognize the face. Result was {results} and distance was {distance}")
