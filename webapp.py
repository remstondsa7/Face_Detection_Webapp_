import streamlit as st
import cv2
from PIL import Image
import numpy as np
import tempfile
import sys
from streamlit import cli as stcli


face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

def detect_faces(img,flag):
    if flag==True:
        img = np.array(img.convert('RGB'))
        
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in faces:
        img = cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        for (ex,ey,ew,eh) in eyes:
            cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
    k = cv2.waitKey(0)
    if k == 27:         # wait for ESC key to exit
        cv2.destroyAllWindows()
    elif k == ord('s'): # wait for 's' key to save and exit
        cv2.imwrite('messigray.png',img)
        cv2.destroyAllWindows()
    return img


def main():
    """Face Recognition App"""

    st.title("WELCOME !!")

    html_temp = """
    <body style="background-color:red;">
    <div style="background-color:teal ;padding:10px">
    <h2 style="color:white;text-align:center;">Face Recognition WebApp</h2>
    </div>
    </body>
    """
    st.markdown(html_temp, unsafe_allow_html=True)

    def Recognise(our_image,flag):
        if st.button("Recognise"):
            result_img= detect_faces(our_image,flag)
            
            return st.image(result_img)

    image_file = st.file_uploader("Upload Image", type=['jpg', 'png', 'jpeg'])
    if image_file is not None:
        our_image = Image.open(image_file)
        print("IMAGE TYPE:",type(our_image))
        st.text("Original Image")
        st.image(our_image)
        Recognise(our_image, True)
        
    elif st.button("Camera"):
        cap=cv2.VideoCapture(0)
        st.text("Original Image")
        stframe=st.empty()
        
        
        if cap.isOpened():
            ret, img = cap.read()
            stframe.image(img,channels="BGR")
            print("IMAGE TYPE:",type(img))
            if not ret:
                print("Can't receive frame (stream end?). Exiting ...")

            our_image=detect_faces(img,False)
            st.image(our_image)


    
    
    def Recog_Video(video):
        if st.button("Recognise-Video"):
            if video!=0:
                cap = cv2.VideoCapture(video.name)
            else:
                cap=cv2.VideoCapture(video)
            stframe=st.empty()
          
            while cap.isOpened():
                ret, img = cap.read()
                stframe.image(img,channels="BGR")
                print("IMAGE TYPE:",type(img))
                if not ret:
                    print("Can't receive frame (stream end?). Exiting ...")
                    break
                result_video=detect_faces(img,False)
                st.image(result_video)
        
    video_file = st.file_uploader("Upload Video", type=['mp4', 'avi', 'mov'])
    
    if video_file is not None:
        our_video = tempfile.NamedTemporaryFile(delete=False)
        our_video.write(video_file.read())
        print(type(our_video))
        st.text("Original Video")
        st.video(our_video.name)
        Recog_Video(our_video)

    

    elif st.button("Camera-Video"):
        cap=cv2.VideoCapture(0)
        st.text("Original Video")
        stframe=st.empty()
        
          
        while cap.isOpened():
            ret, img = cap.read()
            stframe.image(img,channels="BGR")
            print("IMAGE TYPE:",type(img))
            if not ret:
                print("Can't receive frame (stream end?). Exiting ...")
                break
            result_video=detect_faces(img,False)
            st.image(result_video)
    
    

if __name__ == '__main__':
    if st._is_running_with_streamlit:
        main()
    else:
        sys.argv=["streamlit","run","webapp.py"]
        sys.exit(stcli.main())
        
