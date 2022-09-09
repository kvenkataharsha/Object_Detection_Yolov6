# from flask import Flask, Response, render_template
# import cv2
from main_img import imagrec

# app = Flask(__name__, template_folder='./templates')
# video = cv2.VideoCapture(0)
# @app.route('/')
# def index():
#     # return "Default Message"
#     return render_template('index.html')
# def gen(video):
#     while True:
#         success, image = video.read()
#         # ret, jpeg = cv2.imencode('.jpg', image)
#         # cv2.imshow('frame', image)
#         result = imagrec(image)
#         ret, jpeg = cv2.imencode('.jpg', result)
#         frame = jpeg.tobytes()
#         yield (b'--frame\r\n'
#                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
# @app.route('/video_feed')
# def video_feed():
#     global video
#     return Response(gen(video),
#                     mimetype='multipart/x-mixed-replace; boundary=frame')
# if __name__ == '__main__':
#     app.run(host='0.0.0.0', port=5000, threaded=True)

from flask import Flask, render_template, Response, request
import cv2
import datetime, time
import os, sys
import numpy as np
from threading import Thread


global capture,rec_frame, switch, face, out 
capture=0
face=0
switch=1

#make shots directory to save pics
try:
    os.mkdir('./shots')
except OSError as error:
    pass


#instatiate flask app  
app = Flask(__name__, template_folder='./templates')


camera = cv2.VideoCapture(0)



def detect_face(frame):
    start = time.time()
    result = imagrec(frame)
    end = time.time() - start
    print(end)
    return result
 

def gen_frames():  # generate frame by frame from camera
    global out, capture,rec_frame
    while True:
        success, frame = camera.read() 
        if success:
            if(face):                
                frame= detect_face(frame)
            if(capture):
                capture=0
                now = datetime.datetime.now()
                p = os.path.sep.join(['shots', "shot_{}.png".format(str(now).replace(":",''))])
                cv2.imwrite(p, frame)
            try:
                ret, buffer = cv2.imencode('.jpg', frame)
                frame = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            except Exception as e:
                pass
                
        else:
            pass


@app.route('/')
def index():
    return render_template('index.html')
    
    
@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/requests',methods=['POST','GET'])
def tasks():
    global switch,camera
    if request.method == 'POST':
        if request.form.get('click') == 'Capture':
            global capture
            capture=1
        elif  request.form.get('face') == 'Detect Objects':
            global face
            face=not face 
            if(face):
                time.sleep(4)   
                          
                 
    elif request.method=='GET':
        return render_template('index.html')
    return render_template('index.html')


if __name__ == '__main__':
    app.run()
    
camera.release()
cv2.destroyAllWindows()     