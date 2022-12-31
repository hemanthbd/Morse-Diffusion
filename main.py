import mediapipe as mp
import numpy as np
import cv2
from scipy.spatial import distance
from collections import deque
import argparse

import keras_cv
import matplotlib.pyplot as plt

CHAR_TO_MORSE = {

    # Alphabets
    "a": ".-",
    "b": "-...",
    "c": "-.-.",
    "d": "-..",
    "e": ".",
    "f": "..-.",
    "g": "--.",
    "h": "....",
    "i": "..",
    "j": ".---",
    "k": "-.-",
    "l": ".-..",
    "m": "--",
    "n": "-.",
    "o": "---",
    "p": ".--.",
    "q": "--.-",
    "r": ".-.",
    "s": "...",
    "t": "-",
    "u": "..-",
    "v": "...-",
    "w": ".--",
    "x": "-..-",
    "y": "-.--",
    "z": "--..",
    
    # Numbers
    "0": "-----",
    "1": ".----",
    "2": "..---",
    "3": "...--",
    "4": "....-",
    "5": ".....",
    "6": "-....",
    "7": "--...",
    "8": "---..",
    "9": "----.",
    
    # Punctuation
    "&": ".-...",
    "'": ".----.",
    "@": ".--.-.",
    ")": "-.--.-",
    "(": "-.--.",
    ":": "---...",
    ",": "--..--",
    "=": "-...-",
    "!": "-.-.--",
    ".": ".-.-.-",
    "-": "-....-",
    "+": ".-.-.",
    '"': ".-..-.",
    "?": "..--..",
    "/": "-..-.",
    " ": "/"

}

morse_to_character = {morse: character for character, morse in CHAR_TO_MORSE.items()}


#def morse_to_character():
#    return {morse: character for character, morse in CHAR_TO_MORSE.items()}


def cartoon(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #blur = cv2.medianBlur(gray, 7)
    blur = cv2.bilateralFilter(gray, 5, 5, 5)
    edges = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 3, 5)
    frame_edge = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
    return frame_edge


def plot_images(images):
    plt.figure(figsize=(20, 20))
    for i in range(len(images)):
        ax = plt.subplot(1, len(images), i + 1)
        #plt.clf()
        plt.imshow(images[i])
        plt.axis("off")

    plt.show()
    return None


def keras_stable_diffusion(prompt_clean,batch_size=3,img_height=512,img_width=512):

    model = keras_cv.models.StableDiffusion(jit_compile=True,img_height=img_height,img_width=img_width)
    #keras.mixed_precision.set_global_policy("mixed_float16")
    images = model.text_to_image(f"{prompt_clean}", batch_size=batch_size)

    plot_images(images)

    return None


def cal_pnp(frame,face_2d,face_3d):
    cam_matrix = np.array([ [frame.shape[1], 0, frame.shape[0] / 2],
                            [0, frame.shape[1], frame.shape[1] / 2],
                            [0, 0, 1]])


    # Solve PnP
    _, rot_vec, _ = cv2.solvePnP(np.array(face_3d, dtype=np.float64), np.array(face_2d, dtype=np.float64), cam_matrix, np.zeros((4, 1), dtype=np.float64),flags=cv2.SOLVEPNP_ITERATIVE)

    # Get rotational matrix
    rmat, _ = cv2.Rodrigues(rot_vec)

    # Get angles
    angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)

    return angles

def eye_aspect_ratio(p2,p6,p3,p5,p1,p4):
    A = distance.euclidean(p2, p6)
    B = distance.euclidean(p3, p5)
    C = distance.euclidean(p1, p4)
    ear = (A + B) / (2.0 * C)
    return ear


def head_morse(cartoonize="False",webcam_port=0):

    prompt_final = []
    prompt_inter = ""
    prompt = []
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)

    webcam = cv2.VideoCapture(webcam_port)
    #webcam = cv2.VideoCapture("C:\\Users\\User Default\\Videos\\first_dog2.mp4")

    flag1 = 0
    flag2 = 0
    space_keep=0
    while webcam.isOpened():
        _, frame = webcam.read()

        frame = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)

        frame.flags.writeable = False
        
        results = face_mesh.process(frame)
        
        frame.flags.writeable = True
        
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        if cartoonize=="True":
            frame = cartoon(frame)    

        face_3d = []
        face_2d = []
        landmark_lst = [33, 263, 1,61, 291, 199]

        '''

        reye_landmark = [362,386,374,263]
        leye_landmark = [33,159,243,145]
        nose_landmark = [1,195]
        mouth_landmark = [61,291,0,17]
        head_landmark = [8]
        chin_landmark = [18,199, 200]

        landmark_all = reye_landmark + leye_landmark + nose_landmark + mouth_landmark + head_landmark + chin_landmark
        '''

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:

                face_2d = [[int(lm.x * frame.shape[1]),int(lm.y * frame.shape[0])] for idx, lm in enumerate(face_landmarks.landmark) if idx in sorted(landmark_lst)]
                face_3d = [[int(lm.x * frame.shape[1]),int(lm.y * frame.shape[0]),lm.z] for idx, lm in enumerate(face_landmarks.landmark) if idx in sorted(landmark_lst)]

                angles = cal_pnp(frame,face_2d,face_3d)

                vert = angles[0] * 360
                hor = angles[1] * 360

                if hor < -10:
                    text = "."
                elif hor > 5:
                    text = "-"
                elif vert < -5:
                    text = "/"
                elif vert > 15:
                    #print(ok)
                    text = "Pop previous"
                    try:
                        prompt_final.pop()
                    except:
                        print("Its Empty already")
                else:
                    text = "Center"
                    #flag1 = 1

                #print("TEXT",text)
                #print("PROMPT FINAL",prompt_final, len(prompt_final))
                #print("PROMPT",prompt)
                #print(flag1)

                

                if text=='/' and len(prompt_final)==0 and flag1>0:
                    flag2+=flag1

                    cv2.putText(frame, f'For space : {40-flag2}', (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 102, 0), 4)

                    if flag2>40:
                        cv2.putText(frame, "Space entered!", (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 102, 0), 4)

                        #print('Buffer for Space')
                        prompt_inter += " "
                        flag2=0
                        flag1=0
                        space_keep=0                                         
                
                
                '''
                if flag1>0 and space_keep>0 and len(prompt_final)==0 and text=="Center":
                    flag2+=flag1
                    print(flag2)
                
                if flag2>30 and space_keep>0:
                    print('Buffer for Space')
                    prompt_inter += " "
                    flag2=0
                    flag1=0
                    space_keep=0
                    print(ok)
                '''
                if len(prompt)==0 and text!="Center" and text!='/' and text!='Pop previous':
                    #print('Step1')
                    #if text!="Center" and text!='/':
                    flag1=0
                    flag2=0
                    prompt.append(text)

                if len(prompt)==1 and text=="Center":
                    prompt_final.append(prompt[0])
                    prompt=[]
                

                if text=='/' and len(prompt_final)>0:
                    if morse_to_character.get(''.join(prompt_final))=='.':

                        webcam.release()
                        cv2.destroyAllWindows() 
                    
                    try:     
                        prompt_inter += morse_to_character.get(''.join(prompt_final))

                        prompt_final = []
                        flag1=1
                        flag2=0
                        space_keep+=1
                    except:
                        pass
                

                cv2.putText(frame, ''.join(prompt_final), (40, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

                cv2.putText(frame, prompt_inter, (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (128, 0, 128), 2)

        cv2.putText(frame, ".(Dit)", (20, int(webcam.get(4)/2)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        cv2.putText(frame, "-(Dah)", (int(webcam.get(3))-80, int(webcam.get(4)/2)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        cv2.putText(frame, "Pop Previous Dit/Dah", (int(webcam.get(3)/2)-50, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        cv2.putText(frame, "Finish with letter", (int(webcam.get(3)/2)-70, int(webcam.get(4))-50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        cv2.putText(frame, "End Prompt with .-.-.- (Fullstop)", (int(webcam.get(3)/2)-120, int(webcam.get(4))-20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 125, 125), 2)

        #frame = cartoon(frame)    
        cv2.imshow('Morse Head', frame)

        if cv2.waitKey(5) & 0xFF == 27:
            break


    webcam.release()
    cv2.destroyAllWindows()


    prompt_clean = prompt_inter.strip().replace(".", "")

    #print(prompt_clean)
    return prompt_clean


def head_eye_morse(cartoonize="False",webcam_port=0):

    left_id = [7, 33, 133, 144, 145, 153, 154, 155, 157, 158, 159, 160, 161, 163, 173, 246]
    left_upper_id = [27,28,29,30,247,130,25,110,24,23,22,26,112,243,190,56]
    right_id = [249, 263, 362, 373, 374, 380, 381, 382, 384, 385, 386, 387, 388, 390, 398, 466]
    prompt_final = []
    prompt_inter = ""
    prompt = []
    flag=0
    flag2=0
    flag3=0
    flag4=0
    cnt=0
    pts = deque(maxlen=512)
    webcam = cv2.VideoCapture(webcam_port)

    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5,max_num_faces=1,refine_landmarks=True)

    while webcam.isOpened():
        _, frame = webcam.read()

        frame.flags.writeable = False
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = face_mesh.process(frame)

        if cartoonize=="True":
            frame = cartoon(frame) 

        left_2d = []
        right_2d = []
        face_3d = []
        face_2d = []
        landmark_lst = [33, 263, 1,61, 291, 199]
        # Draw the face mesh annotations on the frame.
        frame.flags.writeable = True
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                left_2d = [[int(lm.x * frame.shape[1]),int(lm.y * frame.shape[0])] for idx, lm in enumerate(face_landmarks.landmark) if idx in sorted(left_id)]
                right_2d = [[int(lm.x * frame.shape[1]),int(lm.y * frame.shape[0]),lm.z] for idx, lm in enumerate(face_landmarks.landmark) if idx in sorted(right_id)]

                face_2d = [[int(lm.x * frame.shape[1]),int(lm.y * frame.shape[0])] for idx, lm in enumerate(face_landmarks.landmark) if idx in sorted(landmark_lst)]
                face_3d = [[int(lm.x * frame.shape[1]),int(lm.y * frame.shape[0]),lm.z] for idx, lm in enumerate(face_landmarks.landmark) if idx in sorted(landmark_lst)]

                leftEye = np.array(left_2d)
                rightEye = np.array(right_2d)

                face_2d = np.array(face_2d, dtype=np.float64)

                face_3d = np.array(face_3d, dtype=np.float64)

                leftEAR = eye_aspect_ratio(leftEye[11],leftEye[3],leftEye[9],leftEye[5],leftEye[1],leftEye[2])
                rightEAR = eye_aspect_ratio(rightEye[9],rightEye[5],rightEye[11],rightEye[3],rightEye[2],rightEye[1])
                ear = (leftEAR + rightEAR) / 2.0

                angles = cal_pnp(frame,face_2d,face_3d)

                vert = angles[0] * 360
                hor = angles[1] * 360

                if ear < 0.20: 

                    flag+=1
                    pts.appendleft(flag)

                else:
                    flag=0
                    flag2+=1
                    pts.appendleft(flag)
            
            
                #print("FLag",flag)
                #print("Flag2",flag2)


                for i in range(1, len(pts)):
                    
                    if pts[i] > pts[i - 1]:

                        if pts[i] > 8 and pts[i] < 30:
                            prompt_final.append("-")
                            pts = deque(maxlen=512)
                            break
                        elif pts[i] > 3 and pts[i] < 8:

                            prompt_final.append(".")
                            pts = deque(maxlen=512)
                            break

                if vert < -2:
                    text = "/"
                elif vert > 15:
                    text = "Pop previous"
                    try:
                        prompt_final.pop()
                    except:
                        print("Its Empty already")
                else:
                    text = "Center"


                if text=='/' and len(prompt_final)==0 and flag3>0:
                    flag4+=flag3

                    cv2.putText(frame, f'For space : {40-flag4}', (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 102, 0), 4)

                    if flag4>40:
                        cv2.putText(frame, "Space entered!", (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 102, 0), 4)

                        #print('Buffer for Space')
                        prompt_inter += " "
                        flag4=0
                        flag3=0

                if text=='/' and len(prompt_final)>0:

                    if morse_to_character.get(''.join(prompt_final))=='.':

                        webcam.release()
                        cv2.destroyAllWindows() 
                    
                    try:     
                        prompt_inter += morse_to_character.get(''.join(prompt_final))

                        prompt_final = []
                        flag4=0
                        flag3=1
                    except:
                        pass


                cv2.putText(frame, ''.join(prompt_final), (40, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

                cv2.putText(frame, prompt_inter, (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (128, 0, 128), 2)


        cv2.putText(frame, "Short Blink: .(Dit)", (20, int(webcam.get(4)/2)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        cv2.putText(frame, "Long Blink Pause: -(Dah)", (int(webcam.get(3))-220, int(webcam.get(4)/2)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        cv2.putText(frame, "Pop Previous Dit/Dah", (int(webcam.get(3)/2)-50, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        cv2.putText(frame, "Finish with letter", (int(webcam.get(3)/2)-70, int(webcam.get(4))-50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        cv2.putText(frame, "End Prompt with .-.-.- (Fullstop)", (int(webcam.get(3)/2)-120, int(webcam.get(4))-20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 125, 125), 2)


        cv2.imshow('Morse Eye + Head',frame)
        if cv2.waitKey(5) & 0xFF == 27:
            break
    webcam.release()
    cv2.destroyAllWindows()

    prompt_clean = prompt_inter.strip().replace(".", "")

    #print(prompt_clean)
    return prompt_clean




parser = argparse.ArgumentParser(description="Morse Diffusion")
parser.add_argument("-t","--type",help='Type Morse Code', action='store', type=str)
parser.add_argument("-head","--head_morse",help='Head Movement Morse Code (default: True)',type=str,default=True)
parser.add_argument("-eyehead","--eye_head_morse",help='Eye Blink + Head Movement Morse Code (default: False)',type=str,default=False)

parser.add_argument("-c","--cartoon",help='Cartoonize your frame (default: False)',type=str,default=False)
parser.add_argument("-b","--batch_size",help='Batch-Size of Diffusion function (default: 1)',type=int,default=1)
parser.add_argument("-img_h","--img_height",help='Image height required in your Diffusion function (default: 512)',type=int,default=512)
parser.add_argument("-img_w","--img_width",help='Image weight required in your Diffusion function (default: 512)',type=int,default=512)
parser.add_argument("-wc","--webcam_port",help='Webcam Port (default: 0)',type=int,default=0)

args = parser.parse_args()

if args.type:
    prompt_inter=""
    #print(args.type.split('/'))
    for word in args.type.split(' / '):
        for letter in word.split(' '):
            prompt_inter += morse_to_character.get(''.join(letter))
        prompt_inter+=" "
    prompt_clean = prompt_inter.strip()

    print(prompt_clean)

    keras_stable_diffusion(prompt_clean,batch_size=args.batch_size,img_height=args.img_height,img_width=args.img_width)

if args.eye_head_morse=="True":

    prompt_clean = head_eye_morse(cartoonize=args.cartoon,webcam_port = args.webcam_port)
    print(prompt_clean)
    #fine_prompt = input("Are you fine with your prompt? (y/n) :")

    #if fine_prompt=='y':
    #    keras_stable_diffusion(prompt_clean,batch_size=args.batch_size,img_height=args.img_height,img_width=args.img_width) 

    keras_stable_diffusion(prompt_clean,batch_size=args.batch_size,img_height=args.img_height,img_width=args.img_width) 


if args.head_morse=="True":

    prompt_clean = head_morse(cartoonize=args.cartoon,webcam_port = args.webcam_port)
    print(prompt_clean)
    #fine_prompt = input("Are you fine with your prompt? (y/n) :")

    #if fine_prompt=='y':
    #    keras_stable_diffusion(prompt_clean,batch_size=args.batch_size,img_height=args.img_height,img_width=args.img_width)
    keras_stable_diffusion(prompt_clean,batch_size=args.batch_size,img_height=args.img_height,img_width=args.img_width)
       
