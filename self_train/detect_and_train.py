import cv2 as cv
import os, numpy as np
from cvzone.HandTrackingModule import HandDetector
import argparse

ap = argparse.ArgumentParser()

ap.add_argument('-n', '--name', required=True, help = 'name of the new user')
args = ap.parse_args()
name = ''
curr_path = ''
root = ''

init = 0

if os.path.exists(os.path.join('training_images', args.name)):

    print("\n\nPerson's database already exists!!\n")
    exit()

else:
    name = args.name
    curr_path = os.path.join('training_images', name)
    os.makedirs(curr_path)
    root = curr_path


cap = cv.VideoCapture(0)

cap.set(3, 1280)
cap.set(4, 720)
detector = HandDetector(detectionCon = 0.5, maxHands = 2)

n_img = 0

i=0
# text = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
text = ['0', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', '-1']

sz1, sz2 = 600, 425

signs = cv.resize(cv.imread(os.path.join('self_train', 'american_sign_lang.png')), (sz1, sz2))

wait = 1
wait_c = 0

while True:

    ret, frame = cap.read()
    frame = cv.resize(frame, (sz1, sz2))

    if text[i] == '-1':
        break

    # Find hands
    lm = detector.findHands(frame, draw = False)
    # lm, bbox = detector.findPosition(img)

    x1 = y1 = x2 = y2 = ''

    crop = []

    # print((lm == 0 and wait == 0) or text[i] == '0')
    if (lm and wait == 0) or text[i] == '0':

        # print((lm == 0 and wait == 0) or text[i] == '0')
        # Saving the blank images
        if text[i] == '0':
            custom_name = 1
            for save in range(0, 100):
                if save < 71:
                    curr_path = os.path.join(root, 'train',  text[i])
                    if not os.path.isdir(curr_path):
                        os.makedirs(curr_path)

                    res = cv.imread(os.path.join('self_train', 'blank.jpg'))
                    res = cv.cvtColor(res, cv.COLOR_BGR2GRAY)
                    res = cv.resize(res, (128,128))
                    cv.imwrite(os.path.join(curr_path, str(custom_name) + '.png'), res)
                    custom_name+=1
                elif save > 70:

                    curr_path = os.path.join(root, 'test',  text[i])
                    if not os.path.isdir(curr_path):
                        os.makedirs(curr_path)

                    res = cv.imread(os.path.join('self_train', 'blank.jpg'))
                    res = cv.cvtColor(res, cv.COLOR_BGR2GRAY)
                    res = cv.resize(res, (128,128))
                    cv.imwrite(os.path.join(curr_path, str(custom_name) + '.png'), res)
                    custom_name+=1
            i+=1
            continue            


        if len(lm) == 1:
            x1, y1 = lm[0]['bbox'][0]-100, lm[0]['bbox'][1]-50
            x2, y2 = x1 + lm[0]['bbox'][2]+200, y1 + lm[0]['bbox'][3]+150
        elif len(lm) == 2:
            xl_1, yl_1 = lm[0]['bbox'][0]-100, lm[0]['bbox'][1]-50
            xl_2, yl_2 = xl_1 + lm[0]['bbox'][2]+200, yl_1 + lm[0]['bbox'][3]+150
        
            xr_1, yr_1 = lm[1]['bbox'][0]-100, lm[1]['bbox'][1]-50
            xr_2, yr_2 = xr_1 + lm[1]['bbox'][2]+200, yr_1 + lm[1]['bbox'][3]+150

            if xl_1 < xr_1:
                x1 = xl_1
                x2 = xr_2
            else:
                x1 = xr_1
                x2 = xl_2

            if yl_1 < yr_1:
                y1 = yl_1
                y2 = yr_2
            else:
                y1 = yr_1
                y2 = yl_2

        if int(x1) > 0 and int(x2) > 0 and int(y1) > 0 and int(y2) > 0:

            crop = frame[y1:y2, x1:x2]
            n_img+=1


            if n_img == 101:
                i+=1
                wait = 1
                n_img = 0
                continue

            image = cv.flip(crop, 1) 
            gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
            blur = cv.GaussianBlur(gray,(5,5),2)

            th3 = cv.adaptiveThreshold(blur, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, 11, 2)
            ret, res = cv.threshold(th3, 70, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)
            res = cv.resize(res, (128,128))

            if n_img < 71:

                curr_path = os.path.join(root, 'train',  text[i])
                if not os.path.isdir(curr_path):
                    os.makedirs(curr_path)


            else:

                curr_path = os.path.join(root, 'test',  text[i])
                if not os.path.isdir(curr_path):
                    os.makedirs(curr_path)

                
            cv.imwrite(os.path.join(curr_path, str(n_img) + '.png'), res)

        cv.rectangle(frame, (x1, y1), (x2, y2), color=(0,255,0))
        frame = cv.flip(frame, 1)

    else:
        frame = cv.flip(frame, 1)
        
        if wait == 1:
            wait_c +=1

            cv.putText(frame, 'Please show the hand sign for - '+text[i],  (100,100), fontFace=cv.FONT_HERSHEY_SIMPLEX, fontScale=0.75, color=(0,255,0), thickness=2)

            if wait_c >= 55:
                wait = 0
                wait_c = 0
                continue

    frame = np.concatenate((frame, signs), axis =1)
    cv.imshow("Image", frame)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv.destroyAllWindows()

print('\n\nTraining...\n')

from self_train.model_train import train

train(name)

print('\n\nTraining has been completed. The model has been saved as ' + name + '_Model.h5 in the folder Custom_models.\n')