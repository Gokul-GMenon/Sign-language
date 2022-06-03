from operator import le
from cvzone.HandTrackingModule import HandDetector
import cv2 as cv, os, numpy as np
import argparse

ap = argparse.ArgumentParser()

ap.add_argument('-n', '--name', required=True, help = 'name of the new user')
args = ap.parse_args()
name = ''
last_closed = 0

if os.path.exists(os.path.join('Custom_models', args.name + '_Model.h5')):

    name = args.name
    print("\n\n"+ name+" is available. Please wait...\n")

else:

    print("\n\nNo trained model of the said person is available!!\nEnter an available username or please train using detect_and_train.py!!\n")
    print('Available models are listed below:')
    
    for item in os.listdir('Custom_models'):
        print(item)

    print()
    exit()

cap = cv.VideoCapture(0)

cap.set(3, 1280)
cap.set(4, 720)
detector = HandDetector(detectionCon = 0.5, maxHands = 2)

sz1, sz2 = 600, 425
signs = cv.resize(cv.imread(os.path.join('self_train', 'american_sign_lang.png')), (sz1, sz2))

letter = ''
letter_prv = ''

count_c = 0
count_w = 0
max_words = 7
word = ['']*max_words
wc=0
print(word)

while True:

    ret, frame = cap.read()
    frame = cv.resize(frame, (sz1, sz2))
    
    # Find hands
    lm = detector.findHands(frame, draw = False)

    x1 = y1 = x2 = y2 = ''

    crop = []

    if lm:

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

        letter = ''
        try:
            if int(x1) > 0 and int(x2) > 0 and int(y1) > 0 and int(y2) > 0:

                crop = frame[y1:y2, x1:x2]

                from model_predict import predict
                import numpy as np
            
                if crop != np.array([]):
                    letter = predict(crop, name)
        except:
            continue

        print('letter - ', letter)
        
        if letter != '':

            if letter_prv == '':
                
                letter_prv = letter
                count_w = 1

            else:
                
                if letter_prv == letter:
                
                    if count_w == 5:
                        
                        word[wc] = word[wc] + letter
                        last_closed = 1
                        count_w=0
                
                    else:
                        count_w+=1
                else:
                    letter_prv = ''
                    count_w = 0
        else:

            if count_c <= 5:
                count_c+=1
            else:
                count_w = 0
                count_c = 0
                word[wc] = ''

        cv.rectangle(frame, (x1, y1), (x2, y2), color=(0,255,0))
        frame = cv.flip(frame, 1)
        cv.rectangle(frame, (0, 385), (600, 425), (255,255,255), thickness = -1)

        text = ''
        for i in range (0, wc+1):
            text += word[i]
            text += ' '

        cv.putText(frame, text, (25, 413), fontFace=cv.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=(0, 0, 0), thickness=1)


    else:
        frame = cv.flip(frame, 1)

        from audio_player import toAudio
        if last_closed == 1:
            toAudio(word[wc])

        if word[wc] != '':
            wc += 1
        else:
            if wc == max_words:    
                wc = 0
                word = ['']*max_words


        last_closed = 0
        cv.rectangle(frame, (0, 385), (600, 425), (255,255,255), thickness = -1)
        
        text = ''
        for i in range (0, wc+1):
            text += word[i]
            text += ' '

        cv.putText(frame, text, (25, 413), fontFace=cv.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=(0, 0, 0), thickness=1)
        
    
    frame = np.concatenate((frame, signs), axis =1)
    cv.imshow("Image", frame)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()