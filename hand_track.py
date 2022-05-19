from operator import le
from cvzone.HandTrackingModule import HandDetector
import cv2 as cv

cap = cv.VideoCapture(0)

cap.set(3, 1280)
cap.set(4, 720)
detector = HandDetector(detectionCon = 0.5, maxHands = 2)

letter = ''
letter_prv = ''

count_c = 0
count_w = 0
word = ''

while True:

    ret, frame = cap.read()

    # Find hands
    lm = detector.findHands(frame, draw = False)
    # lm, bbox = detector.findPosition(img)

    x1 = y1 = x2 = y2 = ''

    crop = []

    if lm:

        if len(lm) == 1:
            x1, y1 = lm[0]['bbox'][0]-100, lm[0]['bbox'][1]-50
            x2, y2 = x1 + lm[0]['bbox'][2]+200, y1 + lm[0]['bbox'][3]+150
        elif len(lm) == 2:
            xl_1, yl_1 = lm[0]['bbox'][0]-100, lm[0]['bbox'][1]-50
            print(lm)
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

            print('\n\n', x1, y1, x2, y2, '\n')        
        letter = ''

        if x1 > 0 and x2 > 0 and y1 > 0 and y2 > 0:

            crop = frame[y1:y2, x1:x2]
            
            from model_predict import predict
            import numpy as np
        
            if crop != np.array([]):
                letter = predict(crop)
        
        print('\nletter - ', letter)
        
        if letter != '':

            if letter_prv == '':
                
                letter_prv = letter
                count_w = 1

            else:
                
                if letter_prv == letter:
                
                    if count_w == 10:
                        
                        word = word + letter
                        count_w=0
                        # print('word - ', word)

                    else:
                        # print('Count val - ', count_w)
                        count_w+=1
                else:
                    letter_prv = ''
                    count_w = 0
        else:

            if count_c <= 10:
                count_c+=1
            else:
                # print('Hi')
                count_w = 0
                count_c = 0
                word = ''
        cv.rectangle(frame, (x1, y1), (x2, y2), color=(0,0,255))
        frame = cv.flip(frame, 1)
        if word != '':
            cv.putText(frame, word, (100,100), fontFace=cv.FONT_HERSHEY_SIMPLEX, fontScale=2, color=(0,255,0), thickness=3)
        else:
            cv.putText(frame, ' ', (100,100), fontFace=cv.FONT_HERSHEY_SIMPLEX, fontScale=2, color=(0,255,0), thickness=3)
        
    else:
        frame = cv.flip(frame, 1)
        word = ''
    # else:
    #     cv.putText(frame, '. .', (100,100), fontFace=cv.FONT_HERSHEY_SIMPLEX, fontScale=2, color=(0,255,0), thickness=3)
        

    # Display
    # if crop != []:
    #     cv.imshow("Image", crop)
    # else:
    cv.imshow("Image", frame)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv.destroyAllWindows()