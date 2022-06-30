# American Sign-language to Audio :sunglasses:
This repo hosts the code for a system that can convert American sign language to audio. As the title suggests, this system aims to establish efficient communication between normal language and sign language.  

## Technologies used :sparkling_heart:
- OpenCV
- TensorFlow
- gTTS  
  
## About the system :notes:
- The core of the system is a machine learning model (find the notebook :star2: [here](www.google.com) :star2: ).
- An example dataset for the ML model can be found [here](https://www.kaggle.com/datasets/ardamavi/sign-language-digits-dataset). The images undergoes the following transformations before being used for training:  
:point_down::point_down::point_down:  
```python
  gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
  blur = cv.GaussianBlur(gray,(5,5),2)
  th3 = cv.adaptiveThreshold(blur, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, 11, 2)
  ret, res = cv.threshold(th3, 70, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)
  res = cv.resize(res, (128,128))
```  

- After training the model to the desired accuracy, you can save it in [Custom_models](https://github.com/Gokul-GMenon/Sign-language/tree/main/Custom_models) folder. Save it under the name: base_model.h5 :clap: :clap:
- After the initial training procedure, the model can further be trained by trasnfer learning on the base model on the custom dataset of a person using [detect_and_train.py](https://github.com/Gokul-GMenon/Sign-language/blob/main/self_train/detect_and_train.py). You can run the python file using the command:  
```
root@Gokuls-Laptop:~/Sign-language# python self_train/detect_and_train.py -n [NAME]
```  
where [NAME] is the name of the user whose database was just created. This python file will capture 100 images of each of the letters done the the person's hands :sweat_drops: :sweat_drops: . This database will be then used for transfer learning and the model will be saved with the persons name in [Custom_models](https://github.com/Gokul-GMenon/Sign-language/tree/main/Custom_models).
- After training for the profile, run the python file [hand_track.py](https://github.com/Gokul-GMenon/Sign-language/blob/main/hand_track.py) using the following command:  
```
root@Gokuls-Laptop:~/Sign-language# python hand_track.py -n [NAME]
```  
where [NAME] is the name of the person whose database already exists. To use the base model itself, the arguments can be skipped :broken_heart: and simply run:  
```
root@Gokuls-Laptop:~/Sign-language# python hand_track.py
```  

### Find the notebook for the base model [here](https://colab.research.google.com/drive/1JjcAAYZR43975aWQu8jv4tDkTuHbd3_F?usp=sharing). :revolving_hearts:

## Model Details
- Framework: TensorFlow
- Architecture: Convolutional Neural Network.
- Recommends training till an accuracy of 80% on the training data atleast (can be improved with detect_and_train.py)

<!-- ## Contributors -->
<br />

<table>
  <tr>

<td align="center"><a href="https://github.com/Gokul-GMenon"><img src="https://avatars.githubusercontent.com/u/76942680?s=400&u=610dfaac5f89ca089a69a62ccf9df283017bf60b&v=4" width="90px;" alt=""/><br /><sub><b>Gokul G Menon</b></sub></a><br />

</tr>
</table>
