# Sign-language
- link of colab file  : https://colab.research.google.com/drive/1JjcAAYZR43975aWQu8jv4tDkTuHbd3_F?usp=sharing


## Custom Training
For best accuracy, please train custom model using your hands by running the following command in terminal (after installing all the dependencies):
- python self_train\detect_and_train.py -n [NAME]
where, [NAME] is the name of the person whose hand is being registered.

After training has been completed, run hand_track.py using the following command:
- python hand_track.py -n [NAME]
where, [NAME] is the name of the persons whos trained model is available.
