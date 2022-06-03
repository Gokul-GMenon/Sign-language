from gtts import gTTS

import os

def toAudio(text):

    language = 'en'

    try:
        myobj = gTTS(text=text, lang=language, slow=False)

        # Saving the converted audio in a mp3 file named
        # welcome 
        myobj.save("transcript.mp3")

        from playsound import playsound

        # for playing note.wav file
        playsound('transcript.mp3')

        os.remove('transcript.mp3')

    except:

        return
