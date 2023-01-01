# Morse-Diffusion

This project aims to help people with various hand-tremor disabilties as they wouldn't be able to utilize an entire keyboard, when instead one of the adaptive tools used is Morse Code as it's a single-switch alternative. 
In that respect, Morse Diffusion uses Morse Code to generate Art via the Keras Stable Diffusion API.


### This is done in 3 ways-

#### 1) Typing Morse Code

Here, the user can type the prompt with Morse Code and the script generates the corresponding images via the Diffusion API. For people with spinal muscle atrophy, they can't use the below methods well, so this would be better alternative.

#### 2) Head Movements Morse Code

Generally, even the above way could prove to be cumbersome to many who can't use their hand-motor functions well, in that effect, they could make use of the movements of their head to convey the same morse code prompt, which will then generate images similarly.

##### Instructions:
```
* Moving your head- 

  * Left enters .(Dit)

  * Right enters .(Dah)

  * Up remove the previous Dit/Dah

  * Down (Short) enters the respective letter after completing the Morse Sequence for each letter

  * Down (Long) enters Space ('\') between words (There will be a countdown from 40 till 'Space Entered!')

  * Make sure to do .-.-.- (Dot) after you completed the prompt, which will close the webcam window and trigger the prompt to the Keras Stable Diffusion API.
  
  * Click ESC to exit at any time if not satisfied with the prompt, it will exit the application without triggering the API
```

#### 3) Eye Blinking + Head Movements Morse Code

Often, head movements can get tiring for longer sequences, which is why here, the user can use head movements combined with eye blinking to generate the morse code prompt

##### Instructions:

```
* Short Pause Blink enters .(Dit)

* Longer Pause Blink enters .(Dah)

* Head Up remove the previous Dit/Dah

* Head Down (Short) enters the respective letter after completing the Morse Sequence for each letter

* Head Down (Long) enters Space ('\') between words (There will be a countdown from 40 till 'Space Entered!')

* Make sure to do .-.-.- (Dot) after you completed the prompt, which will close the webcam window and trigger the prompt to the Keras Stable Diffusion API.

* Click ESC to exit at any time if not satisfied with the prompt, it will exit the application without triggering the API
```

#### *0) Yes, Voice (But someone had already implemented a good Whisper in Hugging Face, so didn't want to do the same)*


## Examples!

(I apologize, the Keras Stable Diffusion time for generating 1 image is 5-7 minutes as I am working on my personal laptop which has only CPU, if you device has a GPU, it will take seconds to generate it)
(I couldn't use Colab GPU as the Webcam + Mesh was causing the frame-rate to drop/lag, will work on that later on)


### Head Movements Morse Code

#### a) dog eating spaghetti


https://user-images.githubusercontent.com/34942185/210149941-0a1481cb-0e3c-490c-8625-84e26d845298.mp4


#### b) cat fighting dinosaur


https://user-images.githubusercontent.com/34942185/210149958-1e893324-2664-44d2-9678-1d244e0c3dfc.mp4

### Eye Blinking + Head Movements Morse Code

#### a) dolphin fighting an owl (This is funny because it seems that either an owl dressed by like a dolphin (wink) or that perhaps it needed more information about an owl)


https://user-images.githubusercontent.com/34942185/210150310-fdde45b5-3889-45cf-a854-b9cc6fbcf56b.mp4



## Running the Scripts!


### Prerequiste Libraries

```
 pip install mediapipe opencv-python
 pip install --upgrade keras-cv

```


### Script Help

![script help](https://user-images.githubusercontent.com/34942185/210150121-7042609d-336a-4b35-aaee-fe8512da7101.png)


#### 1) Typing Morse Code

```
 py main.py -t "-.. --- --. / . .- - .. -. --. / ... .--. .- --. .... . - - .." --head_morse False
```

#### 2) Head Movements Morse Code

```
 py main.py --head_morse True
```

#### 3) Eye Blinking + Head Movements Morse Code

```
py main.py -eyehead True --head_morse False
```

### TO DO
-  [ ]  Can pop out previous dit/dah of current letter but not entire previous letter (as frame-rate drops), so need to fix that
-  [ ]  Automatic text-resizing on screen for longer message than frame width
-  [ ]  Head Movement sensitivity control w.r.t frame-rate
-  [ ]  Streamlit? this into a GUI with parameters to control on-fly
-  [ ]  Colab integration
    
    
    

