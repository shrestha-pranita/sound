# Audio based Online Exam Cheating Detection System


## Cloning the project

Firstly, clone the repo to your machine by running following command:

```bash
git clone https://github.com/devibhattaraii/RCT-Dev-Deployment.git
```

## Creating virtual Environment

Enter to the root folder and create virtual environment as follows

```bash
virtualenv -p=/usr/bin/python3.9 env
source env/bin/activate
```
## Installation of packages

Python version 3.9 strictly needed for this to run.

```bash
sudo dnf install python3.9
```

After creating the virtual environement, install project requirements as follows: 

```bash
pip install -r requirements.txt
```  
If pyaudio throws error, run the following command and repeat the installation of packages.

```bash
sudo dnf install portaudio-devel
pip install -r requirements.txt
```
If torch needed for cuda, uninstall torch, torchvision and torchaudio to install the cuda version:
```bash
pip install torch==1.8.1+cu111 torchvision==0.9.1+cu111 torchaudio==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html
```

## If results does not exist

If the results folder is not available, 
go to the drive link for output at the bottom of readme and download version 46 to name it as "result" 

## Running the Project

In order start the project, we will need two servers:

### starting the frontend Server
In one terminal, start the frontend sever using following command:
```bash
# starts the frontend server
streamlit run streamlit.py

```

### starting the backend FastAPI server
In another terminal, start the Backend sever using following command:

```bash
### starts the backend FastAPI server
uvicorn app:app --reload
```

## Working Demo in Fedora 35(workstation)

To visit the streamlit UI, visit [http://localhost:8501](http://localhost:8501)

This UI screenshot is taken in Fedora 35 (workstation). The working demo with the running Frontend server while fetching the model's output via backend is demonstrated.


### UI on Initial State
When the Streamlit UI is visited, The following State is observed.
![Initial UI](https://github.com/devibhattaraii/RCT-Dev-Deployment/blob/b44ed2f0b8718913c5f4f5650503d92ce109acdc/Initial.png)

### UI When Sound is not detected
When you press the start button shown in initial state, the audio is then trasmitted for evaluation. 
Thus, here sound event is not detected so, The warning box shows green saying: "Great Going, Good Luck for the exam"

![Sound Not Detected](https://github.com/devibhattaraii/RCT-Dev-Deployment/blob/b44ed2f0b8718913c5f4f5650503d92ce109acdc/NoSound.png)

### UI When Sound is detected
When you continue to the exam, in between if the human voice is detected,
the warning box goes red saying, "You are requested to not speak during the exam".
Now if the Examinee stops speaking, the warning box goes green with no voice detected.
![Sound Detected](https://github.com/devibhattaraii/RCT-Dev-Deployment/blob/b44ed2f0b8718913c5f4f5650503d92ce109acdc/Sound.png)


When the audio feeding is stopped, the audio file is saved as output.wav which can be played from the media button below "Download as CSV".

## Models obtained

### 10 classes trained model: 
RCT was trained first using all 10 classes provided in DESED task 4. This is the trained model for 10 classes as target classes. The drive link for model is: https://drive.google.com/drive/folders/1kcWliV8DhB8Ac5ka7cJawMk8OB7wOozi?usp=sharing

### Speech only trained model: 
RCT was then trained using speech only as the target class. This is the trained model for speech only as target classes. The drive link for model is: https://drive.google.com/drive/folders/12OLrbajxROJxqbHbMGUeNDkfwtiZNH2l?usp=sharing 


## License
[Devi@ NSD-AI](https://choosealicense.com/licenses/mit/)
