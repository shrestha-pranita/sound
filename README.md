# SoundRecord
To re-create the database:
Go to backend > DjangoRestApi and migrate database
python manage.py makemigrations users
python manage.py makemigrations user_profile
python manage.py makemigrations exams
python manage.py makemigrations questions
python manage.py makemigrations recordings
python manage.py migrate --fake-initial

To run the backend:
python manage.py runserver

To install the frontend requirements:
npm install

To run the frontend:
npm start



Run http://localhost:8081/ in web browser

* To run switch in reactJS
npm install react-router-dom@5.2.0

To install pyannote.audio.core:
pip install https://github.com/pyannote/pyannote-audio/archive/develop.zip


