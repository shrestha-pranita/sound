import MicRecorder from "mic-recorder-to-mp3"
import { useEffect, useState, useRef } from "react"
import axios from "axios"
import  web_link from "../web_link";
const SpeakerRecSubmit= () => {
  // Mic-Recorder-To-MP3
  const recorder = useRef(null) //Recorder
  const audioPlayer = useRef(null) //Ref for the HTML Audio Tag
  const [mediaBlobUrl, setBlobUrl] = useState(null)
  const [audioFile, setAudioFile] = useState(null)
  const [isRecording, setIsRecording] = useState(null)

  useEffect(() => {
    //Declares the recorder object and stores it inside of ref
    recorder.current = new MicRecorder({ bitRate: 128 })
  }, [])

  const sample_startRecording = () => {
    // Check if recording isn't blocked by browser
    recorder.current.start().then(() => {
      setIsRecording(true)
    })
  }

  const sample_stopRecording = () => {
    recorder.current
      .stop()
      .getMp3()
      .then(([buffer, blob]) => {
        const file = new File(buffer, "audio.mp3", {
          type: blob.type,
          lastModified: Date.now(),
        })
        const newBlobUrl = URL.createObjectURL(blob)
        setBlobUrl(newBlobUrl)
        setIsRecording(false)
        setAudioFile(file)
      })
      .catch((e) => console.log(e))
  }

  const handleSave = async () => {
    const audioBlob = await fetch(mediaBlobUrl).then((r) => r.blob());
    //const audioBlob = new Blob(audioChunks, { type: "audio/wav" })
    const audioUrl = URL.createObjectURL(audioBlob);
    const audio = new Audio(audioUrl);
    
    var data = new FormData()
    data.append('file', audioUrl)
    console.log(data)
    //const audioFile = new File([audioBlob], 'voice.wav', { type: 'audio/wav' });
    //const formData = new FormData(); // preparing to send to the server

    //formData.append('file', audioFile);  // preparing to send to the server
    //console.log(formData)
    fetch(web_link+'/api/speakerSample', {
        method: "POST",
        headers: {
            "Content-Type": "application/json",
            //'Access-Control-Allow-Origin': 'http://localhost:8000',
            //'Access-Control-Allow-Credentials': 'true'
        },
        
        body: JSON.stringify({
            data: data,
        }),
        
        })
        .then((response) => response.json())
        .then((result) => {
            console.log('Success:', result);
        })
        .catch((error) => {
            console.error('Error:', error);
        });
  };


  return (
    <div>
      <h1>React Speech Recognition App</h1>
      <audio ref={audioPlayer} src={mediaBlobUrl} controls='controls' />
      <div>
        <button disabled={isRecording} onClick={sample_startRecording}>
          START
        </button>
        <button disabled={!isRecording} onClick={sample_stopRecording}>
          STOP
        </button>
        <button onClick={handleSave}>SUBMIT</button>
      </div>
    </div>
  )
}

export default SpeakerRecSubmit