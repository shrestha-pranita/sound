import { useReactMediaRecorder } from "react-media-recorder";
import React, { useEffect, useState } from "react";
import  web_link from "../web_link";
import Header from '../elements/header';
import {createModel, KaldiRecognizer, Model } from "vosk-browser";
import { LogBox } from 'react-native';

LogBox.ignoreLogs(['Warning: ...']);
LogBox.ignoreAllLogs();

let samples = [];
let context, source, processor;

const RecordView = (props) => {
  const [second, setSecond] = useState("00");
  const [minute, setMinute] = useState("00");
  const [isActive, setIsActive] = useState(false);
  const [counter, setCounter] = useState(0);
  const [result, setResult] = useState({});
  const [loadedModel, setLoadedModel] = useState({});
  const [loading, setLoading] = useState(false);
  const [recognizer, setRecognizer] = useState();

  useEffect(() => {
    const loadModel = async () => {
      setLoading(true);
      //loadedModel?.model.terminate();
  
      const model = await createModel('models/vosk-model-small-en-us-0.15.tar.gz');
      setLoadedModel({ model });
      const recognizer = new model.KaldiRecognizer(16000);
      console.log(recognizer.acceptWaveform)

      setRecognizer(() => {
        setLoading(false);
        return recognizer;
      });

      /*
      
      recognizer.setWords(true);
      recognizer.on("result", (message: any) => {
        const result: VoskResult = message.result;
        setUtterances((utt: VoskResult[]) => [...utt, result]);
      });
  
      recognizer.on("partialresult", (message: any) => {
        setPartial(message.result.partial);
      });
  
      setRecognizer(() => {
        setLoading(false);
        setReady(true);
        return recognizer;
      });
      */
    };
    /*
    const model = createModel('models/vosk-model-small-en-us-0.15.tar.gz');
    console.log(model)
    const recognizer = new model.KaldiRecognizer();
    recognizer.on("result", (message) => {
        console.log(`Result: ${message.result.text}`);
    });
    recognizer.on("partialresult", (message) => {
        console.log(`Partial result: ${message.result.partial}`);
    });
    */
    let intervalId;
    loadModel();
    if (isActive) {
      intervalId = setInterval(() => {
        const secondCounter = counter % 60;
        const minuteCounter = Math.floor(counter / 60);
        let computedSecond =
          String(secondCounter).length === 1
            ? `0${secondCounter}`
            : secondCounter;
        let computedMinute =
          String(minuteCounter).length === 1
            ? `0${minuteCounter}`
            : minuteCounter;
        setSecond(computedSecond);
        setMinute(computedMinute);

        setCounter((counter) => counter + 1);
      }, 650);
    }
    try {
        navigator.mediaDevices
          .getUserMedia({
            audio: { sampleRate: 48000, sampleSize: 16, channelCount: 1 },
          })
          .then((stream) => {
            //localMic = stream;
            context = new AudioContext();
            source = context.createMediaStreamSource(stream);
          })
          .catch((e) => console.log("Mic not Accessible!"));
      } catch (e) {
        console.error("start Mic error", e);
      }

    return () => clearInterval(intervalId);
  }, [isActive, counter]);

  const ButtonStart = () => {
    return <button
    style={{
      padding: "0.8rem 2rem",
      border: "none",
      marginLeft: "15px",
      fontSize: "1rem",
      cursor: "pointer",
      borderRadius: "5px",
      fontWeight: "bold",
      backgroundColor: "#42b72a",
      color: "white",
      transition: "all 300ms ease-in-out",
      transform: "translateY(0)"
    }}
    onClick={() => {
      if (!isActive) {
        startRecording();
        predictSwitch();
      } else {
        pauseRecording();
      }

      setIsActive(!isActive);
    }}
  >
    {isActive ? "Pause" : "Start"}
  </button>;
  };

  const ButtonStop = () => {
    return   <button
    style={{
      padding: "0.8rem 2rem",
      border: "none",
      backgroundColor: "#df3636",
      marginLeft: "15px",
      fontSize: "1rem",
      cursor: "pointer",
      color: "white",
      borderRadius: "5px",
      fontWeight: "bold",
      transition: "all 300ms ease-in-out",
      transform: "translateY(0)"
    }}
    onClick={() => {
      pauseRecording();
      //stopRecording();
      onStopRec();
      setIsActive(!isActive);
    }}
  >
    Stop
  </button>;
  };

  function stopTimer() {
    setIsActive(false);
    setCounter(0);
    setSecond("00");
    setMinute("00");
  }

  const {
    status,
    startRecording,
    stopRecording,
    pauseRecording,
    mediaBlobUrl
  } = useReactMediaRecorder({
    video: false,
    audio: true,
    echoCancellation: true
  });

  const predictSwitch = () => {
    processor = context.createScriptProcessor(16384, 1, 1);
    source.connect(processor);
    processor.connect(context.destination);
    processor.onaudioprocess = (e) => {
    samples = [...samples, ...e.inputBuffer.getChannelData(0)];
    console.log(e.inputBuffer.length)
    console.log(recognizer.Result())
    console.log(recognizer.acceptWaveform(samples));
    if (samples.length > 48000) {
        let out = [];
        for (let i = 0; i < 48000; i += 3) {
          let val = Math.floor(32767 * samples[i]);
          val = Math.min(32767, val);
          val = Math.max(-32768, val);
          out.push(val);
        }
        samples = samples.slice(48000);

        let userData = window.localStorage.getItem('user');
        if(userData){
            userData = JSON.parse(userData);
        }

        let user_id = userData.id
        fetch(web_link+'/api/speechVAD', {
        method: "POST",
        headers: {
            "Content-Type": "application/json",
        },
        body: JSON.stringify({
            data: out,
            user_id: user_id
        }),
        })
        .then((res) => res.json())
        .then((res) => setResult(res))
        .catch((err) => console.log(err))
    }
    };
  };
  const onStopRec = () => {
    stopRecording();
    processor.onaudioprocess = null;
    processor = null;
    };
  return (
    <>
    <div id = "wrapper">
      <Header></Header> 
    </div>
    <h1>
        Cheating Level Alert :{" "}

        {result.cheating_level === "high" ? (
          <span style={{ color: "Red" }}>High</span>
        ) : (
          <span style={{ color: "Green" }}>Low</span>
        )}{" "}
        Speech Detection :{" "}
        {result.speech_detection === "yes" ? (
          <span style={{ color: "Red" }}>Yes</span>
        ) : (
          <span style={{ color: "Green" }}>No</span>
        )}{" "}
      </h1>
      <h2>VAD 2</h2>
      <h2>Good Model</h2>
    <div
      style={{
        border: "1px solid black",
        backgroundColor: "black",
        width: "100%",
        height: "700px"
      }}
    >
      <div
        style={{
          border: "1px solid #bd9f61",
          height: "70px",
          backgroundColor: "#bd9f61",
          display: "flex"
        }}
      >
        <h4
          style={{
            marginLeft: "10px",
            textTransform: "capitalize",
            fontFamily: "sans-serif",
            fontSize: "18px",
            color: "white"
          }}
        >
          {status}
        </h4>
      </div>
      <div style={{ height: "38px",marginTop: "200px", marginLeft: "150px"}}>
        {" "}
        <video src={mediaBlobUrl} controls loop />
      </div>

      <div
        className="col-md-6 col-md-offset-3"
        style={{
          backgroundColor: "black",
          color: "white",
          marginLeft: "500px",
          marginTop: "-100px"
        }}
      >
        <button
          style={{
            backgroundColor: "black",
            borderRadius: "8px",
            color: "white"
          }}
          onClick={stopTimer}
        >
          Clear
        </button>
        <div style={{ marginLeft: "70px", fontSize: "54px" }}>
          <span className="minute">{minute}</span>
          <span>:</span>
          <span className="second">{second}</span>
        </div>

        <div style={{ marginLeft: "20px", display: "flex" }}>
          <label
            style={{
              fontSize: "15px",
              fontWeight: "Normal"
              // marginTop: "20px"
            }}
            htmlFor="icon-button-file"
          >
            <h3 style={{ marginLeft: "15px", fontWeight: "normal" }}>
              Press the Start to record
            </h3>
            <div>
              <ButtonStart/>
              <ButtonStop/>
              
            </div>
          </label>
        </div>
        <b></b>
      </div>
    </div>
    </>
  );
};
export default RecordView;
