import { useReactMediaRecorder } from "react-media-recorder";
import React, { useEffect, useState } from "react";
import  web_link from "../web_link";
import Header from '../elements/header';
import {Redirect,  useHistory } from 'react-router-dom';
import axios from "axios";

let samples = [];
let context, source, processor;

const RecordView = (props) => {
  const [second, setSecond] = useState("00");
  const [minute, setMinute] = useState("00");
  const [isActive, setIsActive] = useState(false);
  const [counter, setCounter] = useState(0);
  const [result, setResult] = useState({});
  const history = useHistory();
  const [stopIsDisabled, stopSetDisabled] = useState(true);
  const [submitIsDisabled, submitSetDisabled] = useState(true);
  const [status_val, setStatusVal] = useState("success");
  const [isRecording, isSetRecording] = useState(false);
  const [isBlocked, isSetBlocked] = useState(false);
  const [blobUrl, isSetBlobUrl] = useState('');
  const [exams, setExams] = useState({});
  const [examName, setExamName] = useState("");



  useEffect(() => {
    let intervalId;

    if(window.localStorage.getItem('isLoggedIn')){
      let userData = window.localStorage.getItem('user');
      if(userData){
          userData = JSON.parse(userData);
      } else {
        history.push("/login");
      }
    }else {
      history.push("/login");
    }

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

    let userData = window.localStorage.getItem('user');
    if(userData){
        userData = JSON.parse(userData);
    }

    let user_id = userData.id
    const pathname = window.location.pathname
    const slug = pathname.split("/").pop();

    /*
    fetch(web_link+'/api/startexam/'+slug, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
            user_id: user_id,
            exam_id: slug
        }),
        })
        .then((res) => res.json())
        .then((res) => {
          setExamName(res[0].exam_name)
          setExams(res)
          setStatusVal("success")
        })
        .catch((err) => {
          setStatusVal("fail")
        })
        */
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
        //start_audio_recording();
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
      //stop_audio_recording();
      setIsActive(!isActive);
    }}
    disabled = {stopIsDisabled}
  >
    Stop
  </button>;
  };

  const ButtonSubmit = () => {
    return   <button
    style={{
      padding: "0.8rem 2rem",
      border: "none",
      backgroundColor: "#0000FF",
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
      onFinalSubmitHandler();
    }}

    disabled = {submitIsDisabled}
  >
    Submit
  </button>;
  };

  function stopTimer() {
    setIsActive(false);
    setCounter(0);
    setSecond("00");
    setMinute("00");
  }

  /*
  function start_audio_recording() {
    Mp3Recorder
      .start()
      .then(() => {
        isSetRecording(true)
      }).catch((e) => console.error(e));
  }

  function stop_audio_recording() {
    Mp3Recorder
    .stop()
    .getMp3()
    .then(([buffer, blob]) => {
      const blobURL = URL.createObjectURL(blob)
      isSetRecording(false)
      isSetBlobUrl(blobUrl)
    }).catch((e) => console.log(e));
  }
  */
  const {
    status,
    startRecording,
    stopRecording,
    pauseRecording,
    mediaBlobUrl,
  } = useReactMediaRecorder({
    video: false,
    audio: true,
    echoCancellation: true,
    //mediaRecorderOptions: { mimeType: 'audio/wav' }
  });

  const predictSwitch = () => {
    stopSetDisabled(false);
    processor = context.createScriptProcessor(16384, 1, 1);
    source.connect(processor);
    processor.connect(context.destination);
    processor.onaudioprocess = (e) => {
    samples = [...samples, ...e.inputBuffer.getChannelData(0)];
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
        fetch(web_link+'/api/sileroVAD', {
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
    submitSetDisabled(false);
  };

  const onFinalSubmitHandler = () => {
    if (mediaBlobUrl == null) return;
    let userData = window.localStorage.getItem('user');
    if(userData){
        userData = JSON.parse(userData);
    }

    let user_id = userData.id;
    const pathname = window.location.pathname
    const slug = pathname.split("/").pop();
    
    fetch(mediaBlobUrl)
      .then((res) => res.blob())
      .then((res) => {
        let data = new FormData();
        const recordedFile = new File([res], 'voice');
        data.append("file", res);
        data.append("user_id", user_id);
        data.append("exam_id", slug);
        let config = {
          header: {
            "Content-Type": "multipart/form-data",
          },
        };
        axios
          .post(web_link + "/api/uploads", data, config)
          .then((response) => {
            console.log("whatttttt")
            history.push("/exam");
          })
          .catch((error) => {
            console.log("error", error);
          });
        
      });
    
  };

  return (
    <>
    <div id = "wrapper">
      <Header></Header> 
    </div>
    <h3>
      
    </h3>
    <h2>
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
      </h2>
    <h3>VAD 1</h3>
    <h3>Best Model</h3>
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
        <audio src={mediaBlobUrl} controls loop />
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
              <ButtonSubmit/>
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
