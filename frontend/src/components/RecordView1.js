import { useReactMediaRecorder } from "react-media-recorder";
import React, { useEffect, useState } from "react";
import { useSnackbar } from "react-simple-snackbar";
import { confirmAlert } from 'react-confirm-alert';
import 'react-confirm-alert/src/react-confirm-alert.css';

let samples = [];
let localMic, context, source, processor;
const PREDICTAPI = process.env.NEXT_PUBLIC_BACKEND_BASE_URL;

const RecordView = (props) => {
  const [second, setSecond] = useState("00");
  const [minute, setMinute] = useState("00");
  const [isActive, setIsActive] = useState(false);
  const [counter, setCounter] = useState(0);
  const [bgcolor, setBgColor] = useState("green");
  const [speakingColor, setSpeakingColor] = useState("green");
  const [noiseSwitchColor, setNoiseSwitchColor] = useState("green");

  const options = {
    position: "top-center",
    style: {
      backgroundColor: "grey",
      border: "2px solid lightgreen",
      fontSize: "20px",
      textAlign: "center",
    },
    closeStyle: {
      color: "lightcoral",
      fontSize: "16px",
    },
  };

  //const [openSnackbar, _] = useSnackbar(options);

  /*useEffect(() => {
    let intervalId;
    

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
    return () => clearInterval(intervalId);
  }, [isActive, counter]);
  */

  useEffect(() => {
    try {
      navigator.mediaDevices
        .getUserMedia({
          audio: { sampleRate: 48000, sampleSize: 16, channelCount: 1 },
        })
        .then((stream) => {
          localMic = stream;
          context = new AudioContext();
          source = context.createMediaStreamSource(stream);
        })
        .catch((e) => console.log("Mic not Accessible!"));
    } catch (e) {
      console.error("start Mic error", e);
    }
  }, []);


  function stopTimer() {
    setIsActive(false);
    setCounter(0);
    setSecond("00");
    setMinute("00");
  }

  function testclick() {
    console.log("here")
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
  console.log("deed", mediaBlobUrl);

  const predictSwitch = () => {
    if (bgcolor === "green") {
      setBgColor("darkred");
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

          fetch(PREDICTAPI, {
            method: "POST",
            headers: {
              "Content-Type": "application/json",
            },
            body: JSON.stringify({
              data: out,
            }),
          })
            .then((res) => console.log("what"))
            .then((res) => {
              console.log(res)
            })
            .catch((err) => console.log(err));
        }
      };
    } else {
      setBgColor("green");
      setSpeakingColor("green");
      processor.onaudioprocess = null;
      processor = null;
    }
  };

  const convertFileToBase64 = (file) =>
    new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.readAsDataURL(file.mediaBlobUrl);

      reader.onload = () =>
        resolve({
          fileName: file.title,
          base64: reader.result
        });
      reader.onerror = reject;
    });

  return (
    <div
      style={{
        border: "1px solid black",
        backgroundColor: "black",
        width: "700px",
        height: "350px"
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
      <div style={{ height: "38px" }}>
        {" "}
        <video src={mediaBlobUrl} controls loop />
      </div>

      <div
        className="col-md-6 col-md-offset-3"
        style={{
          backgroundColor: "black",
          color: "white",
          marginLeft: "357px"
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
              <button
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
                    predictSwitch();
                  } else {
                    pauseRecording();
                  }

                  setIsActive(!isActive);
                }}
              >
                {isActive ? "Pause" : "Start"}
              </button>
              <button
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
                  stopRecording();
                  setIsActive(!isActive);
                }}
              >
                Stop
              </button>

              <button onClick={testclick()}></button>
            </div>
          </label>
        </div>
        <b></b>
      </div>
    </div>
  );
};
export default RecordView;
