import React, { Component } from "react";
import MicRecorder from 'mic-recorder-to-mp3';
import axios from "axios"
import  web_link from "../web_link";
import {Redirect, useHistory, withRouter} from 'react-router-dom';
import Header from '../elements/header';

const Mp3Recorder = new MicRecorder({ bitRate: 128 });

class Audio extends Component {
  constructor(props) {
    super(props);

    /*
     * declare states that will enable and disable
     * buttons that controls the audio widget
     */
    this.state = {
        isRecording: false,
        blobURL: '',
        isBlocked: false,
        isRecordingStp: false,
        audioFile : '',
        isRecording: '',
        blobFile: '',
      }
    //const [blobURL, setBlobUrl] = useState(null)
    //const [audioFile, setAudioFile] = useState(null)
    //const [isRecording, setIsRecording] = useState(null)

    //binds the methods to the component
    this.start = this.start.bind(this);
    this.stop = this.stop.bind(this);
    this.reset = this.reset.bind(this);
    this.submit = this.submit.bind(this);
   }

  componentDidMount(){
    //Prompt the user for permission to allow audio device in browser
    navigator.getUserMedia = (
      navigator.getUserMedia ||
      navigator.webkitGetUserMedia ||
      navigator.mozGetUserMedia ||
      navigator.msGetUserMedia
     );

     //Detects the action on user click to allow or deny permission of audio device
     navigator.getUserMedia({ audio: true },
      () => {
        console.log('Permission Granted');
        this.setState({ isBlocked: false });
      },
      () => {
        console.log('Permission Denied');
        this.setState({ isBlocked: true })
      },
    );
  }
 
  start(){
    /*
     * If the user denys permission to use the audio device
     * in the browser no recording can be done and an alert is shown
     * If the user allows permission the recoding will begin
     */
    if (this.state.isBlocked) {
      alert('Permission Denied');
    } else {
      Mp3Recorder
        .start()
        .then(() => {
          this.setState({ isRecording: true });
        }).catch((e) => console.error(e));
    }
  }

  stop() {
     /*
     * Once the recoding starts the stop button is activated
     * Click stop once recording as finished
     * An MP3 is generated for the user to download the audio
     */
    Mp3Recorder
      .stop()
      .getMp3()
      .then(([buffer, blob]) => {
        this.state.audioFile = new File([blob], "record.wav");
        const blobURL = URL.createObjectURL(blob)
        this.state.blobFile = blob;
        //const blobURL = URL.createObjectURL(blob)
        //var wavfromblob = new File([blob], "record.wav");
        /*
        this.state.audioFile = new File([blob], "record.wav");
        const formData = new FormData()
        formData.append("file", this.state.audioFile)
        axios.post(`${web_link}/api/speakerSample`, formData, {
          headers: {
            'Content-Type': `multipart/form-data`,
        },

        })
        .then(result => {
          console.log(result)
        })
        */
        this.setState({ blobURL, isRecording: false });

        this.setState({ isRecordingStp: true });
        
      }).catch((e) => console.log(e));
  };

  submit() {
    /*
    * Once the recoding starts the stop button is activated
    * Click stop once recording as finished
    * An MP3 is generated for the user to download the audio
    */
    Mp3Recorder
    .getMp3()
    .then(() => {
       //const blobURL = URL.createObjectURL(blob)

       var wavfromblob = new File([this.state.blobFile], "record.wav");

       const formData = new FormData()
       formData.append("file", wavfromblob)

       axios.post(`${web_link}/api/speakerSample`, formData, {
         headers: {
           'Content-Type': `multipart/form-data`,
       },

       })
       .then(result => {
        this.props.history.push("/speakerrec1");
          return <Redirect to='/speakerrec1' />;
          console.log(result)
       })
      }).catch((e) => console.log(e));
    };

  /*
  stop = () => {
    Mp3Recorder
      .stop()
      .getMp3()
      .then(([buffer, blob]) => {
        const file = new File([blob], "record.wav", {
          type: blob.type,
          lastModified: Date.now(),
        })
        const newBlobUrl = URL.createObjectURL(blob)
        setBlobUrl(newBlobUrl)
        setIsRecording(false)
        setAudioFile(file)
      })
      .catch((e) => console.log(e))
  };
  */

  reset() {
      /*
       * The user can reset the audio recording
       * once the stop button is clicked
       */
      document.getElementsByTagName('audio')[0].src = '';
      this.setState({ isRecordingStp: false });
  };

  render() {

    //display view of audio widget and control buttons
    return(
      <div>
        <div id = "wrapper">
          <Header></Header> 
        </div>
        <div className="row d-flex justify-content-center mt-5">
          <button className="btn btn-light" onClick={this.start} disabled={this.state.isRecording}>Record</button>
          <button className="btn btn-danger" onClick={this.stop} disabled={!this.state.isRecording}>Stop</button>
          <button className="btn btn-warning" onClick={this.reset} disabled={!this.state.isRecordingStp}>Reset</button>
          <audio src={this.state.blobURL} controls="controls" />
          <button className="btn btn-light" onClick={this.submit} disabled={!this.state.isRecordingStp}>Submit</button>
        </div>
      </div>
    );
  }
}

export default withRouter(Audio);