import React, { Component } from "react";
import MicRecorder from 'mic-recorder-to-mp3';
import axios from "axios"
import  web_link from "../web_link";
import {Redirect, withRouter} from 'react-router-dom';
import Header from '../elements/header';

const Mp3Recorder = new MicRecorder({ bitRate: 128 });
class Audio extends Component {
  constructor(props) {
    super(props);  
    this.state = {
        isRecording: false,
        blobURL: '',
        isBlocked: false,
        isRecordingStp: false,
        audioFile : '',
        blobFile: '',
      }
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
    Mp3Recorder
      .stop()
      .getMp3()
      .then(([buffer, blob]) => {
        this.state.audioFile = new File([blob], "record.wav");
        const blobURL = URL.createObjectURL(blob)
        this.state.blobFile = blob;
        this.setState({ blobURL, isRecording: false });

        this.setState({ isRecordingStp: true });
        
      }).catch((e) => console.log(e));
  };

  submit() {
    Mp3Recorder
    .getMp3()
    .then(() => {
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

  reset() {
      document.getElementsByTagName('audio')[0].src = '';
      this.setState({ isRecordingStp: false });
  };

  render() {
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