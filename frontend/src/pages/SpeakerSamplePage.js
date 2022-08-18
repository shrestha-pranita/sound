import React, { Component } from 'react';
import axios from 'axios';
import { ReactMediaRecorder } from "react-media-recorder";
//import Audio from "../components/SpeakerRec1View";
import Audio from "../components/SpeakerRecSubmit";
import Submit_sample from "../components/SpeakerRecSubmit";
import Sidebar from '../elements/sidebar';
import AudioReactRecorder, { RecordState } from 'audio-react-recorder'
//import { Recorder } from "react-voice-recorder";
//import { ReactAudioRecorder } from '@sarafhbk/react-audio-recorder'

export default class SpeakerSamplePage extends React.Component {
  render() {
     
    return (
      <div>

        <Audio/>
      </div>
    );
  }
}