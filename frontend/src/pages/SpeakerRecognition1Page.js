import React, { Component } from 'react';
import axios from 'axios';
import { ReactMediaRecorder } from "react-media-recorder";
import Audio from "../components/SpeakerRec1View";
import Sidebar from '../elements/sidebar';

export default class RecordPage extends React.Component {
  render() {
     
    return (
        
      <Audio/>
    );
  }
}
