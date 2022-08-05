import React, { Component } from 'react';
import axios from 'axios';
import { ReactMediaRecorder } from "react-media-recorder";
import Audio from "../components/MulSpeaker1View";
import Sidebar from '../elements/sidebar';

export default class MulSpeaker1Page extends React.Component {
  render() {
     
    return (
      <Audio/>
    );
  }
}
