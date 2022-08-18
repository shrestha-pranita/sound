import React, { Component } from 'react';
import axios from 'axios';
import { ReactMediaRecorder } from "react-media-recorder";
import Audio from "../components/NoiseDetection1View";
import Sidebar from '../elements/sidebar';

export default class NoiseDetection1Page extends React.Component {
  render() {
     
    return (
      <Audio/>
    );
  }
}
