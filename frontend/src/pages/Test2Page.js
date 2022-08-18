/*
import React from 'react'
import {connect} from 'react-redux'
import {Button} from 'semantic-ui-react'
import {injectIntl} from 'react-intl'
//import Recorder from '../components/Recorder'

class SpeechToTextAudioWorklet extends React.Component {

  componentDidMount() {
    this.context = null
    this.recorder = null
  }

  startRecorder() {
    if (!this.context) {
      this.context = new window.AudioContext()
      this.recorder = new Recorder(this.context)
    } else {
      this.context.resume()
    }

    this.recorder && this.recorder.record()
  }

  stopRecorder() {
    this.recorder && this.recorder.stop()
  }

  render() {

    return (<div>Speech to text

      <Button id="start" onClick={() => {
        this.startRecorder()
      }}>Record</Button>
      <Button id="stop" onClick={() => {
        this.stopRecorder()
      }}>Stop</Button>
      <div id="recordingslist"></div>
    </div>)
  }
}

const mapStateToProps = () => {
  return {}
}

export default connect(
  mapStateToProps
)(injectIntl(SpeechToTextAudioWorklet))
*/