import AudioNode from '../worklet'


export default class Recorder {
  constructor(context) {

    this.context = context
    this.recording = false

    console.log(context)

    context.audioWorklet.addModule('processor.js')
      .then(() => {
        console.log('added worklet module')

        navigator.mediaDevices.getUserMedia({audio: true, video: false})
          .then(stream => {
            console.log('Have stream')
            let microphone = context.createMediaStreamSource(stream)
            let wavEncoder = new AudioNode(context)

            microphone.connect(wavEncoder).connect(context.destination)
          })
      })
      .catch((e) => {
        alert('Error getting audio')
        console.log(e)
      })
  }


  record() {
    console.log('Start recorder')
    this.recording = true
  }

  stop() {
    console.log('Stop recorder')
    this.recording = false
  }
}