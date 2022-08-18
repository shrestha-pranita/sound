import http from "../http-common";

const login = () => {
  return http.get("/");
};

const login1 = () => {
  return http.get("/login");
};

const vad1_record = () => {
  return http.get("/vad1");
};

const vad2_record = () => {
  return http.get("/vad2");
};

const vad3_record = () => {
  return http.get("/vad3");
};

const mul_speaker1 = () => {
  return http.get("/mulspeaker1");
};

const mul_speaker2 = () => {
  return http.get("/mulspeaker2");
};

const speaker_rec1 = () => {
  return http.get("/speakerrec1");
};

const speaker_sample = () => {
  return http.get("/speakersample");
};

const noise_detection = () => {
  return http.get("/noisedetection");
};

const recordaudio = () => {
  return http.get("/audio");
};

const test_page = () => {
  return http.get("/test");
};

const speech_page = () => {
  return http.get("/speech");
};

const speaker_rec1_sample = () => {
  return http.get("/speakerSample");
};


const RecordService = {
  login,
  login1,
  vad1,
  vad1_record,
  vad2_record,
  vad3_record,
  mul_speaker1,
  mul_speaker2,
  speaker_rec1,
  recordaudio,
  speaker_rec1_sample,
  test_page,
  speech_page,
};

export default RecordService;
