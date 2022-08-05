import http from "../http-common";

const vad1 = () => {
  return http.get("/");
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

const recordaudio = () => {
  return http.get("/audio");
};

const speaker_rec1_sample = () => {
  return http.get("/speakerSample");
};


const RecordService = {
  vad1,
  vad1_record,
  vad2_record,
  vad3_record,
  mul_speaker1,
  mul_speaker2,
  speaker_rec1,
  recordaudio,
  speaker_rec1_sample
};

export default RecordService;
