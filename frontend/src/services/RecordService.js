import http from "../http-common";

const record = () => {
  return http.get("/");
};

const recordaudio = () => {
  return http.get("/audio");
};


const RecordService = {
  record,
  recordaudio
};

export default RecordService;
