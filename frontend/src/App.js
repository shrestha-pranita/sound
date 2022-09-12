import React, {Component} from 'react';
import {BrowserRouter as Router, Route, Switch} from 'react-router-dom';
import LoginPage from "./pages/User/LoginPage";
import RegisterPage from "./pages/User/RegisterPage";
import ExamListPage from "./pages/ExamListPage";
import VAD1Page from "./pages/VAD1Page";
import VAD2Page from "./pages/VAD2Page";
import VAD3Page from "./pages/VAD3Page";
import MulSpeaker1Page from "./pages/MulSpeaker1Page";
import MulSpeaker2Page from "./pages/MulSpeaker2Page";
import SpeakerRec1Page from "./pages/SpeakerRecognition1Page";
import SpeechPage from "./pages/SpeechPage";
import RecordListPage from "./pages/RecordListPage";
import RecordViewPage from "./pages/RecordViewPage";
import AdminExamListPage from "./pages/AdminExamListPage";
import AdminRecordListPage from "./pages/AdminRecordListPage";
import AdminRecordViewPage from "./pages/AdminRecordViewPage";

//import './App.css';

class App extends Component {
    render() {
        return (
            
            <Router>
            <div className="App">
                    <Switch>
                        <Route exact path='/' component={LoginPage} />
                        <Route exact path='/login' component={LoginPage} />
                        <Route exact path='/dashboard' component={ExamListPage} />
                        <Route exact path='/exam' component={ExamListPage} />
                        <Route path='/register' component={RegisterPage} />
                        <Route exact path='/vad1' component={VAD1Page} />
                        <Route exact path='/vad2' component={VAD2Page} />
                        <Route exact path='/vad3' component={VAD3Page} />
                        <Route exact path='/mulspeaker1' component={MulSpeaker1Page} />
                        <Route exact path='/mulspeaker2' component={MulSpeaker2Page} />
                        <Route exact path='/speakerrec1' component={SpeakerRec1Page} />
                        <Route exact path='/speech' component={SpeechPage} />
                        <Route exact path='/record' component={RecordListPage} />
                        <Route exact path='/records/:record_id' component={RecordViewPage} />
                        <Route exact path='/startexam/:exam_id' component={VAD1Page} />
                        <Route exact path='/admin_exam' component={AdminExamListPage} />
                        <Route exact path='/admin_record/:exam_id' component={AdminRecordListPage} />
                        <Route exact path='/admin_record_view/:record_id' component={AdminRecordViewPage} />
                    </Switch>
                
                </div> 
            </Router>
            
        );
    }
}

export default App;
