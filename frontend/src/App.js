import React, {Component} from 'react';
import {BrowserRouter as Router, Route, Switch} from 'react-router-dom';
import LoginPage from "./pages/User/LoginPage";
import ForgetPassword from './pages/User/ForgetPassword';
import RegisterPage from "./pages/User/RegisterPage";
import ExamListPage from "./pages/ExamListPage";
import VAD1Page from "./pages/VAD1Page";
import VAD2Page from "./pages/VAD2Page";
import VAD3Page from "./pages/VAD3Page";
import MulSpeaker1Page from "./pages/MulSpeaker1Page";
import MulSpeaker2Page from "./pages/MulSpeaker2Page";
import SpeakerSamplePage from "./pages/SpeakerSamplePage";
import SpeakerRecognitionPage from "./pages/SpeakerRecognitionPage";
import SpeechPage from "./pages/SpeechPage";
import RecordListPage from "./pages/RecordListPage";
import RecordViewPage from "./pages/RecordViewPage";
import AdminExamListPage from "./pages/AdminExamListPage";
import AdminRecordListPage from "./pages/AdminRecordListPage";
import AdminRecordViewPage from "./pages/AdminRecordViewPage";
class App extends Component {
    render() {
        return (            
            <Router>
            <div className="App">
                    <Switch>
                        <Route exact path='/' component={LoginPage} />
                        <Route exact path='/login' component={LoginPage} />
                        <Route exact path='/forgetpassword' component={ForgetPassword}/>
                        <Route exact path='/dashboard' component={ExamListPage} />
                        <Route exact path='/exam' component={ExamListPage} />
                        <Route path='/register' component={RegisterPage} />
                        <Route exact path='/records/:record_id' component={RecordViewPage} />
                        <Route exact path='/startexam/:exam_id' component={VAD1Page} />
                        <Route exact path='/admin_exam' component={AdminExamListPage} />
                        <Route exact path='/admin_record/:exam_id' component={AdminRecordListPage} />
                        <Route exact path='/admin_record_view/:record_id' component={AdminRecordViewPage} />
                        <Route exact path='/speakersample' component={SpeakerSamplePage} />
                        <Route exact path='/speakerrec' component={SpeakerRecognitionPage} />
                        <Route exact path='/vad1' component={VAD1Page} />
                        <Route exact path='/vad2' component={VAD2Page} />
                        <Route exact path='/vad3' component={VAD3Page} />
                        <Route exact path='/mulspeaker1' component={MulSpeaker1Page} />
                        <Route exact path='/mulspeaker2' component={MulSpeaker2Page} />                      
                        <Route exact path='/speech' component={SpeechPage} />
                        <Route exact path='/record' component={RecordListPage} />

                    </Switch>
                
                </div> 
            </Router>
            
        );
    }
}

export default App;
