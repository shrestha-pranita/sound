import React, {Component} from 'react';
import {BrowserRouter as Router, Route, Switch} from 'react-router-dom';
import VAD1Page from "./pages/VAD1Page";
import VAD2Page from "./pages/VAD2Page";
import VAD3Page from "./pages/VAD3Page";
import MulSpeaker1Page from "./pages/MulSpeaker1Page";
import MulSpeaker2Page from "./pages/MulSpeaker2Page";
import SpeakerRec1Page from "./pages/SpeakerRecognition1Page";
import TestPage from "./pages/TestPage";

//import './App.css';

class App extends Component {
    render() {
        return (
            
            <Router>
            <div className="App">
                    <Switch>
                        <Route exact path='/' component={VAD1Page} />
                        <Route exact path='/vad1' component={VAD1Page} />
                        <Route exact path='/vad2' component={VAD2Page} />
                        <Route exact path='/vad3' component={VAD3Page} />
                        <Route exact path='/mulspeaker1' component={MulSpeaker1Page} />
                        <Route exact path='/mulspeaker2' component={MulSpeaker2Page} />
                        <Route exact path='/speakerrec1' component={SpeakerRec1Page} />
                        <Route exact path='/vad3' component={VAD3Page} />
                        
                    </Switch>
                
                </div> 
            </Router>
            
        );
    }
}

export default App;
