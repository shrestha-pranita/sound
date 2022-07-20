import React, {Component} from 'react';
import {BrowserRouter as Router, Route, Switch} from 'react-router-dom';
import RecordPage from "./pages/RecordPage";
import TestPage from "./pages/TestPage";

//import './App.css';

class App extends Component {
    render() {
        return (
            
            <Router>
            <div className="App">
                    <Switch>
                        <Route exact path='/' component={RecordPage} />
                        <Route exact path='/audio' component={TestPage} />
                        
                    </Switch>
                
                </div> 
            </Router>
            
        );
    }
}

export default App;
