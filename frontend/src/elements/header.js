import React, { Component } from 'react';
import {Link, Redirect} from "react-router-dom";
import TitleComponent from "../pages/title";


export default class Header extends Component {

    constructor(props) {
        super(props);
        this.handleClickLogout = this.handleClickLogout.bind(this)
    }

    state = {
        toDashboard: false,
        user: {}
    };

    
    componentDidMount(){
        let userData = localStorage.getItem('user');
        if(userData){
            userData = JSON.parse(userData);
            this.setState({user: userData})
        }
    }


    handleClickLogout(){
        console.log("what")
        localStorage.clear()
        this.setState({ toDashboard: true });
    }

    render() {
        if (this.state.toDashboard === true) {
            return <Redirect to='/' />
        }
        return (
            <nav className="navbar navbar-expand navbar-dark bg-dark static-top">
                <TitleComponent title="NS Devil"></TitleComponent>
                <Link to={'/'} className="navbar-brand mr-1">NS DEVIL</Link>
                <div>&nbsp;</div>
                <div>&nbsp;</div>
                <div>&nbsp;</div>
                <div className= "ml-auto">
                    <ul className="navbar-nav ml-auto ml-md-0 nav navbar-nav navbar-right">
                        <li className="nav-item dropdown no-arrow">
                            <span className="nav-link dropdown-toggle" id="userDropdown" role="button"
                            data-toggle="dropdown" aria-haspopup="true" aria-expanded="false">
                                VAD
                            </span>

                            <div className="dropdown-menu dropdown-menu-right" aria-labelledby="userDropdown">
                                <Link to={'/vad1'} className="dropdown-item">VAD 1 (Good model)</Link>
                                <Link to={'/vad2'} className="dropdown-item">VAD 2</Link>
                                <Link to={'/vad3'} className="dropdown-item">VAD 3</Link>
                            </div>
                        </li>
                        
                        <li>
                            &nbsp;
                        </li>

                        <li className="nav-item dropdown no-arrow">
                            <span className="nav-link dropdown-toggle" id="userDropdown" role="button"
                            data-toggle="dropdown" aria-haspopup="true" aria-expanded="false">
                                Multiple speaker
                            </span>

                            <div className="dropdown-menu dropdown-menu-right" aria-labelledby="userDropdown">
                                <Link to={'/noisedetection'} className="dropdown-item">Noise Detection</Link>
                                <Link to={'/mulspeaker2'} className="dropdown-item">Multiple Speaker 2</Link>
                                <Link to={'/test'} className="dropdown-item">Test</Link>
                            </div>
                        </li>

                        <li>
                            &nbsp;
                        </li>
                        <li className="nav-item dropdown no-arrow">
                            <span className="nav-link dropdown-toggle" id="userDropdown" role="button"
                            data-toggle="dropdown" aria-haspopup="true" aria-expanded="false">
                                Speaker recognition
                            </span>

                            <div className="dropdown-menu dropdown-menu-right" aria-labelledby="userDropdown">
                                <Link to={'/speakerrec1'} className="dropdown-item">Speaker Recognition 1</Link>
                                <Link to={'/vad2'} className="dropdown-item">Speaker Recognition 2</Link>
                            </div>
                        </li>

                        <li className="nav-item dropdown no-arrow">
                            <span className="nav-link dropdown-toggle" id="userDropdown" role="button"
                            data-toggle="dropdown" aria-haspopup="true" aria-expanded="false">
                                Speech
                            </span>

                            <div className="dropdown-menu dropdown-menu-right" aria-labelledby="userDropdown">
                                <Link to={'/speech'} className="dropdown-item">Speech</Link>
                                <Link to={'/speech'} className="dropdown-item">Timestamp</Link>
                            </div>
                        </li>

                        <li className="nav-item dropdown no-arrow">
                            <span className="nav-link dropdown-toggle" id="userDropdown" aria-haspopup="true" aria-expanded="false" >
                                <Link to={'#'} onClick={this.handleClickLogout} className="dropdown-item" data-toggle="modal" data-target="#logoutModal">Logout</Link>
                            </span>
                        </li>
                    </ul>
                </div>
                
            </nav>
        );
    }
}
