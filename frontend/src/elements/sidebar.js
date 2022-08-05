import React, { Component } from 'react';
import { Link } from 'react-router-dom';


export default class Sidebar extends Component {

    state = {
        user: {}
    };


    componentDidMount() {
        let userData = localStorage.getItem('user');
        if (userData) {
            userData = JSON.parse(userData);
            this.setState({ user: userData })
        }
    }


    render() {
        return (
            <div id="wrapper">
                <ul className="sidebar navbar-nav">
                    <li className="nav-item active">
                        <Link to={'/dashboard'} className="nav-link"><i className="fas fa-fw fa-tachometer-alt"></i>
                            <span>&nbsp;Dashboard</span></Link>
                    </li>

                    <li className="nav-item">
                        <Link to={'/profile'} className="nav-link"><i className="fas fa-fw fa-user"></i>
                            <span>&nbsp;Profile</span></Link>
                    </li>
                    {this.state.user.contract_signed == 0 && this.state.user.is_approved == 1 && <>
                        <li className="nav-item">
                            <Link to={'/contract'} className="nav-link"><i className="fas fa-fw fa-file-audio"></i>
                                <span>&nbsp;Contract</span></Link>
                        </li>
                    </>
                    }

                    {this.state.user.is_approved && this.state.user.contract_signed && <>
                        <li className="nav-item">
                            <Link to={'/recordlist'} className="nav-link"><i className="fas fa-fw fa-file-audio"></i>
                                <span>&nbsp;Recordings</span></Link>
                        </li>

                        <li className="nav-item">
                            <Link to={'/exams'} className="nav-link"><i className="fas fa-question"></i>
                                <span>&nbsp;Exam list</span></Link>
                        </li>
                    </>}
                </ul>
            </div>
        );
    }
}
