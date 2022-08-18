import React, { Component } from 'react';

import { Link, Redirect } from 'react-router-dom';
import axios from 'axios';
//import web_link from '../web_link';
//import { Empty, Pagination } from 'antd';
import Login from "../components/LoginView";

export default class LoginPage extends Component {
  render() {
     
    return (
      <Login/>
    );
  }
}
