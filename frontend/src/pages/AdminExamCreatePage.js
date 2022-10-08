import React from 'react';
import AdminExamCreateView from "../components/AdminExamCreateView";
import {useHistory, Route , withRouter } from 'react-router-dom';

export default class AdminExamCreatePage extends React.Component {
  render() {     
    return (
      <AdminExamCreateView/>
    );
  }
}


