import React, { Component } from 'react';
import { Link, Redirect } from 'react-router-dom';
import axios from 'axios';
import  web_link from "../web_link";
//import { Empty, Pagination } from 'antd';
import Header from '../elements/header';


export default class SpeechPage extends Component {
  constructor(props) {
    super(props);
    this.token = localStorage.getItem('token');


  }

  componentDidMount() {
    if (window.localStorage.getItem('isLoggedIn')) {
      let userData = window.localStorage.getItem('user');
    } else {
      this.props.history.push('/login');
      return <Redirect to="/login" />;
    }
    if ('token' in localStorage) {
      if ('is_active' in localStorage && 'contract_signed' in localStorage) {
        let active = localStorage.getItem('is_active');
        let contract = localStorage.getItem('contract_signed');
        if (active == 1 && contract == 1) {
            console.log('')
        } else {
          this.props.history.push('/login');
          return <Redirect to="/login" />;
        }
      }
    } else {
      this.props.history.push('/login');
      return <Redirect to="/login" />;
    }

    let userData = window.localStorage.getItem('user');
    if(userData){
        userData = JSON.parse(userData);
    }

    let user_id = userData.id
    fetch(web_link+'/api/speech', {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
            user_id: user_id
        }),
        })
        .then((res) => res.json())
        .catch((err) => console.log(err))

  }

  //handleChange (value){
  //setoffset((value - 1) * 10);
  //};

  render() {

    return (
      <div>
        <Header />
        <div id="wrapper">

          <div className="container h-100">
            <h4 className="text-2xl my-2">Exam List</h4>
            <hr />

            <table className="table table-striped">
            <thead>
                <tr>
                <th scope="col" className="align-middle">
                    #
                </th>
                <th scope="col" className="align-middle">
                    Exam name
                </th>
                <th scope="col" className="align-middle">
                    Created On
                </th>
                <th scope="col" className="align-middle">
                    Action
                </th>
                </tr>
            </thead>
                <tbody>
                  
                </tbody>
              </table>

          </div>
        </div>
      </div>
    );
  }
}
