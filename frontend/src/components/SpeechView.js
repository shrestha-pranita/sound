import React, { Component } from 'react';
import { Redirect } from 'react-router-dom';
import  web_link from "../web_link";
import Header from '../elements/header';
export default class SpeechView extends Component {
  constructor(props) {
    super(props);
    this.token = localStorage.getItem('token');

    this.state = {
      status_val : ""
    };
  }
  componentDidMount() {
    if (window.localStorage.getItem('isLoggedIn')) {
    } else {
      this.props.history.push('/login');
      return <Redirect to="/login" />;
    }
    if ('token' in localStorage) {
      if ('is_active' in localStorage && 'contract_signed' in localStorage) {
        let active = localStorage.getItem('is_active');
        let contract = localStorage.getItem('contract_signed');
        if (active === 1 && contract === 1) {
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
        .then((res) => {
          console.log(res)
          this.setState({
            status_val : res.status
          });
          
        })
        .catch((err) => console.log(err))

  }
  render() {
    const {
      status_val
    } = this.state;
    return (
      <div>
        <Header />
        <div id="wrapper">
          <div className="container h-100">
            <h4 className="text-2xl my-2">Recording List</h4>
            <hr />

            {status_val === "success" ? (
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
          ) : null}
          {status_val === "fail"? (
            <div className="font-weight-bold">
              No recordings exist! Please contact the administrator.
            </div>
          ) : null}
          </div>
        </div>
      </div>
    );
  }
}
