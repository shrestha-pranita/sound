import React, { Component } from 'react';
import { Link, Redirect, useParams, withRouter  } from 'react-router-dom';
import axios from 'axios';
import  web_link from "../web_link";
import Header from '../elements/header';
import ReactAudioPlayer from 'react-audio-player';

export default class AdminRecordView extends Component {
  constructor(props) {
    super(props);
    this.token = localStorage.getItem('token');

    this.state = {
      status_val : "",
      filename : "",
      records : []
    };

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
    
    const pathname = window.location.pathname
    const slug = pathname.split("/").pop();
    console.log(slug)

    fetch(web_link+'/api/admin_record_view/'+slug, {
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
            console.log(res.data)
            this.setState({
              records: res.data,
              status_val: "success",
              filename : res.filename
          });

        })
        .catch((err) => {
          this.setState({
            status_val: "fail",
          });
        })

  }

  render() {
    const {
      status_val,
      records,
      filename
    } = this.state;
    return (
      <div>
        <Header />
        <div id="wrapper">
          <div className="container h-100">
            <h4 className="text-2xl my-2">Recording List</h4>
            <hr />
            <ReactAudioPlayer
              src={`${web_link}${filename}`}
              controls
            />
            {status_val == "success" ? (
            <table className="table table-striped">
            <thead>
              <tr>
                <th scope="col" className="align-middle">
                  Recording name
                </th>
                <th scope="col" className="align-middle">
                  Start time
                </th>
                <th scope="col" className="align-middle">
                  End time
                </th>
              </tr>
            </thead>
            <tbody>
              {records.map(item => (
                <tr>
                  <td className="align-middle">
                    <ReactAudioPlayer
                      src={`${web_link}${item.full}`}
                      controls
                    />
                  </td>
                  <td className="align-middle">
                    {`${item.start}`} seconds
                  </td>

                  <td className="align-middle">
                    {`${item.end}`} seconds
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
          ) : null}
          {status_val == "fail"? (
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
