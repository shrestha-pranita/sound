import React, { Component } from 'react';
import { Link, Redirect } from 'react-router-dom';
import axios from 'axios';
import  web_link from "../web_link";
//import { Empty, Pagination } from 'antd';
import Header from '../elements/header';


export default class AdminRecordListView extends Component {
  constructor(props) {
    super(props);
    this.token = localStorage.getItem('token');

    this.state = {
      status_val : "",
      records : []
    };

    
  }


  componentDidMount() {
    const currentURL = window.location.href // returns the absolute URL of a page


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

    fetch(web_link+'/api/admin_record/' + slug, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
            user_id: user_id,
            exam_id: slug
        }),
        })
        .then((res) => res.json())
        .then((res) => {
          console.log(res.data)
          this.setState({
            records: res.data,
            status_val: "success"
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
      records
    } = this.state;
    return (
      <div>
        <Header />
        <div id="wrapper">
          
          <div className="container h-100">
          <div><button
    style={{
      padding: "0.8rem 2rem",
      border: "none",
      backgroundColor: "#0000FF",
      marginLeft: "15px",
      fontSize: "1rem",
      cursor: "pointer",
      color: "white",
      borderRadius: "5px",
      fontWeight: "bold",
      transition: "all 300ms ease-in-out",
      transform: "translateY(0)"
    }}
    //onClick={() => {
      //onFinalSubmitHandler();
    //}}

    //disabled = {submitIsDisabled}
  >
    Analyze
  </button>
  </div>
            <h4 className="text-2xl my-2">Recording List</h4>
            <hr />

            {status_val == "success" ? (
            <table className="table table-striped">
            <thead>
              <tr>
                <th scope="col" className="align-middle">
                  #
                </th>
                <th scope="col" className="align-middle">
                  Recording name
                </th>
                <th scope="col" className="align-middle">
                  Created At
                </th>
                <th scope="col" className="align-middle">
                  Action
                </th>
              </tr>
            </thead>
            <tbody>
              {records.map((record, index) => {
                return (
                  <tr key={index}>
                    <th scope="row" className="align-middle">
                      {index + 1}
                    </th>
                    <td className="align-middle">{record.filename}</td>
                    <td className="align-middle">{record.created_at}</td>
                    <td className="align-middle">
                      <Link
                        to={`/records/${record.id}`}
                        className="text-blue-600"
                      >
                        View Recording
                      </Link>
                    </td>
                  </tr>
                );
              })}
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
