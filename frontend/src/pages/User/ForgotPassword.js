import React, { Component } from "react";
import axios from "axios";
import { Link, Redirect } from "react-router-dom";
import web_link from "../../web_link";

export default class ForgotPassword extends Component {
  state = {
    newpassword: "",
    repassword: "",
    username: "",
  };
  handleUsernameChange = (e) => {
    this.setState({
      username: e.target.value,
    });
  };
  handleNewpChange = (event) => {
    this.setState({ newpassword: event.target.value });
  };
  handleRepChange = (event) => {
    this.setState({ repassword: event.target.value });
  };

  handleSubmit = (event) => {
    event.preventDefault();
    this.setState({ isLoading: true });
    const url = web_link + "/api/forgetpassword";
    const username = this.state.username;
    const newpassword = this.state.newpassword;
    const repassword = this.state.repassword;

    if(newpassword !== repassword){
        alert("Password does not match");
        return;
    }

    let bodyFormData = new FormData();
    bodyFormData.set("username", username);
    bodyFormData.set("password", newpassword);
    axios
      .put(url, bodyFormData)
      .then((result) => {
        this.setState({ isLoading: false });
        if (result.data.status !== "fail") {
          this.setState({ redirect: true, authError: true });
        } else {
          this.setState({ redirect: false, authError: true });
        }
      })
      .catch((error) => {
        console.log(error);
        this.setState({ authError: true, isLoading: false });
      });
  };

  renderRedirect = () => {
    if (this.state.redirect) {
      return <Redirect to="/login" />;
    }
  };

  render() {
    const isLoading = this.state.isLoading;
    return (
      <div className="container">
        <div className="card card-login mx-auto mt-5">
          <div className="card-header">Reset Password</div>
          <div className="card-body">
            <form onSubmit={this.handleSubmit}>
              <div className="form-group">
                <div className="form-label-group">
                  <input
                    type="text"
                    id="inputName"
                    className="form-control"
                    placeholder="name"
                    name="name"
                    onChange={this.handleUsernameChange}
                    required
                  />
                  <label htmlFor="inputName">Username</label>
                </div>
              </div>
              <div className="form-group">
                <div className="form-label-group">
                  <input
                    type="password"
                    className="form-control"
                    id="inputNewpassword"
                    placeholder="******"
                    name="newpassword"
                    onChange={this.handleNewpChange}
                    required
                  />
                  <label htmlFor="inputNewpassword">New Password</label>
                </div>
              </div>
              <div className="form-group">
                <div className="form-label-group">
                  <input
                    type="password"
                    className="form-control"
                    id="inputRepassword"
                    placeholder="******"
                    name="repassword"
                    onChange={this.handleRepChange}
                    required
                  />
                  <label htmlFor="inputRepassword">Retype Password</label>
                </div>
              </div>

              <div className="form-group">
                <button
                  className="btn btn-primary btn-block"
                  type="submit"
                  disabled={this.state.isLoading ? true : false}
                >
                  Reset Password &nbsp;&nbsp;&nbsp;
                  {isLoading ? (
                    <span
                      className="spinner-border spinner-border-sm"
                      role="status"
                      aria-hidden="true"
                    ></span>
                  ) : (
                    <span></span>
                  )}
                </button>
              </div>
            </form>
            <div className="text-center">
              <Link className="d-block small mt-3" to={""}>
                Login Your Account
              </Link>
            </div>
          </div>
        </div>
        {this.renderRedirect()}
      </div>
    );
  }
}