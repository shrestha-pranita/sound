import React from "react";
import { Link, Redirect } from "react-router-dom";
import web_link from "../web_link";
import Header from "../elements/header";
import "bootstrap/dist/css/bootstrap.min.css";
import { Spinner } from "react-bootstrap";

export default class AdminRecordListView extends React.Component {
  constructor(props) {
    super(props);
    this.token = localStorage.getItem("token");

    this.state = {
      status_val: "",
      records: [],
      exams: [],
      loading: false,
      analyzeIsDisabled: false,
    };

    this.onAnalyzeHandler = this.onAnalyzeHandler.bind(this);
  }
  componentDidMount() {
    if (window.localStorage.getItem("isLoggedIn")) {
      
    } else {
      this.props.history.push("/login");
      return <Redirect to="/login" />;
    }
    if ("token" in localStorage) {
      if ("is_active" in localStorage && "contract_signed" in localStorage) {
        let active = localStorage.getItem("is_active");
        let contract = localStorage.getItem("contract_signed");
        if (active === 1 && contract === 1) {
          console.log("");
        } else {
          this.props.history.push("/login");
          return <Redirect to="/login" />;
        }
      }
    } else {
      this.props.history.push("/login");
      return <Redirect to="/login" />;
    }

    let userData = window.localStorage.getItem("user");
    if (userData) {
      userData = JSON.parse(userData);
    }

    let user_id = userData.id;
    let user_name = userData.username;
    const pathname = window.location.pathname;
    const slug = pathname.split("/").pop();
    console.log(slug);

    fetch(web_link + "/api/admin_record/" + slug, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        user_id: user_id,
        exam_id: slug,
        user_name: user_name,
      }),
    })
      .then((res) => res.json())
      .then((res) => {
        console.log(res.exam);
        this.setState({
          records: res.data,
          exams: res.exam,
          status_val: "success",
        });
        console.log(this.state.exams[0].exam_name);
      })
      .catch((err) => {
        this.setState({
          status_val: "fail",
        });
      });
  }

  onAnalyzeHandler() {
    this.setState({
      analyzeIsDisabled: true,
    });
    console.log("what");
    const pathname = window.location.pathname;
    const slug = pathname.split("/").pop();
    this.setState({ loading: true });
    fetch(web_link + "/api/admin_analyze/" + slug, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        exam_id: slug,
      }),
    })
      .then((res) => res.json())
      .then((res) => {
        window.location.reload(true);
        this.props.history.push("/admin_record/" + slug);
        return <Redirect to="/login" />;
      })
      .catch((err) => {
        this.setState({
          status_val: "fail",
        });
      });
  }

  toggleLoader = () => {
    if (!this.state.loading) {
      this.setState({ loading: true });
    } else {
      this.setState({ loading: false });
    }
  };

  render() {
    const { status_val, records, exams, analyzeIsDisabled } = this.state;
    return (
      <div>
        <Header />
        <div id="wrapper">
          <div className="btnContainer">
            {this.state.loading ? (
              <Spinner
                style={{ marginBottom: 27 }}
                animation="border"
                variant="danger"
              />
            ) : null}
          </div>

          <div className="container h-100">
            {status_val === "success" ? (
              <div>
                <h1>Exam name : {exams[0].exam_name}</h1>
                <div>&nbsp;</div>
                {exams[0].analyze === 0 ? (
                  <button
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
                      transform: "translateY(0)",
                    }}
                    onClick={this.onAnalyzeHandler}
                    disabled={analyzeIsDisabled}

                  >
                    Analyze
                  </button>
                ) : (
                  <button
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
                      transform: "translateY(0)",
                    }}
                    disabled="disabled"
                  >
                    Already Analyzed
                  </button>
                )}
              </div>
            ) : null}
            <div>&nbsp;</div>
            <h4 className="text-2xl my-2">User List</h4>
            <hr />

            {status_val === "success" ? (
              <table className="table table-striped">
                <thead>
                  <tr>
                    <th scope="col" className="align-middle">
                      #
                    </th>
                    <th scope="col" className="align-middle">
                      User Id
                    </th>
                    <th scope="col" className="align-middle">
                      Created At
                    </th>
                    {exams[0].analyze === 1 ? (
                      <th scope="col" className="align-middle">
                        Action
                      </th>
                    ) : null}
                  </tr>
                </thead>
                <tbody>
                  {records.map((record, index) => {
                    let usernames = record.filename.split("/");
                    let user_name = usernames[4];
                    return (
                      <tr key={index}>
                        <th scope="row" className="align-middle">
                          {index + 1}
                        </th>
                        <td className="align-middle">{user_name}</td>
                        <td className="align-middle">{record.created_at}</td>
                        {exams[0].analyze === 1 ? (
                          <td className="align-middle">
                            <Link
                              to={`/admin_record_view/${record.id}`}
                              className="text-blue-600"
                            >
                              View Recording
                            </Link>
                          </td>
                        ) : null}
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            ) : null}
            {status_val === "fail" ? (
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
