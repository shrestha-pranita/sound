import React, { Component } from "react";
import Select from "react-select";
import { Link, Redirect } from "react-router-dom";
import axios from "axios";
import web_link from "../web_link";
import Header from "../elements/header";
export default class AdminExamListView extends Component {
  constructor(props) {
    super(props);
    this.token = localStorage.getItem("token");

    this.state = {
      status_val: "",
      exams: [],
    };
  }

  componentDidMount() {
    if (window.localStorage.getItem("isLoggedIn")) {
      console.log("");
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

    axios({
      method: "get",
      url: web_link + "/api/admin_exam",
    })
      .then((response) => {
        this.setState({
          exams: response.data,
          status_val: "success",
        });
      })
      .catch((error) => {
        console.log(error);
        this.setState({
          status_val: "fail",
        });
      });
  }
  // create exam
  createNewExam = (exam) => {
    fetch(web_link + "/api/admin_exam_create/", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(exam),
    })
      .then((response) => response.json())
      .then((exam) => {
        this.setState({ exams: this.state.exams.concat([exam]) });
      });
  };
  updateExam = (newExam) => {
    fetch(web_link + `/api/exam/${newExam.id}/`, {
      method: "PUT",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(newExam),
    })
      .then((response) => response.json())
      .then((newExam) => {
        const newExams = this.state.exams.map((exam) => {
          if (exam.id === newExam.id) {
            return Object.assign({}, newExam);
          } else {
            return exam;
          }
        });
        this.setState({ exams: newExams });
      });
  };
  deleteExam = (examId) => {
    fetch(web_link + `/api/exam/${examId}/`, {
      method: "DELETE",
      headers: {
        "Content-Type": "application/json",
      },
    }).then(() => {
      this.setState({
        exams: this.state.exams.filter((exam) => exam.id !== examId),
      });
    });
  };

  render() {
    const { status_val, exams } = this.state;
    return (
      <div>
        <Header />
        <div id="wrapper">
          <div className="container h-100">
            <h4 className="text-2xl my-2">Exam List</h4>
            <hr />
            <ToggleableExamForm onExamCreate={this.createNewExam} />

            {status_val === "success" ? (
              <table className="table table-striped">
                <thead>
                  <tr>
                    <th scope="col" className="align-middle">
                      #
                    </th>
                    <th scope="col" className="align-middle">
                      Exam Name
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
                  {exams.map((exam, index) => {
                    return (
                      <tr key={index}>
                        <th scope="row" className="align-middle">
                          {index + 1}
                        </th>
                        <td className="align-middle">{exam.exam_name}</td>
                        <td className="align-middle">{exam.created_at}</td>
                        <td className="align-middle">
                          <Link
                            to={`/admin_record/${exam.id}`}
                            className="text-blue-600"
                          >
                            View Details
                          </Link>
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            ) : null}
            {status_val === "fail" ? (
              <div className="font-weight-bold">
                No exams exist! Please contact the administrator.
              </div>
            ) : null}
          </div>
        </div>
      </div>
    );
  }
}
class ToggleableExamForm extends React.Component {
  state = {
    inCreateMode: false,
  };
  handleCreateClick = () => {
    this.setState({ inCreateMode: true });
  };
  leaveCreateMode = () => {
    this.setState({ inCreateMode: false });
  };
  handleCancleClick = () => {
    this.leaveCreateMode();
  };
  handleFormSubmit = (exam) => {
    this.leaveCreateMode();
    this.props.onExamCreate(exam);
  };
  render() {
    if (this.state.inCreateMode) {
      return (
        <div className="mb-3 p-4" style={{ boxShadow: "0 0 10px #ccc" }}>
          <ExamForm
            onFormSubmit={this.handleFormSubmit}
            onCancelClick={this.handleCancleClick}
          ></ExamForm>
        </div>
      );
    }
    return (
      <button onClick={this.handleCreateClick} className="btn btn-secondary">
        <i className="fas fa-plus"></i>
      </button>
    );
  }
}
class ExamForm extends React.Component {
  state = {
    exam_name: this.props.exam_name || "",
    status: this.props.status || "",
    analyze: this.props.analyze || "",
  };

  handleFormSubmit = (evt) => {
    evt.preventDefault();
    this.props.onFormSubmit({ ...this.state });
  };
  handleExamNameUpdate = (evt) => {
    this.setState({ exam_name: evt.target.value });
  };
  handleStatusUpdate = (evt) => {
    this.setState({ status: evt.target.value });
  };
  handleAnalyzeUpdate = (evt) => {
    this.setState({ analyze: evt.target.value });
  };

  render() {
    const buttonText = this.props.id ? "Update Exam" : "Create Exam";
    const options_status = [
      { value: 1, label: "Active" },
      { value: 0, label: "Inactive" },
    ];
    const options_analyze = [
      { value: 1, label: "Yes" },
      { value: 0, label: "No" },
    ];

    return (
      <form onSubmit={this.handleFormSubmit}>
        <div className="form-group">
          <label>Exam Name</label>
          <input
            type="text"
            placeholder="Enter Exam name"
            value={this.state.exam_name}
            onChange={this.handleExamNameUpdate}
            className="form-control"
          />
        </div>

        <div className="form-group">
          <label>Status</label>
          {/* <input
            type="text"
            placeholder="Author's name"
            value={this.state.author}
            onChange={this.handleAuthorUpdate}
            className="form-control"
          /> */}
          {/* <Select options={options_status} onChange={this.handleStatusUpdate} /> */}
          <select onChange={this.handleStatusUpdate} className="form-control">
            <option value={1}>Active</option>
            <option value={0}>Inactive</option>
          </select>
        </div>

        <div className="form-group">
          <label>Analyze</label>
          {/* <textarea
            className="form-control"
            placeholder="Book Description"
            rows="5"
            value={this.state.description}
            onChange={this.handleDescriptionUpdate}
          >
            {this.state.description}
          </textarea> */}
          {/* <Select
            options={options_analyze}
            onChange={this.handleAnalyzeUpdate}
          /> */}
          <select onChange={this.handleStatusUpdate} className="form-control">
            <option value={1}>Yes</option>
            <option value={0}>No</option>
          </select>
        </div>

        <div className="form-group d-flex justify-content-between">
          <button type="submit" className="btn btn-md btn-primary">
            {buttonText}
          </button>
          <button
            type="button"
            className="btn btn-md btn-secondary"
            onClick={this.props.onCancelClick}
          >
            Cancel
          </button>
        </div>
      </form>
    );
  }
}

class EditableExam extends React.Component {
  state = {
    inEditMode: false,
  };

  enterEditMode = () => {
    this.setState({ inEditMode: true });
  };

  leaveEditMode = () => {
    this.setState({ inEditMode: false });
  };
  handleDelete = () => {
    this.props.onDeleteClick(this.props.id);
  };
  handleUpdate = (exam) => {
    this.leaveEditMode();
    exam.id = this.props.id;
    this.props.onUpdateClick(exam);
  };

  render() {
    const component = () => {
      if (this.state.inEditMode) {
        return (
          <ExamForm
            id={this.props.id}
            exam_name={this.props.exam_name}
            status={this.props.status}
            analyze={this.props.analyze}
            onCancelClick={this.leaveEditMode}
            onFormSubmit={this.handleUpdate}
          />
        );
      }
      return (
        <Exam
          exam_name={this.props.exam_name}
          status={this.props.status}
          analyze={this.props.analyze}
          onEditClick={this.enterEditMode}
          onDeleteClick={this.handleDelete}
        />
      );
    };
    return (
      <div className="mb-3 p-4" style={{ boxShadow: "0 0 10px #ccc" }}>
        {component()}
      </div>
    );
  }
}

class Exam extends React.Component {
  render() {
    return (
      <div className="card" /* style="width: 18rem;" */>
        <div className="card-header d-flex justify-content-between">
          <span>
            <strong>Exam Name: </strong>
            {this.props.exam_name}
          </span>
          <div>
            <span onClick={this.props.onEditClick} className="mr-2">
              <i className="far fa-edit"></i>
            </span>
            <span onClick={this.props.onDeleteClick}>
              <i className="fas fa-trash"></i>
            </span>
          </div>
        </div>
        <div className="card-body">{this.props.analyze}</div>
        <div className="card-footer">
          <strong>Status:</strong> {this.props.status}
        </div>
      </div>
    );
  }
}
