import React, { Component } from 'react';
import axios from 'axios';
import { ReactMediaRecorder } from "react-media-recorder";
//import Audio from "../components/RecordView2";
import  web_link from "../web_link";


export default class TestPage extends React.Component {
    componentDidMount() {
        axios.get(web_link + '/api/audio', {
            headers: {
                Authorization: `Bearer ${this.token}`
            }
        })
        .then(res=>{
            if(res.status === 200){
                // set current userprofile data
                console.log("first")

            }
            if(res.status === 204){
                console.log("second")

            }
            console.log(res)

        })        
    }
    render() {
        return (
        <p>here</p>
        );
    }
}
