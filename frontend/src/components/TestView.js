import React from "react";
//import  web_link from "../web_link";
import  web_link from "../web_link";

function App() {

    async function getAllData() {
      console.log("Here")

      const blob =
            new Blob(
                ["This is some important text"],
                { type: "text/plain" }
            );
  
        // Creating a new blob  
        // Hostname and port of the local server
        fetch(web_link+'/api/test', {
  
            // HTTP request type
            method: "POST",
            headers: {
              'Accept': 'application/json, text/plain',
              'Content-Type': 'application/json;charset=UTF-8'
          },
          mode: 'no-cors',
            // Sending our blob with our request
            body: blob
        })
        .then(response => alert('Blob Uploaded'))
        .catch(err => alert(err));
      /*
      fetch(web_link+'/api/test', {
        method: "POST",
        headers: {
            "Content-Type": "application/json",
            //'Access-Control-Allow-Origin': 'http://localhost:8000',
            //'Access-Control-Allow-Credentials': 'true'
        },
        body: JSON.stringify({
            data: [{"test":"test"}],
        }),
    })
    */
      //console.log("Here")
      //const res = await fetch(`${web_link}/api/test`);
      //console.log(res)
      //try {
        //const res = await fetch(`${web_link}/api/test`);
        
        //if (!res.ok) {
          //throw new Error("tst");
        //}
        //const data = await res.json();
        //console.log(data)
        //console.log("what")

      //} catch (err) {
       // console.log("error");
      //}
    }


  

    return (
      <div className="card">
        <div className="card-header">React Fetch GET - BezKoder.com</div>
        <div className="card-body">
          <div className="input-group input-group-sm">
            <button className="btn btn-sm btn-primary" onClick={getAllData}>Get All</button>
            <input type="text" className="form-control ml-2" placeholder="Id" />
            <div className="input-group-append">
              <button className="btn btn-sm btn-primary">Get by Id</button>
            </div>
            <input type="text"  className="form-control ml-2" placeholder="Title" />
            <div className="input-group-append">
              <button className="btn btn-sm btn-primary" >Find By Title</button>
            </div>
            <button className="btn btn-sm btn-warning ml-2">Clear</button>
          </div>   
          
          <div className="alert alert-secondary mt-2" role="alert"><pre></pre></div> 
        </div>
      </div>
    );
  }
  export default App;