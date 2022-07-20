import { Component } from 'react';
import { Switch, Route } from 'react-router-dom';
import { withRouter } from 'react-router';
import RecordPage from "./pages/RecordPage";
import TestPage from "./pages/TestPage";

class RouterView extends Component {

    constructor(props) {
        super(props);
    }

    render() {
        return <Switch>
            <Route exact path='/' component={RecordPage} />
            <Route exact path='/audio' component={TestPage} />
        </Switch>
    }
}

export default withRouter(RouterView)