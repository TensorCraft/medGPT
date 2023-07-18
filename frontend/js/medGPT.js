var websocket;
var current_msg = null;

context = "";

function initWebSocket() {
    var wsUrl = "ws://127.0.0.1:8080";
    websocket = new WebSocket(wsUrl);

    websocket.onopen = function () {
        console.log("WebSocket connection established.");
    };

    websocket.onmessage = function (event) {
        var message = event.data;

        if (message === "<BEAT>")
            return;
        if (message == "<OVER>") {
            websocket.close();
            current_msg = null;
            alert("Currently under high demand, try again later!")
        }

        if (current_msg == null) {
            current_msg = document.createElement("div")
            current_msg.className = "msg_received"
            current_msg.appendChild(document.createElement('p'))
            document.getElementById("chatlist").appendChild(current_msg);
            current_msg = current_msg.childNodes[0];
        }

        if (message != "<EOA>") {
            current_msg.innerHTML += message + " ";
            context += message + " ";
        }
        else {
            context += " <EOA>\n";
            current_msg = null;
        }
    };

    websocket.onclose = function () {
        console.log("WebSocket connection closed.");
        alert("Connection Failed!");
        location.reload();
    };
}

function sendMessage() {
    var text = document.getElementById("chat-input").value;
    context += "User: " + text + "\n" + "medGPT: "
    document.getElementById("chat-input").value = "";

    var new_msg = document.createElement("div");
    new_msg.className = "msg_sent";
    document.getElementById("chatlist").appendChild(new_msg);
    new_msg.innerHTML = "<p>" + text + "</p>";

    if (websocket.readyState === WebSocket.OPEN) {
        websocket.send(context);
    }
}

window.onload = function () {
    initWebSocket();

    document.getElementById("clean-btn").onclick = function () {
        location.reload();
    }

    document.getElementById("send-btn").onclick = function () {
        sendMessage()
    }
}