document.addEventListener("DOMContentLoaded", function() {
    var button = document.getElementById("chatButton");
    var windowBox = document.getElementById("chatWindow");

    if (button && windowBox) {
        button.addEventListener("click", function() {
            if (windowBox.style.display === "none" || windowBox.style.display === "") {
                windowBox.style.display = "block";
            } else {
                windowBox.style.display = "none";
            }
        });
    }
});

function sendMessage() {
    var input = document.getElementById("chatInput");
    var text = input.value.trim();
    if (text === "") return;

    var chatContent = document.getElementById("chatContent");
    var userMsg = document.createElement("div");
    userMsg.innerHTML = "<b>Vous:</b> " + text;
    chatContent.appendChild(userMsg);

    var botMsg = document.createElement("div");
    botMsg.innerHTML = "<b>Bot:</b> " + getBotResponse(text);
    chatContent.appendChild(botMsg);

    input.value = "";
    chatContent.scrollTop = chatContent.scrollHeight;
}

function getBotResponse(message) {
    if (message.toLowerCase().includes("torque")) {
        return "Le torque est la force de rotation lors du squat.";
    } else if (message.toLowerCase().includes("zone")) {
        return "La zone safe est entre 90° et 130°.";
    } else {
        return "Pose-moi tes questions sur le squat !";
    }
}
