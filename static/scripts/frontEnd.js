let noise_download_panel;
let main_interval

$(document).ready(function () {
    socket = io(window.location.origin);
    console.log("ready");
    noise_download_panel = $("#noise_download_panel");
    setNoisePanelVision()

    $("#filter").on("change", setNoisePanelVision);
    $("#repetitions").on("change", setNoisePanelVision);

    socket.on("connect", function () {
        if (main_interval === undefined) {
            main_interval = setInterval(socket.emit("update"))
        } else {
            clearInterval(main_interval);
            main_interval = setInterval();
        }

    });

})

function setNoisePanelVision() {
    console.log("change");
    if ($("#filter").is(':checked')) {
        noise_download_panel.show();
    } else {
        noise_download_panel.hide();
    }
}