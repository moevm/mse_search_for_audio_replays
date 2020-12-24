let noise_download_panel;

$(document).ready(function () {
    console.log("ready");
    noise_download_panel = $("#noise_download_panel");
    setNoisePanelVision()

    $("#filter").on("change", setNoisePanelVision);

    $("#repetitions").on("change", setNoisePanelVision);

})

function setNoisePanelVision() {
    console.log("change");
    if ($("#filter").is(':checked')) {
        noise_download_panel.show();
    } else {
        noise_download_panel.hide();
    }
}