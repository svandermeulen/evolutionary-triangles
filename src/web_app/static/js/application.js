$(document).ready(function () {
    //connect to the socket server.
    var socket = io.connect('http://' + document.domain + ':' + location.port + '/index');

    //receive details from server
    socket.on('reload', function () {
        location.reload();
    })
});