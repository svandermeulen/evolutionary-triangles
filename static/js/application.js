$(document).ready(function () {
    //connect to the socket server.
    var socket = io.connect('http://' + document.domain + ':' + location.port + '/index');

    socket.on('generation', function ({integer, total}) {
        console.log("Computed generations: " + integer + " of " + total);
        numbers_string = 'Computed generations: ' + integer.toString() + '/' + total.toString();
        $('#generation').html(numbers_string);
    });

});
