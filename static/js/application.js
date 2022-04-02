$(document).ready(function () {
    //connect to the socket server.
    let socket = io.connect('http://' + document.domain + ':' + location.port + '/index');

    //receive details from server
    socket.on('reload', function () {
        location.reload();
    })

    socket.on('generation', function ({integer, total}) {
        let numbers_string = `Computed generations: ${integer}/${total}`;
        let progress = Math.floor((integer / total) * 100)
        console.log(numbers_string);
        $( "#generation" ).progressbar({
            value: progress
        });
    });

});
