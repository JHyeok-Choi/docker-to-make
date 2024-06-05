const sendAudio = blob => {
    if (blob == null) return;

    let filename = new Date().toString() + ".wav";
    
    console.log('b', blob)
    console.log(file)


    let fd = new FormData();
    fd.append("fname", filename)
    fd.append("file", blob);
    fetch('/test', {method:"POST", body:blob})
            .then(function(res) {
                console.log('okok')
            }).then(console.log)
            .catch(console.error);
}



const file = new Blob;
	let audioIN = { audio: true };
	// audio is true, for recording

	// Access the permission for use
	// the microphone
	navigator.mediaDevices.getUserMedia(audioIN)

	// 'then()' method returns a Promise
	.then(function (mediaStreamObj) {

		// Connect the media stream to the
		// first audio element
		let audio = document.querySelector('audio');
		//returns the recorded audio via 'audio' tag

		// 'srcObject' is a property which 
		// takes the media object
		// This is supported in the newer browsers
		if ("srcObject" in audio) {
		audio.srcObject = mediaStreamObj;
		}
		else { // Old version
		audio.src = window.URL
			.createObjectURL(mediaStreamObj);
		}

		// It will play the audio
		audio.onloadedmetadata = function (ev) {
		// Play the audio in the 2nd audio
		// element what is being recorded
		audio.play();
		};

		let start = document.getElementById('btnStart');
		let stop = document.getElementById('btnStop');

		// 2nd audio tag for play the audio
		let playAudio = document.getElementById('audioPlay');

		// This is the main thing to recorded 
		// the audio 'MediaRecorder' API
		let mediaRecorder = new MediaRecorder(mediaStreamObj);
        MediaRecorder.isTypeSupported("audio/wav;codecs=MS_PCM")
		// Pass the audio stream 

		// Start event
		start.addEventListener('click', function (ev) {
		mediaRecorder.start();
		})

		// Stop event
		stop.addEventListener('click', function (ev) {
		mediaRecorder.stop();

		});

		mediaRecorder.ondataavailable = function (ev) {
        dataArray.push(ev.data);
        console.log('buffer ?: ', dataArray)
		}
        

		let dataArray = [];


		mediaRecorder.onstop = function (ev) {


		let audioData = new Blob(dataArray, 
					{ 'type' : 'audio/wav; codecs=MS_PCM' });


        var formData = new FormData();
        formData.append('file', audioData)
        console.log('real ??? : ', audioData)
        console.log('test ab : ', audioData.arrayBuffer())
        console.log('data 111 : ', Object.keys(audioData))
        
        for (let key of formData.keys()) {
            console.log('fkey : ', key);
        };
        for (let value of formData.values()) {
            console.log('fval : ', value)
        };
        for (let pair of formData.entries()) {
            console.log('fpair : ', pair)
        }
        
        let xhr = new XMLHttpRequest();
        xhr.open('POST', '/test');
        xhr.send(formData);
		

		// After fill up the chunk 
		// array make it empty
		dataArray = [];

		// Creating audio url with reference 
		// of created blob named 'audioData'
		let audioSrc = window.URL
			.createObjectURL(audioData);


        globalThis.file = audioData;
        
		// Pass the audio url to the 2nd video tag
		playAudio.src = audioSrc;
		}
	})
	.catch(function (err) {
		console.log(err.name, err.message);
	});