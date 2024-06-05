const express = require('express')
const bodyParser = require('body-parser')
const mysql = require('sync-mysql')
const env = require('dotenv').config({ path: "../.env" });
const request = require('request')
const axios = require('axios');
const multer = require("multer");
var FormData = require('form-data');


const storage = multer.diskStorage({
    destination: function (req, file, cb) {
        cb(null, './uploads')
    },
    filename: function(req, file, cb) {
        console.log(file.originalname)
        console.log(file.mimetype)
        const filename = file.originalname
        cb(null, filename)
    }
})

const upload = multer({storage})

const app = express()

app.use(bodyParser.json())
app.use(bodyParser.urlencoded({ extended: true }))
app.use(express.json())
app.use(express.urlencoded({ extended: true }))



var connection = new mysql({
    host: process.env.host,
    user: process.env.user,
    port: process.env.port,
    password: process.env.password,
    database: process.env.database
});

app.get('/Hello', (req, res) => {
    res.send("Hello World")
})


app.post('/test', upload.single("file"), (req, res) => {
    console.log('Check File : ', req.file)
    console.log('Check File Type : ', typeof(req.file))

    const formData = new FormData();
    formData.append("file", req.file);

    console.log('Check forData : ', formData)

    axios.post('http://13.124.236.192:3000/model_test', formData)
                .then(function (result) {
                    console.log('Axios Status : ', result.status);  // 200
                    console.log('DATA :', result.data)
                    console.log('FILE : ', formData)
                })
                .catch(function (error) {
                    console.log(error);
                });
    res.send(200);
})

module.exports = app;
