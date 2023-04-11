# web server

This is aiming to provide an compatiable layer of openai REST interface so that you can use the applicaiton developed for openai.

## How to use openai client to call llama.cpp
1. Start worker application which will listen on local 1234 port.
1. Install openai via pip install openai
1. Override OPENAI_API_BASE enviroment variable to "http://localhost:1234/v1", override OPENAI_API_KEY enviroment variable to any string.
1. Run your openai applications
