#pragma once

#include <string>

int engine_init(int argc, char **argv);
void engine_add_request();

struct mg_connection;

struct request
{
    struct mg_connection *c;
    std::string model;
    std::string prompt;
    std::string stop;
    int n_token;
    int stream_interval;
    float n_temp;
};