#include "mongoose.h"
#include "cJSON.h"

#include <cassert>
#include <cstring>
#include <fstream>
#include <string>
#include <iterator>
#include <algorithm>
#include <thread>
#include <iostream>
#include <sstream>

#include "thread.h"
#include "engine.h"

static void server_completion(struct mg_connection *c, mg_http_message *hm) {
    std::string prompt = mg_json_get_str(hm->body, "$.prompt");
    if (prompt.empty()) prompt = "<|endoftext|>";

    // TODO:
    //  Support logit_bias, stop
    Request * req = new CompletionRequest(c);

    // Check if we can still enque new requests
    if (engine_add_request(req))
    {
        // this is error case
        mg_http_reply(c, 500, "", "Server is too busy");
        delete req;
        return;
    }
}

static void server_embeddings(struct mg_connection *c, struct mg_http_message *hm) {
    
}

static void server_models(struct mg_connection *c, struct mg_http_message *hm) {
    cJSON *data = cJSON_CreateArray();

    // create first object in data array
    cJSON *model0 = cJSON_CreateObject();
    cJSON_AddStringToObject(model0, "id", "llama");
    cJSON_AddStringToObject(model0, "object", "model");
    cJSON_AddStringToObject(model0, "owned_by", "Meta");
    cJSON_AddItemToObject(model0, "permission", cJSON_CreateArray());
    cJSON_AddItemToArray(data, model0);

    // create top-level object
    cJSON *root = cJSON_CreateObject();
    cJSON_AddItemToObject(root, "data", data);
    cJSON_AddStringToObject(root, "object", "list");

    char *json_text = cJSON_Print(root);
    mg_http_reply(c, 200, "", json_text);
    free(json_text);

    cJSON_Delete(data);
}

static void server_callback(struct mg_connection *c, int ev, void *ev_data, void *fn_data)
{
    if (ev == MG_EV_HTTP_MSG)
    {
        struct mg_http_message *hm = (struct mg_http_message *)ev_data;

        if (mg_http_match_uri(hm, "/v1/models"))
        {
            server_models(c, hm);
        }
        else if (mg_http_match_uri(hm, "/v1/completions"))
        {
            server_completion(c, hm);
        }
        else if (mg_http_match_uri(hm, "/v1/chat/completions"))
        {

        }
        else if (mg_http_match_uri(hm, "/v1/embeddings"))
        {
            server_embeddings(c, hm);
        }

        mg_http_reply(c, 404, "", "Not Found");
    }

    if (ev == MG_EV_CLOSE)
    {
        // TODO: cancel the request

    }
}

int main(int argc, char **argv)
{
    engine_init(argc, argv);
    // initialize mongoose
    struct mg_mgr mgr;
    mg_mgr_init(&mgr);

    mg_http_listen(&mgr, "http://localhost:1234", server_callback, NULL);

    int i = 0;
    for (;;)
    {
        mg_mgr_poll(&mgr, 1000);
    }
    mg_mgr_free(&mgr);
    return 0;
}