#include "engine.h"
#include "mongoose.h"
#include "cJSON.h"

void CompletionRequestBase::Prepare(mg_http_message *hm)
{
    max_tokens = mg_json_get_long(hm->body, "$.max_tokens", 16);
    temperature = 1.0;
    mg_json_get_num(hm->body, "$.temperature", &temperature);
    top_p = 1.0;
    mg_json_get_num(hm->body, "$.top_p", &top_p);
    n = mg_json_get_long(hm->body, "$.n", 1);
    stream = false;
    mg_json_get_bool(hm->body, "$.stream", &stream);
    // logprobs = mg_json_get_long(hm->body, "$.logprobs", 0);
    // echo = false;
    mg_json_get_bool(hm->body, "$.echo", &echo);
    presence_penalty = 0.0;
    mg_json_get_num(hm->body, "$.presence_penalty", &presence_penalty);
    frequency_penalty = 0.0;
    mg_json_get_num(hm->body, "$.frequency_penalty", &frequency_penalty);
    int best_of = mg_json_get_long(hm->body, "$.best_of", 1);

    Request::Prepare(hm);
}

void CompleteionRequest::Prepare(mg_http_message *hm) {
    CompletionRequestBase::Prepare(hm);
}

void ChatCompleteionRequest::Prepare(mg_http_message *hm) {
    CompletionRequestBase::Prepare(hm);
}
