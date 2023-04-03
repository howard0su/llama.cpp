#include <cassert>
#include <cstring>
#include <fstream>
#include <string>
#include <iterator>
#include <algorithm>
#include <thread>
#include <iostream>
#include <sstream>

#include "common.h"
#include "mongoose.h"
#include "llama.h"

#include "thread.h"

struct worker_params
{
    std::string host = "locahost";
    int port = 21002;
    std::string worker_address = "http://localhost:21002";
    std::string controller_address = "http://localhost:21001";

    int limit_model_concurrency = 5;
    int stream_interval = 2;
};

#define MAX_THREAD_COUNT 5
struct thread_params
{
    struct mg_connection *c;
    std::string model;
    std::string prompt;
    std::string stop;
    int n_token;
    float n_temp;
    pthread_t thread;
};

struct thread_params ThreadParams[MAX_THREAD_COUNT];

#define REGISTER_API "/register_worker"
#define HEARTBEAT_API "/receive_heart_beat"

gpt_params model_params;
worker_params params;
llama_context *ctx;

// todo
int queue_length = 1;

static std::string get_status()
{
    char json_data[1024];
    snprintf(json_data, sizeof(json_data), "{ \"model_names\":[\"%s\"],\"speed\":1,\"queue_length\":%d}",
             "alpaca", queue_length);

    return json_data;
}

static void http_post(struct mg_connection *c, int ev, void *ev_data, void *fn_data)
{
    if (ev == MG_EV_CONNECT)
    {
        struct mg_str host = mg_url_host(params.controller_address.c_str());

        char data[1024];
        if (strcmp((const char *)fn_data, REGISTER_API) == 0)
        {
            snprintf(data, sizeof(data), "{\"worker_name\":\"%s\",\"check_heart_beat\":%s,\"worker_status\": %s}",
                     params.worker_address.c_str(), "true", get_status().c_str());
        }
        else if (strcmp((const char *)fn_data, HEARTBEAT_API) == 0)
        {
            snprintf(data, sizeof(data), "{\"worker_name\":\"%s\",\"queue_length\": %d}",
                     params.worker_address.c_str(), queue_length);
        }
        // Send request
        mg_printf(c,
                  "POST %s HTTP/1.1\r\n"
                  "Host: %.*s\r\n"
                  "Content-Type: application/json\r\n"
                  "Content-Length: %d\r\n"
                  "\r\n",
                  fn_data, (int)host.len, host.ptr, strlen(data));
        mg_send(c, data, strlen(data));
    }

    if (ev == MG_EV_HTTP_MSG)
    {
        struct mg_http_message *hm = (struct mg_http_message *)ev_data;
    }

    if (ev == MG_EV_ERROR)
    {
        printf("Error: %s", (char *)ev_data);
    }
}

void worker_print_usage(int /*argc*/, char **argv, const worker_params &params)
{
    fprintf(stderr, "usage: %s [options]\n", argv[0]);
    fprintf(stderr, "\n");
    fprintf(stderr, "options:\n");
    fprintf(stderr, "  -h, --help            show this help message and exit\n");
    fprintf(stderr, "  --host                The listen host, default is %s\n", params.host.c_str());
    fprintf(stderr, "  --port            the listen port, default is %d\n", params.port);
    fprintf(stderr, "  --concurrency     the number of concurrent session, default is %d\n", params.limit_model_concurrency);
    fprintf(stderr, "  --interval        the number of token to send every time, default is %d\n", params.stream_interval);
}

bool worker_params_parse(int argc, char **argv, worker_params &params, gpt_params &gpt_params)
{
    bool invalid_param = false;
    std::string arg;
    for (int i = 1; i < argc; i++)
    {
        arg = argv[i];

        if (arg == "--host")
        {
            if (++i >= argc)
            {
                invalid_param = true;
                break;
            }
            params.host = argv[i];
        }
        else if (arg == "-h" || arg == "--help")
        {
            worker_print_usage(argc, argv, params);
            gpt_print_usage(0, argv, gpt_params);
            exit(0);
        }
        else if (arg == "--port")
        {
            if (++i >= argc)
            {
                invalid_param = true;
                break;
            }
            params.port = std::atoi(argv[i]);
        }
        else
        {
            fprintf(stderr, "error: unknown argument: %s\n", arg.c_str());
            worker_print_usage(argc, argv, params);
            exit(1);
        }
    }
    if (invalid_param)
    {
        fprintf(stderr, "error: invalid parameter for argument: %s\n", arg.c_str());
        worker_print_usage(argc, argv, params);
        gpt_print_usage(0, argv, gpt_params);
        exit(1);
    }

    return true;
}

static std::string stream_response(const char *text, int code)
{
    char buffer[1024];
    snprintf(buffer, sizeof(buffer), "{\"text\": \"%s\", \"error_code\": %d}", text, code);

    return buffer;
}

static thread_ret_t eval_thread(void *context)
{
    struct thread_params *local_params = (struct thread_params *)context;
    auto c = local_params->c;

    // Add a space in front of the first character to match OG llama tokenizer behavior
    local_params->prompt.insert(0, 1, ' ');

    // tokenize the prompt
    printf("Process: %s\n", local_params->prompt.c_str());
    auto embd_inp = ::llama_tokenize(ctx, local_params->prompt, true);

    std::vector<llama_token> embd;

    int n_past = 0;
    int n_remain = local_params->n_token;
    int n_consumed = 0;
    std::vector<llama_token> emb_output;
    std::stringstream final_output;

    const int n_ctx = llama_n_ctx(ctx);
    std::vector<llama_token> last_n_tokens(n_ctx);
    std::fill(last_n_tokens.begin(), last_n_tokens.end(), 0);
    llama_token stop_token = llama_token_eos();

    if (!local_params->stop.empty())
    {
        auto stop_tokens = ::llama_tokenize(ctx, local_params->prompt, true);
        stop_token = stop_tokens[0];
        printf("Stop Token: %d\n", stop_token);
    }

    for (int i = 0; i < n_remain; i++)
    {
        if (emb_output.size() > 0)
        {
            // infinite text generation via context swapping
            // if we run out of context:
            // - take the n_keep first tokens from the original prompt (via n_past)
            // - take half of the last (n_ctx - n_keep) tokens and recompute the logits in a batch
            if (n_past + (int)emb_output.size() > n_ctx)
            {
                const int n_left = n_past - model_params.n_keep;

                n_past = model_params.n_keep;

                // insert n_left/2 tokens at the start of embd from last_n_tokens
                emb_output.insert(emb_output.begin(), last_n_tokens.begin() + n_ctx - n_left / 2 - emb_output.size(), last_n_tokens.end() - emb_output.size());
            }

            if (llama_eval(ctx, emb_output.data(), emb_output.size(), n_past, 16))
            { // hack
                fprintf(stderr, "%s : failed to eval\n", __func__);
                return 1;
            }
        }

        n_past += emb_output.size();
        emb_output.clear();

        if ((int)embd_inp.size() <= n_consumed)
        {
            // out of user input, sample next token
            const int32_t top_k = model_params.top_k;
            const float top_p = model_params.top_p;
            const float temp = local_params->n_temp;
            const float repeat_penalty = model_params.repeat_penalty;

            llama_token id = 0;

            {
                auto logits = llama_get_logits(ctx);

                if (model_params.ignore_eos)
                {
                    logits[llama_token_eos()] = 0;
                }

                id = llama_sample_top_p_top_k(ctx,
                                              last_n_tokens.data() + n_ctx - model_params.repeat_last_n,
                                              model_params.repeat_last_n, top_k, top_p, temp, repeat_penalty);

                last_n_tokens.erase(last_n_tokens.begin());
                last_n_tokens.push_back(id);
            }

            // add it to the context
            emb_output.push_back(id);
            final_output << llama_token_to_str(ctx, id);

            // decrement remaining sampling budget
            --n_remain;
        }
        else
        {
            // some user input remains from prompt or interaction, forward it to processing
            while ((int)embd_inp.size() > n_consumed)
            {
                emb_output.push_back(embd_inp[n_consumed]);
                final_output << llama_token_to_str(ctx, embd_inp[n_consumed]);

                last_n_tokens.erase(last_n_tokens.begin());
                last_n_tokens.push_back(embd_inp[n_consumed]);
                ++n_consumed;
                if ((int)emb_output.size() >= model_params.n_batch)
                {
                    break;
                }
            }
        }

        if (i % params.stream_interval == 0) {
            auto resp = stream_response(final_output.str().c_str(), 0);
            mg_send(c, resp.c_str(), resp.length() + 1); // 1 more byte as '\0'
            printf("Response: %s\n", resp.c_str());
        }

        if (emb_output.back() == llama_token_eos() || emb_output.back() == stop_token || n_remain == 0)
        {
            auto resp = stream_response(final_output.str().c_str(), 2);
            mg_send(c, resp.c_str(), resp.length() + 1); // 1 more byte as '\0'
            printf("Response: %s\n", resp.c_str());

            mg_close_conn(c);
            break;
        }

    }

    printf("Connection is closed\n");
    local_params->thread = 0;

    return 0;
}

static void server_callback(struct mg_connection *c, int ev, void *ev_data, void *fn_data)
{
    if (ev == MG_EV_HTTP_MSG)
    {
        struct mg_http_message *hm = (struct mg_http_message *)ev_data;
        if (mg_http_match_uri(hm, "/worker_get_status"))
        {
            mg_http_reply(c, 200, "", "%s", get_status().c_str()); // Send dynamic JSON response
        }
        else if (mg_http_match_uri(hm, "/worker_generate_stream"))
        {
            int pos = -1;
            // check if we how many concurrent session
            // TODO: need lock to avoid race
            for (int i = 0; i < MAX_THREAD_COUNT; i++)
            {
                if (ThreadParams[i].thread == 0)
                {
                    pos = i;
                    break;
                }
            }

            if (pos == -1)
            {
                auto resp = stream_response("Server is Busy", 1);
                mg_http_reply(c, 500, "", resp.c_str(), resp.length());
                return;
            }

            // create thread to serve this request, pass connection down
            ThreadParams[pos].c = c;
            ThreadParams[pos].model = mg_json_get_str(hm->body, "$.model");
            ThreadParams[pos].prompt = mg_json_get_str(hm->body, "$.prompt");
            ThreadParams[pos].stop = mg_json_get_str(hm->body, "$.stop");
            ThreadParams[pos].n_token = mg_json_get_long(hm->body, "$.max_new_tokens", 512);
            double temp;
            mg_json_get_num(hm->body, "$.temperature", &temp);
            ThreadParams[pos].n_temp = (float)temp;

            mg_printf(c, "HTTP/1.1 200 OK\r\n"
                         "Content-Type: application/json\r\n"
                         "\r\n");

            pthread_create(&ThreadParams[pos].thread, NULL, eval_thread, &ThreadParams[pos]);
        }
    }

    if (ev == MG_EV_CLOSE)
    {
        printf("Detect close\n");
    }
}

int main(int argc, char **argv)
{
    model_params.model = "C:\\GPT\\en-models\\7B\\ggml-alpaca-7b-q4.bin";

    if (worker_params_parse(argc, argv, params, model_params) == false)
    {
        return 1;
    }

    // initialize model, load
    model_params.seed = (int32_t)time(NULL);

    std::mt19937 rng(model_params.seed);

    {
        auto lparams = llama_context_default_params();

        lparams.n_ctx = model_params.n_ctx;
        lparams.n_parts = model_params.n_parts;
        lparams.seed = model_params.seed;
        lparams.f16_kv = model_params.memory_f16;
        lparams.use_mlock = model_params.use_mlock;
        ctx = llama_init_from_file(model_params.model.c_str(), lparams);
        if (ctx == NULL)
        {
            fprintf(stderr, "%s: error: failed to load model '%s'\n", __func__, model_params.model.c_str());
            return 1;
        }
    }

    // initialize mongoose
    struct mg_mgr mgr;
    mg_mgr_init(&mgr);

    mg_http_connect(&mgr, params.controller_address.c_str(), http_post, REGISTER_API);

    mg_http_listen(&mgr, params.worker_address.c_str(), server_callback, NULL);

    int i = 0;
    for (;;)
    {
        mg_mgr_poll(&mgr, 1000);
        if (i++ == 30)
        {
            mg_http_connect(&mgr, params.controller_address.c_str(), http_post, HEARTBEAT_API);
        }
    }
    mg_mgr_free(&mgr);
    return 0;
}