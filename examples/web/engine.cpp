#include <cassert>
#include <cstring>
#include <fstream>
#include <string>
#include <iterator>
#include <algorithm>
#include <thread>
#include <iostream>
#include <sstream>

#include "llama.h"
#include "common.h"
#include "thread.h"
#include "engine.h"

static llama_context *ctx;
static gpt_params params;
#if 0
static thread_ret_t eval_thread(void *context)
{
    struct request *req = (struct request *)context;

    // Add a space in front of the first character to match OG llama tokenizer behavior
    req->prompt.insert(0, 1, ' ');

    // tokenize the prompt
    printf("Process: %s\n", req->prompt.c_str());
    auto embd_inp = ::llama_tokenize(ctx, req->prompt, true);

    std::vector<llama_token> embd;

    int n_past = 0;
    int n_remain = req->n_token;
    int n_consumed = 0;
    std::vector<llama_token> emb_output;
    std::stringstream final_output;

    const int n_ctx = llama_n_ctx(ctx);
    std::vector<llama_token> last_n_tokens(n_ctx);
    std::fill(last_n_tokens.begin(), last_n_tokens.end(), 0);

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
                const int n_left = n_past - params.n_keep;

                n_past = params.n_keep;

                // insert n_left/2 tokens at the start of embd from last_n_tokens
                emb_output.insert(emb_output.begin(), last_n_tokens.begin() + n_ctx - n_left / 2 - emb_output.size(), last_n_tokens.end() - emb_output.size());
            }

            if (llama_eval(ctx, emb_output.data(), emb_output.size(), n_past, 16))
            { // hack
                fprintf(stderr, "%s : failed to eval\n", __func__);
                return NULL;
            }
        }

        n_past += emb_output.size();
        emb_output.clear();

        if ((int)embd_inp.size() <= n_consumed)
        {
            // out of user input, sample next token
            const int32_t top_k = params.top_k;
            const float top_p = params.top_p;
            const float temp = req->n_temp;
            const float repeat_penalty = params.repeat_penalty;

            llama_token id = 0;

            {
                auto logits = llama_get_logits(ctx);

                if (params.ignore_eos)
                {
                    logits[llama_token_eos()] = 0;
                }

                id = llama_sample_top_p_top_k(ctx,
                                              last_n_tokens.data() + n_ctx - params.repeat_last_n,
                                              params.repeat_last_n, top_k, top_p, temp, repeat_penalty);

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
                if ((int)emb_output.size() >= params.n_batch)
                {
                    break;
                }
            }
        }

        if (i % req->stream_interval == 0) {
            // auto resp = stream_response(final_output.str().c_str(), 0);
            // mg_send(c, resp.c_str(), resp.length() + 1); // 1 more byte as '\0'
            // printf("Response: %s\n", resp.c_str());
        }

        if (emb_output.back() == llama_token_eos() || n_remain == 0)
        {
            // auto resp = stream_response(final_output.str().c_str(), 2);
            // mg_send(c, resp.c_str(), resp.length() + 1); // 1 more byte as '\0'
            // printf("Response: %s\n", resp.c_str());

            break;
        }

    }

    printf("Connection is closed\n");

    return NULL;
}
#endif

int engine_init(int argc, char *argv[])
{
    params.model = "models/llama-7B/ggml-model.bin";

    if (gpt_params_parse(argc, argv, params) == false)
    {
        return 1;
    }

    // initialize model, load
    params.seed = (int32_t)time(NULL);

    std::mt19937 rng(params.seed);

    {
        auto lparams = llama_context_default_params();

        lparams.n_ctx = params.n_ctx;
        lparams.n_parts = params.n_parts;
        lparams.seed = params.seed;
        lparams.f16_kv = params.memory_f16;
        lparams.use_mlock = params.use_mlock;
        ctx = llama_init_from_file(params.model.c_str(), lparams);
        if (ctx == NULL)
        {
            fprintf(stderr, "%s: error: failed to load model '%s'\n", __func__, params.model.c_str());
            return 1;
        }
    }

    return 0;
}

void Request::Prepare(mg_http_message * hm) {
    // fetch the paramters
    this->model = mg_json_get_str(hm->body, "$.model");
    if (model.empty())
    {
        throw "Required field model is missing";
    }

    this->user = mg_json_get_str(hm->body, "$.user");
}