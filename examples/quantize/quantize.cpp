#include "ggml.h"
#include "llama.h"
#include "rwkv.h"

#include <cstdio>
#include <string>

int usage(const char* cmd)
{
    fprintf(stderr, "usage: %s model-f32.bin model-quant.bin type model\n", cmd);
    fprintf(stderr, "  type = 2 - q4_0\n");
    fprintf(stderr, "  type = 3 - q4_1\n");
    fprintf(stderr, "  model = llama - llama\n");
    fprintf(stderr, "  model = rwkv - rwkv\n");
    return 1;
}
// usage:
//  ./llama-quantize models/llama/ggml-model.bin models/llama/ggml-model-quant.bin type
//
int main(int argc, char ** argv) {
    ggml_time_init();

    if (argc != 5) {
        return usage(argv[0]);
    }

    // needed to initialize f16 tables
    {
        struct ggml_init_params params = { 0, NULL, false };
        struct ggml_context * ctx = ggml_init(params);
        ggml_free(ctx);
    }

    const std::string fname_inp = argv[1];
    const std::string fname_out = argv[2];
    const std::string model = argv[4];

    const int itype = atoi(argv[3]);

    // validate model
    if (model != "llama" && model != "rwkv")
        return usage(argv[0]);

    // validate itype
    if (itype != 2 && itype != 3)
        return usage(argv[0]);

    const int64_t t_main_start_us = ggml_time_us();

    int64_t t_quantize_us = 0;

    // load the model
    {
        const int64_t t_start_us = ggml_time_us();
        bool ret;
        
        if (model == "llama")
            ret = llama_model_quantize(fname_inp.c_str(), fname_out.c_str(), itype);
        else if (model == "rwkv")
            ret = rwkv_model_quantize(fname_inp.c_str(), fname_out.c_str(), itype);

        if (ret) {
            fprintf(stderr, "%s: failed to quantize model from '%s'\n", __func__, fname_inp.c_str());
            return 1;
        }

        t_quantize_us = ggml_time_us() - t_start_us;
    }

    // report timing
    {
        const int64_t t_main_end_us = ggml_time_us();

        printf("\n");
        printf("%s: quantize time = %8.2f ms\n", __func__, t_quantize_us/1000.0);
        printf("%s:    total time = %8.2f ms\n", __func__, (t_main_end_us - t_main_start_us)/1000.0);
    }

    return 0;
}
