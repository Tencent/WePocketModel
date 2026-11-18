// Stable, small CPU ABI over the pinned llama.cpp C API.
#include "llama.h"
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
thread_local std::string error;
void log_cpu(ggml_log_level level, const char * message, void *) {
    if (level == GGML_LOG_LEVEL_WARN || level == GGML_LOG_LEVEL_ERROR)
        std::fputs(message, stderr);
}
struct engine {
    llama_model * model = nullptr;
    llama_context * ctx = nullptr;
    int width = 0;
    int vocabulary = 0;
    int context = 0;
    ~engine() {
        if (ctx) llama_free(ctx);
        if (model) llama_model_free(model);
    }
};
}

extern "C" {
const char * wepm_error() { return error.c_str(); }
int wepm_abi_version() { return 1; }

void * wepm_load(const char * path, int context, int threads) {
    try {
        if (context < 128 || threads < 1) throw std::runtime_error("invalid CPU context or threads");
        llama_log_set(std::getenv("WEPM_GGUF_VERBOSE") ? nullptr : log_cpu, nullptr);
        llama_backend_init();
        auto value = std::make_unique<engine>();
        auto mp = llama_model_default_params();
        mp.n_gpu_layers = 0;
        mp.load_mtp = false;
        value->model = llama_model_load_from_file(path, mp);
        if (!value->model) throw std::runtime_error("failed to load GGUF language weights");
        value->width = llama_model_n_embd(value->model);
        value->vocabulary = llama_vocab_n_tokens(llama_model_get_vocab(value->model));
        value->context = context;
        auto cp = llama_context_default_params();
        cp.n_ctx = context;
        cp.n_batch = std::min(context, 256);
        cp.n_ubatch = cp.n_batch;
        cp.n_seq_max = 1;
        cp.n_threads = cp.n_threads_batch = threads;
        cp.pooling_type = LLAMA_POOLING_TYPE_NONE;
        cp.attention_type = LLAMA_ATTENTION_TYPE_CAUSAL;
        cp.offload_kqv = false;
        cp.op_offload = false;
        cp.no_perf = true;
        value->ctx = llama_init_from_model(value->model, cp);
        if (!value->ctx) throw std::runtime_error("failed to create CPU GGUF context");
        return value.release();
    } catch (const std::exception & ex) { error = ex.what(); return nullptr; }
}

void wepm_free(void * handle) { delete static_cast<engine *>(handle); }
int wepm_width(void * handle) { return static_cast<engine *>(handle)->width; }
int wepm_vocabulary(void * handle) { return static_cast<engine *>(handle)->vocabulary; }

// Inputs use section-major positions [sequence, height, width, temporal].
// Embedding input is used during both prefill and decode so multimodal RoPE
// coordinates remain independent of the physical cache position.
int wepm_eval(void * handle, const float * embeddings, const int32_t * positions,
              int count, int reset, int read_hidden, float * output) {
    try {
        auto * value = static_cast<engine *>(handle);
        if (!value || !embeddings || !positions || !output || count < 1)
            throw std::runtime_error("invalid GGUF evaluation input");
        if (positions[count-1] >= value->context)
            throw std::runtime_error("GGUF context budget exceeded");
        if (reset) llama_memory_clear(llama_get_memory(value->ctx), true);
        llama_set_causal_attn(value->ctx, true);
        llama_set_embeddings(value->ctx, read_hidden != 0);
        for (int offset = 0; offset < count; offset += 256) {
            const int size = std::min(256, count-offset);
            std::vector<llama_pos> pos(4*size);
            std::vector<int32_t> seq_counts(size, 1);
            std::vector<llama_seq_id> seq_values(size, 0);
            std::vector<llama_seq_id *> seqs(size);
            std::vector<int8_t> outputs(size, read_hidden ? 1 : 0);
            outputs.at(size-1) = 1;
            for (int i = 0; i < size; ++i) {
                seqs[i] = &seq_values[i];
                for (int axis = 0; axis < 4; ++axis)
                    pos[axis*size+i] = positions[axis*count+offset+i];
            }
            llama_batch batch = {};
            batch.n_tokens = size;
            batch.embd = const_cast<float *>(embeddings + offset*value->width);
            batch.pos = pos.data();
            batch.n_seq_id = seq_counts.data();
            batch.seq_id = seqs.data();
            batch.logits = outputs.data();
            if (llama_decode(value->ctx, batch) != 0)
                throw std::runtime_error("llama_decode failed");
        }
        auto * data = read_hidden ? llama_get_embeddings_ith(value->ctx, -1)
                                 : llama_get_logits_ith(value->ctx, -1);
        if (!data) throw std::runtime_error("GGUF model did not return the requested output");
        std::memcpy(output, data, sizeof(float)*(read_hidden ? value->width : value->vocabulary));
        return 0;
    } catch (const std::exception & ex) { error = ex.what(); return -1; }
}
}
