/**
 * @file export.c
 * @brief Utility to extract individual tensor weights from llama2.c binary model files.
 *
 * Target Data Source:
 *   $ wget https://huggingface.co/karpathy/tinyllamas/resolve/main/stories15M.bin
 */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>

#define MAKE_DIR(path) mkdir(path, 0755)

typedef struct
{
    int dim;
    int hidden_dim;
    int n_layers;
    int n_heads;
    int n_kv_heads;
    int vocab_size;
    int seq_len;
} Config;

void create_directory_if_not_exists(const char* dir_path)
{
    struct stat st = {0};
    if (stat(dir_path, &st) == -1) {
        if (MAKE_DIR(dir_path) == 0) {
            printf("📂 Creating output directory: %s\n", dir_path);
        }
        else {
            fprintf(stderr, "❌ Failed to create directory: %s\n", dir_path);
            exit(1);
        }
    }
    else {
        printf("📂 Using existing directory: %s\n", dir_path);
    }
}

void save_tensor(const char* output_dir, const char* name, const float* data, int* shape, int ndim)
{
    char filepath[512];
    snprintf(filepath, sizeof(filepath), "%s/%s.bin", output_dir, name);

    FILE* f = fopen(filepath, "wb");
    if (!f) {
        fprintf(stderr, "❌ Error: Cannot write to %s\n", filepath);
        return;
    }

    fwrite(&ndim, sizeof(int), 1, f);

    fwrite(shape, sizeof(int), ndim, f);

    size_t num_elements = 1;
    for (int i = 0; i < ndim; i++) {
        num_elements *= shape[i];
    }
    fwrite(data, sizeof(float), num_elements, f);
    fclose(f);

    printf("💾 Saved: %-25s Shape: [", name);
    for (int i = 0; i < ndim; i++) {
        printf("%d%s", shape[i], (i < ndim - 1) ? ", " : "");
    }
    printf("]\n");
}

void print_usage(const char* program_name)
{
    printf("Usage: %s <model_file> [output_dir]\n\n", program_name);
    printf("Example:\n");
    printf("  %s stories15M.bin my_weights\n", program_name);
    printf("  %s stories15M.bin  (defaults to 'weights')\n\n", program_name);
}

int main(int argc, char* argv[])
{
    if (argc < 2) {
        print_usage(argv[0]);
        return 1;
    }

    if (strcmp(argv[1], "-h") == 0 || strcmp(argv[1], "--help") == 0) {
        print_usage(argv[0]);
        return 0;
    }

    const char* model_path = argv[1];
    const char* output_dir = (argc >= 3) ? argv[2] : "weights";

    printf("📖 Reading Model: %s\n", model_path);

    FILE* file = fopen(model_path, "rb");
    if (!file) {
        fprintf(stderr, "❌ Cannot open model file: %s\n", model_path);
        return 1;
    }

    Config config;
    if (fread(&config, sizeof(Config), 1, file) != 1) {
        fprintf(stderr, "❌ Failed to read Config.\n");
        fclose(file);
        return 1;
    }

    int shared_weights = config.vocab_size > 0 ? 1 : 0;
    config.vocab_size  = abs(config.vocab_size);

    printf("⚙️  Model Configuration:\n");
    printf("   • dim:        %d\n", config.dim);
    printf("   • hidden_dim: %d\n", config.hidden_dim);
    printf("   • n_layers:   %d\n", config.n_layers);
    printf("   • n_heads:    %d\n", config.n_heads);
    printf("   • n_kv_heads: %d\n", config.n_kv_heads);
    printf(
        "   • vocab_size: %d %s\n", config.vocab_size, shared_weights ? "(Shared)" : "(Unshared)");
    printf("   • seq_len:    %d\n", config.seq_len);
    printf("----------------------------------------\n");

    fseek(file, 0, SEEK_END);
    long file_size = ftell(file);
    fseek(file, sizeof(Config), SEEK_SET);

    long weights_bytes = file_size - sizeof(Config);
    if (weights_bytes <= 0) {
        fprintf(stderr, "❌ File contains no weight data.\n");
        fclose(file);
        return 1;
    }

    float* weights_data = (float*)malloc(weights_bytes);
    if (!weights_data) {
        fprintf(stderr, "❌ Memory allocation failed (%ld bytes).\n", weights_bytes);
        fclose(file);
        return 1;
    }

    printf("🚀 Loading weights into memory...\n");
    if (fread(weights_data, 1, weights_bytes, file) != weights_bytes) {
        fprintf(stderr, "❌ Failed to read weight data.\n");
        free(weights_data);
        fclose(file);
        return 1;
    }
    fclose(file);

    create_directory_if_not_exists(output_dir);
    printf("🚀 Starting extraction...\n");

    float* ptr = weights_data;

    int head_size = config.dim / config.n_heads;

#define SAVE_AND_ADVANCE(name, ndim, ...)            \
    do {                                             \
        int s[] = {__VA_ARGS__};                     \
        save_tensor(output_dir, name, ptr, s, ndim); \
        size_t size = 1;                             \
        for (int i = 0; i < ndim; i++) size *= s[i]; \
        ptr += size;                                 \
    } while (0)

    SAVE_AND_ADVANCE("token_embeddings", 2, config.vocab_size, config.dim);
    SAVE_AND_ADVANCE("layers_rms_att_weight", 2, config.n_layers, config.dim);
    SAVE_AND_ADVANCE("layers_wq", 3, config.n_layers, config.dim, config.n_heads * head_size);
    SAVE_AND_ADVANCE("layers_wk", 3, config.n_layers, config.dim, config.n_kv_heads * head_size);
    SAVE_AND_ADVANCE("layers_wv", 3, config.n_layers, config.dim, config.n_kv_heads * head_size);
    SAVE_AND_ADVANCE("layers_wo", 3, config.n_layers, config.n_heads * head_size, config.dim);
    SAVE_AND_ADVANCE("layers_rms_ffn_weight", 2, config.n_layers, config.dim);
    SAVE_AND_ADVANCE("layers_w1", 3, config.n_layers, config.dim, config.hidden_dim);
    SAVE_AND_ADVANCE("layers_w2", 3, config.n_layers, config.hidden_dim, config.dim);
    SAVE_AND_ADVANCE("layers_w3", 3, config.n_layers, config.dim, config.hidden_dim);
    SAVE_AND_ADVANCE("final_rms_norm", 1, config.dim);

    if (shared_weights) {
        printf("ℹ️  Shared Weights detected: Copying token_embeddings to output_wcls...\n");
        int s[] = {config.vocab_size, config.dim};
        save_tensor(output_dir, "output_wcls", weights_data, s, 2);
    }
    else {
        SAVE_AND_ADVANCE("output_wcls", 2, config.vocab_size, config.dim);
    }

    free(weights_data);
    printf("\n✨ Done! All weights have been extracted to '%s'.\n", output_dir);

    return 0;
}
