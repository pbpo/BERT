#pragma once

#include <string>
#include <cstddef> // for size_t

struct BertConfig {
    // Model Dimensions
    int vocab_size = 30522;
    int hidden_size = 768; // Also known as n_embd
    int num_hidden_layers = 12;
    int num_attention_heads = 12;
    int intermediate_size = 3072; // Size of the FFN intermediate layer (4 * hidden_size)
    int max_position_embeddings = 512;
    int type_vocab_size = 2; // For segment token embeddings (e.g., sentence A vs B)

    // Activation
    std::string hidden_act = "gelu";

    // Regularization
    float hidden_dropout_prob = 0.1f;
    float attention_probs_dropout_prob = 0.1f;

    // Initialization
    float initializer_range = 0.02f;

    // Normalization
    float layer_norm_eps = 1e-12;
};