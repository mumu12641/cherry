/*
 * This code includes logic from Andrej Karpathy's llama2.c (run.c).
 * Source: https://github.com/karpathy/llama2.c
 * Copyright (c) 2023 Andrej Karpathy
 * Licensed under the MIT License.
 */

#include <ctype.h>
#include <fcntl.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
extern "C" {

// ----------------------------------------------------------------------------
// The Byte Pair Encoding (BPE) Tokenizer that translates strings <-> tokens
typedef struct
{
    char* str;
    int   id;
} TokenIndex;

typedef struct
{
    char**        vocab;
    float*        vocab_scores;
    TokenIndex*   sorted_vocab;
    int           vocab_size;
    unsigned int  max_token_length;
    unsigned char byte_pieces[512];   // stores all single-byte strings
} Tokenizer;

Tokenizer tokenizer;

void build_tokenizer(int vocab_size, char* alloc, char* tokenizer_path)
{

    // i should have written the vocab_size into the tokenizer file... sigh
    tokenizer.vocab_size = vocab_size;
    // malloc space to hold the scores and the strings
    tokenizer.vocab        = (char**)malloc(vocab_size * sizeof(char*));
    tokenizer.vocab_scores = (float*)malloc(vocab_size * sizeof(float));
    tokenizer.sorted_vocab = NULL;   // initialized lazily
    for (int i = 0; i < 256; i++) {
        tokenizer.byte_pieces[i * 2]     = (unsigned char)i;
        tokenizer.byte_pieces[i * 2 + 1] = '\0';
    }
    // read in the file
    FILE* file = fopen(tokenizer_path, "rb");
    if (!file) {
        fprintf(stderr, "couldn't load %s\n", tokenizer_path);
        exit(EXIT_FAILURE);
    }
    if (fread(&tokenizer.max_token_length, sizeof(int), 1, file) != 1) {
        fprintf(stderr, "failed read\n");
        exit(EXIT_FAILURE);
    }
    int len;
    for (int i = 0; i < vocab_size; i++) {
        if (fread(tokenizer.vocab_scores + i, sizeof(float), 1, file) != 1) {
            fprintf(stderr, "failed read\n");
            exit(EXIT_FAILURE);
        }
        if (fread(&len, sizeof(int), 1, file) != 1) {
            fprintf(stderr, "failed read\n");
            exit(EXIT_FAILURE);
        }
        tokenizer.vocab[i] = (char*)malloc(len + 1);
        if (fread(tokenizer.vocab[i], len, 1, file) != 1) {
            fprintf(stderr, "failed read\n");
            exit(EXIT_FAILURE);
        }
        tokenizer.vocab[i][len] = '\0';   // add the string terminating token
    }
    fclose(file);
}

void free_tokenizer()
{
    for (int i = 0; i < tokenizer.vocab_size; i++) {
        free(tokenizer.vocab[i]);
    }
    free(tokenizer.vocab);
    free(tokenizer.vocab_scores);
    free(tokenizer.sorted_vocab);
}

void safe_printf(char* piece)
{
    // piece might be a raw byte token, and we only want to print printable chars or whitespace
    // because some of the other bytes can be various control codes, backspace, etc.
    if (piece == NULL) {
        return;
    }
    if (piece[0] == '\0') {
        return;
    }
    if (piece[1] == '\0') {
        unsigned char byte_val = piece[0];
        if (!(isprint(byte_val) || isspace(byte_val))) {
            return;   // bad byte, don't print it
        }
    }
    printf("%s", piece);
}

void decode(int prev_token, int token)
{
    char* piece = tokenizer.vocab[token];
    // following BOS (1) token, sentencepiece decoder strips any leading whitespace (see PR #89)
    if (prev_token == 1 && piece[0] == ' ') {
        piece++;
    }
    // careful, some tokens designate raw bytes, and look like e.g. '<0x01>'
    // parse this and convert and return the actual byte
    unsigned char byte_val;
    if (sscanf(piece, "<0x%02hhX>", &byte_val) == 1) {
        piece = (char*)tokenizer.byte_pieces + byte_val * 2;
    }
    safe_printf(piece);
    fflush(stdout);
}
}
