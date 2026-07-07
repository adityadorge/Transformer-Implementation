# Transformer Architecture from Scratch

A from-scratch implementation of the Transformer architecture based on the **"Attention Is All You Need"** research paper. This project is built as a learning exercise to understand how Transformers work internally by implementing each component without relying on high-level libraries.

## Overview

The goal of this project is to reproduce the core ideas introduced in the original Transformer paper while gaining a deeper understanding of modern sequence modeling architectures.

The implementation focuses on building the architecture step by step, including:

- Scaled Dot-Product Attention
- Multi-Head Attention
- Positional Encoding
- Layer Normalization
- Residual Connections
- Position-wise Feed Forward Networks
- Encoder Stack
- Decoder Stack
- Masking (Padding & Look-Ahead)
- Complete Transformer Model
## Learning Objectives

This project aims to understand:

- Why self-attention works
- How multi-head attention improves representation learning
- The role of positional encoding
- Why residual connections and layer normalization are important
- How encoder-decoder Transformers process sequences
- The implementation details behind the original Transformer architecture

## Reference

**Attention Is All You Need**

Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Łukasz Kaiser, and Illia Polosukhin.

Paper:
https://arxiv.org/abs/1706.03762

## Future Improvements

- Training on larger datasets
- Beam search decoding
- Mixed precision training
- Better experiment logging
- Support for custom datasets
- Unit tests for individual modules

## Acknowledgements

This implementation is inspired by the original **Attention Is All You Need** paper and is intended for educational purposes to better understand the Transformer architecture.
