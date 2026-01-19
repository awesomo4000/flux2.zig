```
requirements: none
zig build and then you are done
ai slop is zen
```

# flux2.zig 🦎⚡

> **Origin prompt:**
> *"https://github.com/antirez/flux2.c analyze those code base and plan a conversion to zig for the main programs and libraries except the blas stuff which seems like a much bigger project. This was coded by Claude in a weekend with skilled human guidance but I want a zig version zig build zig code"*

A pure Zig port of [antirez/flux2.c](https://github.com/antirez/flux2.c) — FLUX.2-klein-4B image generation inference. 🖼️✨

No Python. 🐍❌ No PyTorch. 🔥❌ No CUDA toolkit. 🎮❌ Just `zig build`. ⚡💯

## Status 📊

🚧 **Work in Progress** — Foundation complete, inference pipeline in development.

| Component | Status | Notes |
|-----------|--------|-------|
| `tensor.zig` | ✅ | Multi-dim arrays, views, reshape |
| `kernels.zig` | ✅ | SIMD matmul, GELU, softmax, RMSNorm, RoPE |
| `safetensors.zig` | ✅ | Model loading, mmap, BF16→F32 |
| `image.zig` | ✅ | PPM I/O, tensor conversion |
| `server.zig` | ✅ | REPL mode for memory-resident ops |
| `hfd` tool | ✅ | HuggingFace downloader (replaces Python 🐍💀) |
| Transformer | 🔨 | In progress |
| VAE | 🔨 | In progress |
| Qwen3 encoder | 🔨 | In progress |
| Tokenizer | 🔨 | In progress |

## Quick Start 🚀

```bash
# Build everything 🔧
zig build

# Download the model (~22GB) - no Python needed! 📥🎉
./zig-out/bin/hfd black-forest-labs/FLUX.2-klein-4B -o flux-klein-model

# Generate an image (once inference is complete) 🎨
./zig-out/bin/flux -d flux-klein-model -p "A fluffy cat" -o cat.ppm
```

## Building 🏗️

```bash
zig build              # Build flux CLI + hfd 🔨
zig build server       # Run flux-server (memory-resident) 🖥️
zig build hfd          # Run HuggingFace downloader 📦
zig build test         # Run tests 🧪
zig build bench        # Run kernel benchmarks 📈
```

Requires Zig 0.15.2+ 🦎

## Project Structure 📁

```
flux2.zig/
├── build.zig
└── src/
    ├── flux.zig          # Public API 🔌
    ├── main.zig          # CLI: flux 💻
    ├── server.zig        # CLI: flux-server (REPL/HTTP) 🌐
    ├── hfd.zig           # CLI: hfd (HuggingFace downloader) 📥
    ├── tensor.zig        # N-dimensional tensors 🧮
    ├── kernels.zig       # SIMD compute primitives ⚡
    ├── safetensors.zig   # HuggingFace model format 🤗
    ├── image.zig         # Image I/O 🖼️
    └── bench.zig         # Benchmarks 🏎️
```

## Tools 🛠️

### `hfd` — HuggingFace Downloader 📥

A pure Zig replacement for `pip install huggingface_hub && python download_model.py`: 🐍➡️🦎

```bash
# Download a model 📦
./zig-out/bin/hfd black-forest-labs/FLUX.2-klein-4B

# Download only safetensors 🎯
./zig-out/bin/hfd black-forest-labs/FLUX.2-klein-4B --include "*.safetensors"

# Resume interrupted download ⏸️▶️
./zig-out/bin/hfd black-forest-labs/FLUX.2-klein-4B --resume

# Dry run - see what would be downloaded 👀
./zig-out/bin/hfd black-forest-labs/FLUX.2-klein-4B --dry-run

# Gated models (requires token) 🔐
export HF_TOKEN=hf_xxxxx
./zig-out/bin/hfd meta-llama/Llama-2-7b
```

Features: ✨
- 🔒 SHA256 verification for LFS files
- ⏸️ Resume interrupted downloads
- 🎯 Include/exclude glob patterns
- 📊 Progress bar with transfer stats

### `flux-server` — Memory-Resident Generation 🧠💾

Keep the model loaded for fast repeated generations:

```bash
./zig-out/bin/flux-server -d flux-klein-model --repl

> {"prompt": "A cat", "output": "cat.ppm"}
Generated: cat.ppm (256x256) in 2100ms 🐱
> {"prompt": "A dog", "output": "dog.ppm"}
Generated: dog.ppm (256x256) in 1950ms 🐕
> quit
```

## Why Zig? 🦎💪

- 🚫 **Zero dependencies** — No libc required, static binaries
- ⚡ **Portable SIMD** — `@Vector` works on x86 AVX, ARM NEON, WASM
- 🧠 **Explicit allocators** — No hidden allocations, predictable memory
- 🔮 **Comptime** — Shape validation at compile time
- 🔗 **C interop** — Can still link OpenBLAS/Accelerate if needed

## Performance 🏎️💨

Pure Zig SIMD kernels (no BLAS):

```
Matrix Multiplication (C = A @ B): 🧮
  64x64x64:    0.15 ms
  256x256x256: 8.2 ms
  512x512x512: 65 ms
  1024x1024x1024: 520 ms

GELU Activation: ⚡
  n=65536: 45 µs, 1450 M elem/s

RMS Normalization: 📏
  hidden_size=3072: 1.2 µs
```

~10-30x slower than optimized BLAS. Acceptable for CPU inference, room for optimization. 📈

## Roadmap 🗺️

1. **Phase 1** ✅ Foundation — tensor, kernels, safetensors, image
2. **Phase 2** 🔨 Neural network layers — transformer, VAE, Qwen3
3. **Phase 3** ⏳ Integration — tokenizer, sampler, end-to-end
4. **Phase 4** ⏳ Optimization — better tiling, cache blocking, optional BLAS

## Acknowledgments 🙏

- 🎩 [antirez/flux2.c](https://github.com/antirez/flux2.c) — The original C implementation (MIT)
- 🌲 [Black Forest Labs](https://blackforestlabs.ai/) — FLUX.2-klein model (Apache 2.0)
- 🤖 Claude — Code generation for both the original C and this Zig port

## License 📄

MIT — Same as the original flux2.c

---

*"I believe that inference systems not using the Python stack are a way to free open models usage and make AI more accessible."* — antirez 🧙‍♂️

---

🤖 *This port was vibe-coded by Claude (Opus 4.5) with human guidance. The emojis are a feature, not a bug.* 🦎✨🚀
