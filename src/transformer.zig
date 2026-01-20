//! FLUX.2 Diffusion Transformer (DiT)
//!
//! Architecture (FLUX.2-klein-4B):
//! - 5 double-stream blocks (MM-DiT: separate image/text, joint attention)
//! - 20 single-stream blocks (parallel DiT: fused QKV+FFN)
//! - 24 attention heads, 128 dim per head (3072 hidden)
//! - SwiGLU activation, RoPE embeddings, AdaLN-Zero modulation

const std = @import("std");
const Allocator = std.mem.Allocator;
const kernels = @import("kernels.zig");
const SafeTensors = @import("safetensors.zig").SafeTensors;

/// Transformer configuration for FLUX.2-klein-4B
pub const Config = struct {
    hidden_size: usize = 3072,
    num_heads: usize = 24,
    head_dim: usize = 128,
    mlp_hidden: usize = 9216, // hidden * 3
    num_double_layers: usize = 5,
    num_single_layers: usize = 20,
    text_dim: usize = 7680,
    latent_channels: usize = 128,
    rope_theta: f32 = 2000.0,
    axis_dim: usize = 32, // RoPE axes
    eps: f32 = 1e-6,
};

/// Double-stream block weights
const DoubleBlock = struct {
    // Image stream
    img_q_weight: []f32,
    img_k_weight: []f32,
    img_v_weight: []f32,
    img_norm_q_weight: []f32,
    img_norm_k_weight: []f32,
    img_proj_weight: []f32,
    img_mlp_gate_weight: []f32,
    img_mlp_up_weight: []f32,
    img_mlp_down_weight: []f32,

    // Text stream
    txt_q_weight: []f32,
    txt_k_weight: []f32,
    txt_v_weight: []f32,
    txt_norm_q_weight: []f32,
    txt_norm_k_weight: []f32,
    txt_proj_weight: []f32,
    txt_mlp_gate_weight: []f32,
    txt_mlp_up_weight: []f32,
    txt_mlp_down_weight: []f32,
};

/// Single-stream block weights
const SingleBlock = struct {
    qkv_mlp_weight: []f32, // Fused [Q, K, V, gate, up] projection
    norm_q_weight: []f32,
    norm_k_weight: []f32,
    proj_mlp_weight: []f32, // Fused output projection
};

/// Time embedding weights
const TimeEmbed = struct {
    fc1_weight: []f32, // [hidden, 256]
    fc2_weight: []f32, // [hidden, hidden]
    sincos_dim: usize = 256,
};

/// FLUX.2 Transformer
pub const Transformer = struct {
    allocator: Allocator,
    arena: std.heap.ArenaAllocator, // Arena for all weight allocations
    config: Config,

    // Input projections
    img_in_weight: []f32, // [hidden, latent_channels]
    txt_in_weight: []f32, // [hidden, text_dim]

    // Time embedding
    time_embed: TimeEmbed,

    // Modulation weights (shared across blocks)
    adaln_double_img_weight: []f32, // [hidden*6, hidden]
    adaln_double_txt_weight: []f32, // [hidden*6, hidden]
    adaln_single_weight: []f32, // [hidden*3, hidden]

    // Blocks
    double_blocks: []DoubleBlock,
    single_blocks: []SingleBlock,

    // Final layer
    final_norm_weight: []f32, // [hidden*2, hidden] for scale/shift
    final_proj_weight: []f32, // [latent_channels, hidden]

    // Working memory (allocated from main allocator, not arena)
    work1: []f32,
    work2: []f32,
    work3: []f32,
    img_hidden: []f32,
    txt_hidden: []f32,

    const Self = @This();

    /// Load transformer from safetensors
    pub fn load(allocator: Allocator, path: []const u8) !*Self {
        var st = try SafeTensors.open(allocator, path);
        defer st.close();

        const self = try allocator.create(Self);
        errdefer allocator.destroy(self);

        self.allocator = allocator;
        self.arena = std.heap.ArenaAllocator.init(allocator);
        errdefer self.arena.deinit();

        const arena_alloc = self.arena.allocator();
        self.config = Config{};
        const cfg = self.config;

        // Input projections (use arena for all weights)
        self.img_in_weight = try loadTensor(&st, arena_alloc, "x_embedder.weight");
        self.txt_in_weight = try loadTensor(&st, arena_alloc, "context_embedder.weight");

        // Time embedding
        self.time_embed = TimeEmbed{
            .fc1_weight = try loadTensor(&st, arena_alloc, "time_guidance_embed.timestep_embedder.linear_1.weight"),
            .fc2_weight = try loadTensor(&st, arena_alloc, "time_guidance_embed.timestep_embedder.linear_2.weight"),
        };

        // Modulation weights
        self.adaln_double_img_weight = try loadTensor(&st, arena_alloc, "double_stream_modulation_img.linear.weight");
        self.adaln_double_txt_weight = try loadTensor(&st, arena_alloc, "double_stream_modulation_txt.linear.weight");
        self.adaln_single_weight = try loadTensor(&st, arena_alloc, "single_stream_modulation.linear.weight");

        // Double blocks
        self.double_blocks = try arena_alloc.alloc(DoubleBlock, cfg.num_double_layers);
        for (0..cfg.num_double_layers) |i| {
            self.double_blocks[i] = try loadDoubleBlock(&st, arena_alloc, i, cfg);
        }

        // Single blocks
        self.single_blocks = try arena_alloc.alloc(SingleBlock, cfg.num_single_layers);
        for (0..cfg.num_single_layers) |i| {
            self.single_blocks[i] = try loadSingleBlock(&st, arena_alloc, i);
        }

        // Final layer
        self.final_norm_weight = try loadTensor(&st, arena_alloc, "norm_out.linear.weight");
        self.final_proj_weight = try loadTensor(&st, arena_alloc, "proj_out.weight");

        // Allocate working memory (from main allocator, not arena - these may need resizing)
        const max_seq: usize = 18000; // Support up to ~1024x1024
        const hidden = cfg.hidden_size;
        self.work1 = try allocator.alloc(f32, max_seq * hidden * 4);
        self.work2 = try allocator.alloc(f32, max_seq * hidden * 4);
        self.work3 = try allocator.alloc(f32, max_seq * hidden * 4);
        self.img_hidden = try allocator.alloc(f32, max_seq * hidden);
        self.txt_hidden = try allocator.alloc(f32, max_seq * hidden);

        return self;
    }

    /// Forward pass through transformer
    /// img_latent: [batch, latent_channels, h, w] - patchified latent
    /// txt_emb: [batch, txt_seq, text_dim] - text embeddings from encoder
    /// timestep: diffusion timestep in [0, 1]
    /// Returns: [batch, latent_channels, h, w] - predicted noise
    pub fn forward(
        self: *Self,
        img_latent: []const f32,
        img_h: usize,
        img_w: usize,
        txt_emb: []const f32,
        txt_seq: usize,
        timestep: f32,
    ) ![]f32 {
        const cfg = self.config;
        const hidden = cfg.hidden_size;
        const img_seq = img_h * img_w;

        // Get timestep embedding
        var t_sincos: [256]f32 = undefined;
        getTimestepEmbedding(&t_sincos, timestep * 1000.0, 256, 10000.0);

        var t_emb: [3072]f32 = undefined;
        self.timeEmbedForward(&t_emb, &t_sincos);

        // Compute RoPE frequencies
        const img_rope_cos = try self.allocator.alloc(f32, img_seq * cfg.head_dim);
        defer self.allocator.free(img_rope_cos);
        const img_rope_sin = try self.allocator.alloc(f32, img_seq * cfg.head_dim);
        defer self.allocator.free(img_rope_sin);
        computeRope2D(img_rope_cos, img_rope_sin, img_h, img_w, cfg.axis_dim, cfg.rope_theta);

        const txt_rope_cos = try self.allocator.alloc(f32, txt_seq * cfg.head_dim);
        defer self.allocator.free(txt_rope_cos);
        const txt_rope_sin = try self.allocator.alloc(f32, txt_seq * cfg.head_dim);
        defer self.allocator.free(txt_rope_sin);
        computeRopeText(txt_rope_cos, txt_rope_sin, txt_seq, cfg.axis_dim, cfg.rope_theta);

        // Transpose image from NCHW to NLC
        const img_transposed = try self.allocator.alloc(f32, img_seq * cfg.latent_channels);
        defer self.allocator.free(img_transposed);
        for (0..img_seq) |pos| {
            for (0..cfg.latent_channels) |c| {
                img_transposed[pos * cfg.latent_channels + c] = img_latent[c * img_seq + pos];
            }
        }

        // Project inputs to hidden
        kernels.linearNoBias(self.img_hidden[0 .. img_seq * hidden], img_transposed, self.img_in_weight, img_seq, cfg.latent_channels, hidden);
        kernels.linearNoBias(self.txt_hidden[0 .. txt_seq * hidden], txt_emb, self.txt_in_weight, txt_seq, cfg.text_dim, hidden);

        // Double-stream blocks
        for (0..cfg.num_double_layers) |i| {
            try self.doubleBlockForward(
                &self.double_blocks[i],
                &t_emb,
                img_rope_cos,
                img_rope_sin,
                txt_rope_cos,
                txt_rope_sin,
                img_seq,
                txt_seq,
            );
        }

        // Concatenate for single-stream: [txt, img]
        const total_seq = img_seq + txt_seq;
        const concat_hidden = try self.allocator.alloc(f32, total_seq * hidden);
        defer self.allocator.free(concat_hidden);
        @memcpy(concat_hidden[0 .. txt_seq * hidden], self.txt_hidden[0 .. txt_seq * hidden]);
        @memcpy(concat_hidden[txt_seq * hidden ..][0 .. img_seq * hidden], self.img_hidden[0 .. img_seq * hidden]);

        // Single-stream blocks
        for (0..cfg.num_single_layers) |i| {
            try self.singleBlockForward(
                concat_hidden,
                &self.single_blocks[i],
                &t_emb,
                img_rope_cos,
                img_rope_sin,
                txt_rope_cos,
                txt_rope_sin,
                total_seq,
                txt_seq,
            );
        }

        // Extract image hidden (after text portion)
        @memcpy(self.img_hidden[0 .. img_seq * hidden], concat_hidden[txt_seq * hidden ..][0 .. img_seq * hidden]);

        // Final layer: AdaLN modulation -> project to latent channels
        var t_emb_silu: [3072]f32 = undefined;
        for (0..hidden) |i| {
            const x = t_emb[i];
            t_emb_silu[i] = x / (1.0 + @exp(-x));
        }

        var final_mod: [6144]f32 = undefined;
        kernels.linearNoBias(&final_mod, &t_emb_silu, self.final_norm_weight, 1, hidden, hidden * 2);

        const final_scale = final_mod[0..hidden];
        const final_shift = final_mod[hidden..][0..hidden];

        applyAdaLN(self.work1, self.img_hidden, final_shift, final_scale, img_seq, hidden, cfg.eps);

        const output_nlc = try self.allocator.alloc(f32, img_seq * cfg.latent_channels);
        defer self.allocator.free(output_nlc);
        kernels.linearNoBias(output_nlc, self.work1, self.final_proj_weight, img_seq, hidden, cfg.latent_channels);

        // Transpose output from NLC to NCHW
        const output = try self.allocator.alloc(f32, img_seq * cfg.latent_channels);
        for (0..img_seq) |pos| {
            for (0..cfg.latent_channels) |c| {
                output[c * img_seq + pos] = output_nlc[pos * cfg.latent_channels + c];
            }
        }

        return output;
    }

    fn timeEmbedForward(self: *Self, out: []f32, t_sincos: []const f32) void {
        const hidden = self.config.hidden_size;
        var h: [3072]f32 = undefined;

        // fc1: [256] -> [hidden]
        kernels.linearNoBias(&h, t_sincos, self.time_embed.fc1_weight, 1, 256, hidden);

        // SiLU
        kernels.silu(&h);

        // fc2: [hidden] -> [hidden]
        kernels.linearNoBias(out, &h, self.time_embed.fc2_weight, 1, hidden, hidden);
    }

    fn doubleBlockForward(
        self: *Self,
        block: *const DoubleBlock,
        t_emb: []const f32,
        img_rope_cos: []const f32,
        img_rope_sin: []const f32,
        txt_rope_cos: []const f32,
        txt_rope_sin: []const f32,
        img_seq: usize,
        txt_seq: usize,
    ) !void {
        const cfg = self.config;
        const hidden = cfg.hidden_size;
        const heads = cfg.num_heads;
        const head_dim = cfg.head_dim;
        const mlp_hidden = cfg.mlp_hidden;

        // Apply SiLU to t_emb for modulation
        var t_emb_silu: [3072]f32 = undefined;
        for (0..hidden) |i| {
            const x = t_emb[i];
            t_emb_silu[i] = x / (1.0 + @exp(-x));
        }

        // Compute modulation parameters (6 per stream)
        var img_mod: [18432]f32 = undefined; // hidden * 6
        var txt_mod: [18432]f32 = undefined;
        kernels.linearNoBias(&img_mod, &t_emb_silu, self.adaln_double_img_weight, 1, hidden, hidden * 6);
        kernels.linearNoBias(&txt_mod, &t_emb_silu, self.adaln_double_txt_weight, 1, hidden, hidden * 6);

        const img_shift1 = img_mod[0..hidden];
        const img_scale1 = img_mod[hidden..][0..hidden];
        const img_gate1 = img_mod[hidden * 2 ..][0..hidden];
        const img_shift2 = img_mod[hidden * 3 ..][0..hidden];
        const img_scale2 = img_mod[hidden * 4 ..][0..hidden];
        const img_gate2 = img_mod[hidden * 5 ..][0..hidden];

        const txt_shift1 = txt_mod[0..hidden];
        const txt_scale1 = txt_mod[hidden..][0..hidden];
        const txt_gate1 = txt_mod[hidden * 2 ..][0..hidden];
        const txt_shift2 = txt_mod[hidden * 3 ..][0..hidden];
        const txt_scale2 = txt_mod[hidden * 4 ..][0..hidden];
        const txt_gate2 = txt_mod[hidden * 5 ..][0..hidden];

        // Image stream: AdaLN -> QKV
        const img_norm = self.work1[0 .. img_seq * hidden];
        applyAdaLN(img_norm, self.img_hidden, img_shift1, img_scale1, img_seq, hidden, cfg.eps);

        const img_q = self.work2[0 .. img_seq * hidden];
        const img_k = self.work2[img_seq * hidden ..][0 .. img_seq * hidden];
        const img_v = self.work2[img_seq * hidden * 2 ..][0 .. img_seq * hidden];

        kernels.linearNoBias(img_q, img_norm, block.img_q_weight, img_seq, hidden, hidden);
        kernels.linearNoBias(img_k, img_norm, block.img_k_weight, img_seq, hidden, hidden);
        kernels.linearNoBias(img_v, img_norm, block.img_v_weight, img_seq, hidden, hidden);

        // QK norm and RoPE for image
        applyQKNorm(img_q, img_k, block.img_norm_q_weight, block.img_norm_k_weight, img_seq, heads, head_dim, cfg.eps);
        applyRope2D(img_q, img_rope_cos, img_rope_sin, img_seq, heads, head_dim);
        applyRope2D(img_k, img_rope_cos, img_rope_sin, img_seq, heads, head_dim);

        // Text stream: AdaLN -> QKV
        const txt_norm = self.work1[img_seq * hidden ..][0 .. txt_seq * hidden];
        applyAdaLN(txt_norm, self.txt_hidden, txt_shift1, txt_scale1, txt_seq, hidden, cfg.eps);

        const txt_q = self.work2[img_seq * hidden * 3 ..][0 .. txt_seq * hidden];
        const txt_k = self.work2[img_seq * hidden * 3 + txt_seq * hidden ..][0 .. txt_seq * hidden];
        const txt_v = self.work2[img_seq * hidden * 3 + txt_seq * hidden * 2 ..][0 .. txt_seq * hidden];

        kernels.linearNoBias(txt_q, txt_norm, block.txt_q_weight, txt_seq, hidden, hidden);
        kernels.linearNoBias(txt_k, txt_norm, block.txt_k_weight, txt_seq, hidden, hidden);
        kernels.linearNoBias(txt_v, txt_norm, block.txt_v_weight, txt_seq, hidden, hidden);

        // QK norm and RoPE for text
        applyQKNorm(txt_q, txt_k, block.txt_norm_q_weight, block.txt_norm_k_weight, txt_seq, heads, head_dim, cfg.eps);
        applyRope2D(txt_q, txt_rope_cos, txt_rope_sin, txt_seq, heads, head_dim);
        applyRope2D(txt_k, txt_rope_cos, txt_rope_sin, txt_seq, heads, head_dim);

        // Joint attention
        const img_attn_out = try self.allocator.alloc(f32, img_seq * hidden);
        defer self.allocator.free(img_attn_out);
        const txt_attn_out = try self.allocator.alloc(f32, txt_seq * hidden);
        defer self.allocator.free(txt_attn_out);

        try jointAttention(
            self.allocator,
            img_attn_out,
            txt_attn_out,
            img_q,
            img_k,
            img_v,
            txt_q,
            txt_k,
            txt_v,
            img_seq,
            txt_seq,
            heads,
            head_dim,
        );

        // Project attention output
        const img_proj = self.work1[0 .. img_seq * hidden];
        const txt_proj = self.work1[img_seq * hidden ..][0 .. txt_seq * hidden];
        kernels.linearNoBias(img_proj, img_attn_out, block.img_proj_weight, img_seq, hidden, hidden);
        kernels.linearNoBias(txt_proj, txt_attn_out, block.txt_proj_weight, txt_seq, hidden, hidden);

        // Apply gate and add residual
        for (0..img_seq) |s| {
            for (0..hidden) |h| {
                self.img_hidden[s * hidden + h] += img_gate1[h] * img_proj[s * hidden + h];
            }
        }
        for (0..txt_seq) |s| {
            for (0..hidden) |h| {
                self.txt_hidden[s * hidden + h] += txt_gate1[h] * txt_proj[s * hidden + h];
            }
        }

        // FFN for image
        applyAdaLN(img_norm, self.img_hidden, img_shift2, img_scale2, img_seq, hidden, cfg.eps);
        swigluFFN(img_proj, img_norm, block.img_mlp_gate_weight, block.img_mlp_up_weight, block.img_mlp_down_weight, img_seq, hidden, mlp_hidden);
        for (0..img_seq) |s| {
            for (0..hidden) |h| {
                self.img_hidden[s * hidden + h] += img_gate2[h] * img_proj[s * hidden + h];
            }
        }

        // FFN for text
        applyAdaLN(txt_norm, self.txt_hidden, txt_shift2, txt_scale2, txt_seq, hidden, cfg.eps);
        swigluFFN(txt_proj, txt_norm, block.txt_mlp_gate_weight, block.txt_mlp_up_weight, block.txt_mlp_down_weight, txt_seq, hidden, mlp_hidden);
        for (0..txt_seq) |s| {
            for (0..hidden) |h| {
                self.txt_hidden[s * hidden + h] += txt_gate2[h] * txt_proj[s * hidden + h];
            }
        }
    }

    fn singleBlockForward(
        self: *Self,
        hidden_state: []f32,
        block: *const SingleBlock,
        t_emb: []const f32,
        img_rope_cos: []const f32,
        img_rope_sin: []const f32,
        txt_rope_cos: []const f32,
        txt_rope_sin: []const f32,
        total_seq: usize,
        txt_seq: usize,
    ) !void {
        const cfg = self.config;
        const h_size = cfg.hidden_size;
        const heads = cfg.num_heads;
        const head_dim = cfg.head_dim;
        const mlp_hidden = cfg.mlp_hidden;
        const img_seq = total_seq - txt_seq;

        // Apply SiLU to t_emb
        var t_emb_silu: [3072]f32 = undefined;
        for (0..h_size) |i| {
            const x = t_emb[i];
            t_emb_silu[i] = x / (1.0 + @exp(-x));
        }

        // Compute modulation (3 params: shift, scale, gate)
        var mod_params: [9216]f32 = undefined; // h_size * 3
        kernels.linearNoBias(&mod_params, &t_emb_silu, self.adaln_single_weight, 1, h_size, h_size * 3);

        const shift = mod_params[0..h_size];
        const scale = mod_params[h_size..][0..h_size];
        const gate = mod_params[h_size * 2 ..][0..h_size];

        // AdaLN norm
        const norm = self.work1[0 .. total_seq * h_size];
        applyAdaLN(norm, hidden_state, shift, scale, total_seq, h_size, cfg.eps);

        // Fused QKV + MLP projection
        const fused_dim = h_size * 3 + mlp_hidden * 2;
        const fused_out = try self.allocator.alloc(f32, total_seq * fused_dim);
        defer self.allocator.free(fused_out);
        kernels.linearNoBias(fused_out, norm, block.qkv_mlp_weight, total_seq, h_size, fused_dim);

        // Split outputs
        const q = try self.allocator.alloc(f32, total_seq * h_size);
        defer self.allocator.free(q);
        const k = try self.allocator.alloc(f32, total_seq * h_size);
        defer self.allocator.free(k);
        const v = try self.allocator.alloc(f32, total_seq * h_size);
        defer self.allocator.free(v);
        const mlp_gate_buf = try self.allocator.alloc(f32, total_seq * mlp_hidden);
        defer self.allocator.free(mlp_gate_buf);
        const mlp_up_buf = try self.allocator.alloc(f32, total_seq * mlp_hidden);
        defer self.allocator.free(mlp_up_buf);

        for (0..total_seq) |s| {
            const row = fused_out[s * fused_dim ..][0..fused_dim];
            @memcpy(q[s * h_size ..][0..h_size], row[0..h_size]);
            @memcpy(k[s * h_size ..][0..h_size], row[h_size..][0..h_size]);
            @memcpy(v[s * h_size ..][0..h_size], row[h_size * 2 ..][0..h_size]);
            @memcpy(mlp_gate_buf[s * mlp_hidden ..][0..mlp_hidden], row[h_size * 3 ..][0..mlp_hidden]);
            @memcpy(mlp_up_buf[s * mlp_hidden ..][0..mlp_hidden], row[h_size * 3 + mlp_hidden ..][0..mlp_hidden]);
        }

        // QK norm
        applyQKNorm(q, k, block.norm_q_weight, block.norm_k_weight, total_seq, heads, head_dim, cfg.eps);

        // Apply RoPE: text portion then image portion
        applyRope2D(q[0 .. txt_seq * h_size], txt_rope_cos, txt_rope_sin, txt_seq, heads, head_dim);
        applyRope2D(k[0 .. txt_seq * h_size], txt_rope_cos, txt_rope_sin, txt_seq, heads, head_dim);
        applyRope2D(q[txt_seq * h_size ..][0 .. img_seq * h_size], img_rope_cos, img_rope_sin, img_seq, heads, head_dim);
        applyRope2D(k[txt_seq * h_size ..][0 .. img_seq * h_size], img_rope_cos, img_rope_sin, img_seq, heads, head_dim);

        // Self-attention
        const attn_out = try self.allocator.alloc(f32, total_seq * h_size);
        defer self.allocator.free(attn_out);
        try mhaForward(self.allocator, attn_out, q, k, v, total_seq, heads, head_dim);

        // SwiGLU: silu(gate) * up
        kernels.silu(mlp_gate_buf);
        kernels.mul(mlp_gate_buf, mlp_gate_buf, mlp_up_buf);

        // Fused output projection
        const concat = try self.allocator.alloc(f32, total_seq * (h_size + mlp_hidden));
        defer self.allocator.free(concat);
        for (0..total_seq) |s| {
            @memcpy(concat[s * (h_size + mlp_hidden) ..][0..h_size], attn_out[s * h_size ..][0..h_size]);
            @memcpy(concat[s * (h_size + mlp_hidden) + h_size ..][0..mlp_hidden], mlp_gate_buf[s * mlp_hidden ..][0..mlp_hidden]);
        }

        const proj_out = self.work1[0 .. total_seq * h_size];
        kernels.linearNoBias(proj_out, concat, block.proj_mlp_weight, total_seq, h_size + mlp_hidden, h_size);

        // Apply gate and add residual
        for (0..total_seq) |s| {
            for (0..h_size) |h| {
                hidden_state[s * h_size + h] += gate[h] * proj_out[s * h_size + h];
            }
        }
    }

    pub fn deinit(self: *Self) void {
        // Free working memory (allocated from main allocator)
        self.allocator.free(self.work1);
        self.allocator.free(self.work2);
        self.allocator.free(self.work3);
        self.allocator.free(self.img_hidden);
        self.allocator.free(self.txt_hidden);

        // Free all weights at once via arena
        self.arena.deinit();

        self.allocator.destroy(self);
    }
};

// ─────────────────────────────────────────────────────────────────────────────
// Helper functions
// ─────────────────────────────────────────────────────────────────────────────

fn loadTensor(st: *SafeTensors, allocator: Allocator, name: []const u8) ![]f32 {
    var tensor = try st.getTensor(allocator, name);
    defer tensor.deinit();
    return try allocator.dupe(f32, tensor.data);
}

fn loadDoubleBlock(st: *SafeTensors, allocator: Allocator, idx: usize, cfg: Config) !DoubleBlock {
    var buf: [256]u8 = undefined;
    const h = cfg.hidden_size;
    const mlp = cfg.mlp_hidden;

    var block: DoubleBlock = undefined;

    // Image stream
    block.img_norm_q_weight = try loadTensor(st, allocator, std.fmt.bufPrint(&buf, "transformer_blocks.{d}.attn.norm_q.weight", .{idx}) catch unreachable);
    block.img_norm_k_weight = try loadTensor(st, allocator, std.fmt.bufPrint(&buf, "transformer_blocks.{d}.attn.norm_k.weight", .{idx}) catch unreachable);
    block.img_q_weight = try loadTensor(st, allocator, std.fmt.bufPrint(&buf, "transformer_blocks.{d}.attn.to_q.weight", .{idx}) catch unreachable);
    block.img_k_weight = try loadTensor(st, allocator, std.fmt.bufPrint(&buf, "transformer_blocks.{d}.attn.to_k.weight", .{idx}) catch unreachable);
    block.img_v_weight = try loadTensor(st, allocator, std.fmt.bufPrint(&buf, "transformer_blocks.{d}.attn.to_v.weight", .{idx}) catch unreachable);
    block.img_proj_weight = try loadTensor(st, allocator, std.fmt.bufPrint(&buf, "transformer_blocks.{d}.attn.to_out.0.weight", .{idx}) catch unreachable);

    // Image FFN (fused gate+up)
    const ff_in = try loadTensor(st, allocator, std.fmt.bufPrint(&buf, "transformer_blocks.{d}.ff.linear_in.weight", .{idx}) catch unreachable);
    defer allocator.free(ff_in);
    block.img_mlp_gate_weight = try allocator.dupe(f32, ff_in[0 .. mlp * h]);
    block.img_mlp_up_weight = try allocator.dupe(f32, ff_in[mlp * h ..][0 .. mlp * h]);
    block.img_mlp_down_weight = try loadTensor(st, allocator, std.fmt.bufPrint(&buf, "transformer_blocks.{d}.ff.linear_out.weight", .{idx}) catch unreachable);

    // Text stream
    block.txt_norm_q_weight = try loadTensor(st, allocator, std.fmt.bufPrint(&buf, "transformer_blocks.{d}.attn.norm_added_q.weight", .{idx}) catch unreachable);
    block.txt_norm_k_weight = try loadTensor(st, allocator, std.fmt.bufPrint(&buf, "transformer_blocks.{d}.attn.norm_added_k.weight", .{idx}) catch unreachable);
    block.txt_q_weight = try loadTensor(st, allocator, std.fmt.bufPrint(&buf, "transformer_blocks.{d}.attn.add_q_proj.weight", .{idx}) catch unreachable);
    block.txt_k_weight = try loadTensor(st, allocator, std.fmt.bufPrint(&buf, "transformer_blocks.{d}.attn.add_k_proj.weight", .{idx}) catch unreachable);
    block.txt_v_weight = try loadTensor(st, allocator, std.fmt.bufPrint(&buf, "transformer_blocks.{d}.attn.add_v_proj.weight", .{idx}) catch unreachable);
    block.txt_proj_weight = try loadTensor(st, allocator, std.fmt.bufPrint(&buf, "transformer_blocks.{d}.attn.to_add_out.weight", .{idx}) catch unreachable);

    const txt_ff_in = try loadTensor(st, allocator, std.fmt.bufPrint(&buf, "transformer_blocks.{d}.ff_context.linear_in.weight", .{idx}) catch unreachable);
    defer allocator.free(txt_ff_in);
    block.txt_mlp_gate_weight = try allocator.dupe(f32, txt_ff_in[0 .. mlp * h]);
    block.txt_mlp_up_weight = try allocator.dupe(f32, txt_ff_in[mlp * h ..][0 .. mlp * h]);
    block.txt_mlp_down_weight = try loadTensor(st, allocator, std.fmt.bufPrint(&buf, "transformer_blocks.{d}.ff_context.linear_out.weight", .{idx}) catch unreachable);

    return block;
}

fn loadSingleBlock(st: *SafeTensors, allocator: Allocator, idx: usize) !SingleBlock {
    var buf: [256]u8 = undefined;

    return SingleBlock{
        .norm_q_weight = try loadTensor(st, allocator, std.fmt.bufPrint(&buf, "single_transformer_blocks.{d}.attn.norm_q.weight", .{idx}) catch unreachable),
        .norm_k_weight = try loadTensor(st, allocator, std.fmt.bufPrint(&buf, "single_transformer_blocks.{d}.attn.norm_k.weight", .{idx}) catch unreachable),
        .qkv_mlp_weight = try loadTensor(st, allocator, std.fmt.bufPrint(&buf, "single_transformer_blocks.{d}.attn.to_qkv_mlp_proj.weight", .{idx}) catch unreachable),
        .proj_mlp_weight = try loadTensor(st, allocator, std.fmt.bufPrint(&buf, "single_transformer_blocks.{d}.attn.to_out.weight", .{idx}) catch unreachable),
    };
}

/// Sinusoidal timestep embedding
fn getTimestepEmbedding(out: []f32, t: f32, dim: usize, max_period: f32) void {
    const half = dim / 2;
    const log_max = @log(max_period);

    for (0..half) |i| {
        const freq = @exp(-log_max * @as(f32, @floatFromInt(i)) / @as(f32, @floatFromInt(half)));
        const angle = t * freq;
        out[i] = @cos(angle);
        out[i + half] = @sin(angle);
    }
}

/// AdaLN: out = (1 + scale) * LayerNorm(x) + shift
fn applyAdaLN(out: []f32, x: []const f32, shift: []const f32, scale: []const f32, seq: usize, hidden: usize, eps: f32) void {
    for (0..seq) |s| {
        const x_row = x[s * hidden ..][0..hidden];
        const out_row = out[s * hidden ..][0..hidden];

        // Layer norm
        var sum: f32 = 0;
        for (x_row) |v| sum += v;
        const mean = sum / @as(f32, @floatFromInt(hidden));

        var var_sum: f32 = 0;
        for (x_row) |v| {
            const diff = v - mean;
            var_sum += diff * diff;
        }
        const inv_std = 1.0 / @sqrt(var_sum / @as(f32, @floatFromInt(hidden)) + eps);

        // Apply modulation
        for (0..hidden) |i| {
            const norm_val = (x_row[i] - mean) * inv_std;
            out_row[i] = (1.0 + scale[i]) * norm_val + shift[i];
        }
    }
}

/// QK normalization (RMSNorm per head)
fn applyQKNorm(q: []f32, k: []f32, q_weight: []const f32, k_weight: []const f32, seq: usize, heads: usize, head_dim: usize, eps: f32) void {
    for (0..seq) |s| {
        for (0..heads) |h| {
            const qh = q[s * heads * head_dim + h * head_dim ..][0..head_dim];
            const kh = k[s * heads * head_dim + h * head_dim ..][0..head_dim];

            // Q RMS norm
            var sum_sq: f32 = 0;
            for (qh) |v| sum_sq += v * v;
            var rms_inv = 1.0 / @sqrt(sum_sq / @as(f32, @floatFromInt(head_dim)) + eps);
            for (0..head_dim) |d| qh[d] = qh[d] * rms_inv * q_weight[d];

            // K RMS norm
            sum_sq = 0;
            for (kh) |v| sum_sq += v * v;
            rms_inv = 1.0 / @sqrt(sum_sq / @as(f32, @floatFromInt(head_dim)) + eps);
            for (0..head_dim) |d| kh[d] = kh[d] * rms_inv * k_weight[d];
        }
    }
}

/// Compute 2D RoPE frequencies for image tokens
fn computeRope2D(cos_out: []f32, sin_out: []f32, patch_h: usize, patch_w: usize, axis_dim: usize, theta: f32) void {
    const half_axis = axis_dim / 2;
    const head_dim = axis_dim * 4;

    for (0..patch_h) |hy| {
        for (0..patch_w) |wx| {
            const pos = hy * patch_w + wx;
            const cos_p = cos_out[pos * head_dim ..][0..head_dim];
            const sin_p = sin_out[pos * head_dim ..][0..head_dim];

            // Axis 0: T=0 (identity)
            for (0..axis_dim) |d| {
                cos_p[d] = 1.0;
                sin_p[d] = 0.0;
            }

            // Axis 1: H position
            for (0..half_axis) |d| {
                const freq = 1.0 / std.math.pow(f32, theta, @as(f32, @floatFromInt(2 * d)) / @as(f32, @floatFromInt(axis_dim)));
                const angle = @as(f32, @floatFromInt(hy)) * freq;
                const c = @cos(angle);
                const s = @sin(angle);
                cos_p[axis_dim + d * 2] = c;
                cos_p[axis_dim + d * 2 + 1] = c;
                sin_p[axis_dim + d * 2] = s;
                sin_p[axis_dim + d * 2 + 1] = s;
            }

            // Axis 2: W position
            for (0..half_axis) |d| {
                const freq = 1.0 / std.math.pow(f32, theta, @as(f32, @floatFromInt(2 * d)) / @as(f32, @floatFromInt(axis_dim)));
                const angle = @as(f32, @floatFromInt(wx)) * freq;
                const c = @cos(angle);
                const s = @sin(angle);
                cos_p[axis_dim * 2 + d * 2] = c;
                cos_p[axis_dim * 2 + d * 2 + 1] = c;
                sin_p[axis_dim * 2 + d * 2] = s;
                sin_p[axis_dim * 2 + d * 2 + 1] = s;
            }

            // Axis 3: L=0 (identity)
            for (0..axis_dim) |d| {
                cos_p[axis_dim * 3 + d] = 1.0;
                sin_p[axis_dim * 3 + d] = 0.0;
            }
        }
    }
}

/// Compute text RoPE frequencies (position in axis 3)
fn computeRopeText(cos_out: []f32, sin_out: []f32, txt_seq: usize, axis_dim: usize, theta: f32) void {
    const half_axis = axis_dim / 2;
    const head_dim = axis_dim * 4;

    for (0..txt_seq) |s| {
        const cos_p = cos_out[s * head_dim ..][0..head_dim];
        const sin_p = sin_out[s * head_dim ..][0..head_dim];

        // Axes 0-2: identity
        for (0..axis_dim * 3) |d| {
            cos_p[d] = 1.0;
            sin_p[d] = 0.0;
        }

        // Axis 3: L position
        for (0..half_axis) |d| {
            const freq = 1.0 / std.math.pow(f32, theta, @as(f32, @floatFromInt(2 * d)) / @as(f32, @floatFromInt(axis_dim)));
            const angle = @as(f32, @floatFromInt(s)) * freq;
            const c = @cos(angle);
            const si = @sin(angle);
            cos_p[axis_dim * 3 + d * 2] = c;
            cos_p[axis_dim * 3 + d * 2 + 1] = c;
            sin_p[axis_dim * 3 + d * 2] = si;
            sin_p[axis_dim * 3 + d * 2 + 1] = si;
        }
    }
}

/// Apply 2D RoPE to Q/K
fn applyRope2D(x: []f32, cos_freq: []const f32, sin_freq: []const f32, seq: usize, heads: usize, head_dim: usize) void {
    for (0..seq) |s| {
        const cos_s = cos_freq[s * head_dim ..][0..head_dim];
        const sin_s = sin_freq[s * head_dim ..][0..head_dim];

        for (0..heads) |h| {
            const vec = x[(s * heads + h) * head_dim ..][0..head_dim];

            var d: usize = 0;
            while (d < head_dim) : (d += 2) {
                const cos_val = cos_s[d];
                const sin_val = sin_s[d];
                const x0 = vec[d];
                const x1 = vec[d + 1];
                vec[d] = x0 * cos_val - x1 * sin_val;
                vec[d + 1] = x1 * cos_val + x0 * sin_val;
            }
        }
    }
}

/// SwiGLU FFN
fn swigluFFN(out: []f32, x: []const f32, gate_weight: []const f32, up_weight: []const f32, down_weight: []const f32, seq: usize, hidden: usize, mlp_hidden: usize) void {
    var gate: [9216 * 100]f32 = undefined; // Max seq * mlp_hidden
    var up: [9216 * 100]f32 = undefined;

    kernels.linearNoBias(gate[0 .. seq * mlp_hidden], x, gate_weight, seq, hidden, mlp_hidden);
    kernels.linearNoBias(up[0 .. seq * mlp_hidden], x, up_weight, seq, hidden, mlp_hidden);

    // SiLU(gate) * up
    kernels.silu(gate[0 .. seq * mlp_hidden]);
    kernels.mul(gate[0 .. seq * mlp_hidden], gate[0 .. seq * mlp_hidden], up[0 .. seq * mlp_hidden]);

    // Down projection
    kernels.linearNoBias(out, gate[0 .. seq * mlp_hidden], down_weight, seq, mlp_hidden, hidden);
}

/// Multi-head self-attention
fn mhaForward(allocator: Allocator, out: []f32, q: []const f32, k: []const f32, v: []const f32, seq: usize, heads: usize, head_dim: usize) !void {
    const scale = 1.0 / @sqrt(@as(f32, @floatFromInt(head_dim)));
    const scores = try allocator.alloc(f32, seq * seq);
    defer allocator.free(scores);

    for (0..heads) |h| {
        // Q @ K^T for this head
        for (0..seq) |i| {
            for (0..seq) |j| {
                var dot: f32 = 0;
                for (0..head_dim) |d| {
                    dot += q[(i * heads + h) * head_dim + d] * k[(j * heads + h) * head_dim + d];
                }
                scores[i * seq + j] = dot * scale;
            }
        }

        // Softmax
        kernels.softmax(scores, seq);

        // scores @ V
        for (0..seq) |i| {
            for (0..head_dim) |d| {
                var sum: f32 = 0;
                for (0..seq) |j| {
                    sum += scores[i * seq + j] * v[(j * heads + h) * head_dim + d];
                }
                out[(i * heads + h) * head_dim + d] = sum;
            }
        }
    }
}

/// Joint attention for double blocks
fn jointAttention(
    allocator: Allocator,
    img_out: []f32,
    txt_out: []f32,
    img_q: []const f32,
    img_k: []const f32,
    img_v: []const f32,
    txt_q: []const f32,
    txt_k: []const f32,
    txt_v: []const f32,
    img_seq: usize,
    txt_seq: usize,
    heads: usize,
    head_dim: usize,
) !void {
    const total_seq = img_seq + txt_seq;
    const hidden = heads * head_dim;
    const scale = 1.0 / @sqrt(@as(f32, @floatFromInt(head_dim)));

    // Concatenate K, V: [txt, img]
    const cat_k = try allocator.alloc(f32, total_seq * hidden);
    defer allocator.free(cat_k);
    const cat_v = try allocator.alloc(f32, total_seq * hidden);
    defer allocator.free(cat_v);

    @memcpy(cat_k[0 .. txt_seq * hidden], txt_k[0 .. txt_seq * hidden]);
    @memcpy(cat_k[txt_seq * hidden ..][0 .. img_seq * hidden], img_k[0 .. img_seq * hidden]);
    @memcpy(cat_v[0 .. txt_seq * hidden], txt_v[0 .. txt_seq * hidden]);
    @memcpy(cat_v[txt_seq * hidden ..][0 .. img_seq * hidden], img_v[0 .. img_seq * hidden]);

    // Attention scores buffer
    const max_scores = @max(img_seq, txt_seq) * total_seq;
    const scores = try allocator.alloc(f32, max_scores);
    defer allocator.free(scores);

    for (0..heads) |h| {
        // Image Q @ cat_K^T
        for (0..img_seq) |i| {
            for (0..total_seq) |j| {
                var dot: f32 = 0;
                for (0..head_dim) |d| {
                    dot += img_q[(i * heads + h) * head_dim + d] * cat_k[(j * heads + h) * head_dim + d];
                }
                scores[i * total_seq + j] = dot * scale;
            }
        }

        // Softmax rows
        for (0..img_seq) |i| {
            const row = scores[i * total_seq ..][0..total_seq];
            var max_val: f32 = -std.math.inf(f32);
            for (row) |v| max_val = @max(max_val, v);
            var sum: f32 = 0;
            for (row) |*v| {
                v.* = @exp(v.* - max_val);
                sum += v.*;
            }
            for (row) |*v| v.* /= sum;
        }

        // scores @ cat_V -> img_out
        for (0..img_seq) |i| {
            for (0..head_dim) |d| {
                var sum: f32 = 0;
                for (0..total_seq) |j| {
                    sum += scores[i * total_seq + j] * cat_v[(j * heads + h) * head_dim + d];
                }
                img_out[(i * heads + h) * head_dim + d] = sum;
            }
        }

        // Text Q @ cat_K^T
        for (0..txt_seq) |i| {
            for (0..total_seq) |j| {
                var dot: f32 = 0;
                for (0..head_dim) |d| {
                    dot += txt_q[(i * heads + h) * head_dim + d] * cat_k[(j * heads + h) * head_dim + d];
                }
                scores[i * total_seq + j] = dot * scale;
            }
        }

        // Softmax rows
        for (0..txt_seq) |i| {
            const row = scores[i * total_seq ..][0..total_seq];
            var max_val: f32 = -std.math.inf(f32);
            for (row) |v| max_val = @max(max_val, v);
            var sum: f32 = 0;
            for (row) |*v| {
                v.* = @exp(v.* - max_val);
                sum += v.*;
            }
            for (row) |*v| v.* /= sum;
        }

        // scores @ cat_V -> txt_out
        for (0..txt_seq) |i| {
            for (0..head_dim) |d| {
                var sum: f32 = 0;
                for (0..total_seq) |j| {
                    sum += scores[i * total_seq + j] * cat_v[(j * heads + h) * head_dim + d];
                }
                txt_out[(i * heads + h) * head_dim + d] = sum;
            }
        }
    }
}
