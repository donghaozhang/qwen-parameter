
Vision-Language Merger Flow:

    ┌───────────────┐      ┌──────────────┐
    │ Vision Model  │      │ Text Tokens  │
    └───────┬───────┘      └──────┬───────┘
            │                     │
            ▼                     ▼
    ┌───────────────┐      ┌──────────────  ┐
    │Vision Features│      │Token Embeddings│
    └───────┬───────┘      └──────┬───────  ┘
            │                     │
            ▼                     │
    ┌───────────────┐             │
    │Vision Projection│           │
    └───────┬───────┘             │
            │                     │
            ▼                     ▼
    ┌─────────────────────────────────┐
    │     Merged Representation       │
    └─────────────────┬───────────────┘
                      │
                      ▼
    ┌─────────────────────────────────┐
    │      Language Model Layers      │
    └─────────────────┬───────────────┘
                      │
                      ▼
    ┌─────────────────────────────────┐
    │          Output Tokens          │
    └─────────────────────────────────┘
    
Loading Qwen/Qwen2.5-VL-7B-Instruct...

=== MODEL COMPONENTS ANALYSIS ===
+---------------------------+------------+------------+
|         Component         | Parameters | % of Total |
+---------------------------+------------+------------+
|       Vision Encoder      |  676.55M   |   8.16%    |
|       Language Model      |   7.07B    |   85.27%   |
| Vision-Language Connector |     0      |   0.00%    |
|      Other Components     |  545.00M   |   6.57%    |
|           Total           |   8.29B    |  100.00%   |
+---------------------------+------------+------------+

Detailed Vision Component Analysis:
+--------------------+---------------------------------------+
|     Component      |               Parameters              |
+--------------------+---------------------------------------+
|  Patch Embedding   |                 1.51M                 |
| Transformer Blocks | 630.47M (32 blocks, 19.70M per block) |
+--------------------+---------------------------------------+

Vision Component Structure:
- patch_embed: Qwen2_5_VisionPatchEmbed (1.51M params)
- rotary_pos_emb: Qwen2_5_VisionRotaryEmbedding (0 params)
- blocks: ModuleList (630.47M params)
- merger: Qwen2_5_VLPatchMerger (44.57M params)

Detailed Language Model Analysis:
+--------------------+--------------------------------------+
|     Component      |              Parameters              |
+--------------------+--------------------------------------+
|  Token Embeddings  |               545.00M                |
| Transformer Layers | 6.53B (28 layers, 233.06M per layer) |
|      LM Head       |               545.00M                |
|     Final Norm     |                3.58K                 |
+--------------------+--------------------------------------+

=== VISION-LANGUAGE MERGER ANALYSIS ===
+---------------------+-----------------------+------------+----------------------------+
|      Component      |          Type         | Parameters |           Shape            |
+---------------------+-----------------------+------------+----------------------------+
|    visual.merger    | Qwen2_5_VLPatchMerger |   44.57M   |    ln_q.weight: [1280]     |
|                     |                       |            | mlp.0.weight: [5120, 5120] |
|                     |                       |            |     mlp.0.bias: [5120]     |
|                     |                       |            |            ...             |
|  visual.merger.ln_q |      Qwen2RMSNorm     |   1.28K    |       weight: [1280]       |
|  visual.merger.mlp  |       Sequential      |   44.57M   |   0.weight: [5120, 5120]   |
|                     |                       |            |       0.bias: [5120]       |
|                     |                       |            |   2.weight: [3584, 5120]   |
|                     |                       |            |            ...             |
| visual.merger.mlp.0 |         Linear        |   26.22M   |    weight: [5120, 5120]    |
|                     |                       |            |        bias: [5120]        |
| visual.merger.mlp.2 |         Linear        |   18.35M   |    weight: [3584, 5120]    |
|                     |                       |            |        bias: [3584]        |
+---------------------+-----------------------+------------+----------------------------+

Vision-Language Information Flow Analysis:
Vision Embedding Dimension: 1280
Language Embedding Dimension: 3584

Overall Model Architecture:
1. visual: Qwen2_5_VisionTransformerPretrainedModel (676.55M params)
2. model: Qwen2_5_VLModel (7.07B params)
3. lm_head: Linear (545.00M params)

=== MODEL STRUCTURE ANALYSIS ===

Vision-related modules:
- visual: Qwen2_5_VisionTransformerPretrainedModel (676.55M parameters)
- visual.patch_embed: Qwen2_5_VisionPatchEmbed (1.51M parameters)
- visual.patch_embed.proj: Conv3d (1.51M parameters)
- visual.rotary_pos_emb: Qwen2_5_VisionRotaryEmbedding (0 parameters)
- visual.blocks: ModuleList (630.47M parameters)
- visual.blocks.0: Qwen2_5_VLVisionBlock (19.70M parameters)
- visual.blocks.0.norm1: Qwen2RMSNorm (1.28K parameters)
- visual.blocks.0.norm2: Qwen2RMSNorm (1.28K parameters)
- visual.blocks.0.attn: Qwen2_5_VLVisionSdpaAttention (6.56M parameters)
- visual.blocks.0.attn.qkv: Linear (4.92M parameters)
- visual.blocks.0.attn.proj: Linear (1.64M parameters)
- visual.blocks.0.mlp: Qwen2_5_VLMLP (13.14M parameters)
- visual.blocks.0.mlp.gate_proj: Linear (4.38M parameters)
- visual.blocks.0.mlp.up_proj: Linear (4.38M parameters)
- visual.blocks.0.mlp.down_proj: Linear (4.38M parameters)
- visual.blocks.0.mlp.act_fn: SiLU (0 parameters)
- visual.blocks.1: Qwen2_5_VLVisionBlock (19.70M parameters)
- visual.blocks.1.norm1: Qwen2RMSNorm (1.28K parameters)
- visual.blocks.1.norm2: Qwen2RMSNorm (1.28K parameters)
- visual.blocks.1.attn: Qwen2_5_VLVisionSdpaAttention (6.56M parameters)
- visual.blocks.1.attn.qkv: Linear (4.92M parameters)
- visual.blocks.1.attn.proj: Linear (1.64M parameters)
- visual.blocks.1.mlp: Qwen2_5_VLMLP (13.14M parameters)
- visual.blocks.1.mlp.gate_proj: Linear (4.38M parameters)
- visual.blocks.1.mlp.up_proj: Linear (4.38M parameters)
- visual.blocks.1.mlp.down_proj: Linear (4.38M parameters)
- visual.blocks.1.mlp.act_fn: SiLU (0 parameters)
- visual.blocks.2: Qwen2_5_VLVisionBlock (19.70M parameters)
- visual.blocks.2.norm1: Qwen2RMSNorm (1.28K parameters)
- visual.blocks.2.norm2: Qwen2RMSNorm (1.28K parameters)
- visual.blocks.2.attn: Qwen2_5_VLVisionSdpaAttention (6.56M parameters)
- visual.blocks.2.attn.qkv: Linear (4.92M parameters)
- visual.blocks.2.attn.proj: Linear (1.64M parameters)
- visual.blocks.2.mlp: Qwen2_5_VLMLP (13.14M parameters)
- visual.blocks.2.mlp.gate_proj: Linear (4.38M parameters)
- visual.blocks.2.mlp.up_proj: Linear (4.38M parameters)
- visual.blocks.2.mlp.down_proj: Linear (4.38M parameters)
- visual.blocks.2.mlp.act_fn: SiLU (0 parameters)
- visual.blocks.3: Qwen2_5_VLVisionBlock (19.70M parameters)
- visual.blocks.3.norm1: Qwen2RMSNorm (1.28K parameters)
- visual.blocks.3.norm2: Qwen2RMSNorm (1.28K parameters)
- visual.blocks.3.attn: Qwen2_5_VLVisionSdpaAttention (6.56M parameters)
- visual.blocks.3.attn.qkv: Linear (4.92M parameters)
- visual.blocks.3.attn.proj: Linear (1.64M parameters)
- visual.blocks.3.mlp: Qwen2_5_VLMLP (13.14M parameters)
- visual.blocks.3.mlp.gate_proj: Linear (4.38M parameters)
- visual.blocks.3.mlp.up_proj: Linear (4.38M parameters)
- visual.blocks.3.mlp.down_proj: Linear (4.38M parameters)
- visual.blocks.3.mlp.act_fn: SiLU (0 parameters)
- visual.blocks.4: Qwen2_5_VLVisionBlock (19.70M parameters)
- visual.blocks.4.norm1: Qwen2RMSNorm (1.28K parameters)
- visual.blocks.4.norm2: Qwen2RMSNorm (1.28K parameters)
- visual.blocks.4.attn: Qwen2_5_VLVisionSdpaAttention (6.56M parameters)
- visual.blocks.4.attn.qkv: Linear (4.92M parameters)
- visual.blocks.4.attn.proj: Linear (1.64M parameters)
- visual.blocks.4.mlp: Qwen2_5_VLMLP (13.14M parameters)
- visual.blocks.4.mlp.gate_proj: Linear (4.38M parameters)
- visual.blocks.4.mlp.up_proj: Linear (4.38M parameters)
- visual.blocks.4.mlp.down_proj: Linear (4.38M parameters)
- visual.blocks.4.mlp.act_fn: SiLU (0 parameters)
- visual.blocks.5: Qwen2_5_VLVisionBlock (19.70M parameters)
- visual.blocks.5.norm1: Qwen2RMSNorm (1.28K parameters)
- visual.blocks.5.norm2: Qwen2RMSNorm (1.28K parameters)
- visual.blocks.5.attn: Qwen2_5_VLVisionSdpaAttention (6.56M parameters)
- visual.blocks.5.attn.qkv: Linear (4.92M parameters)
- visual.blocks.5.attn.proj: Linear (1.64M parameters)
- visual.blocks.5.mlp: Qwen2_5_VLMLP (13.14M parameters)
- visual.blocks.5.mlp.gate_proj: Linear (4.38M parameters)
- visual.blocks.5.mlp.up_proj: Linear (4.38M parameters)
- visual.blocks.5.mlp.down_proj: Linear (4.38M parameters)
- visual.blocks.5.mlp.act_fn: SiLU (0 parameters)
- visual.blocks.6: Qwen2_5_VLVisionBlock (19.70M parameters)
- visual.blocks.6.norm1: Qwen2RMSNorm (1.28K parameters)
- visual.blocks.6.norm2: Qwen2RMSNorm (1.28K parameters)
- visual.blocks.6.attn: Qwen2_5_VLVisionSdpaAttention (6.56M parameters)
- visual.blocks.6.attn.qkv: Linear (4.92M parameters)
- visual.blocks.6.attn.proj: Linear (1.64M parameters)
- visual.blocks.6.mlp: Qwen2_5_VLMLP (13.14M parameters)
- visual.blocks.6.mlp.gate_proj: Linear (4.38M parameters)
- visual.blocks.6.mlp.up_proj: Linear (4.38M parameters)
- visual.blocks.6.mlp.down_proj: Linear (4.38M parameters)
- visual.blocks.6.mlp.act_fn: SiLU (0 parameters)
- visual.blocks.7: Qwen2_5_VLVisionBlock (19.70M parameters)
- visual.blocks.7.norm1: Qwen2RMSNorm (1.28K parameters)
- visual.blocks.7.norm2: Qwen2RMSNorm (1.28K parameters)
- visual.blocks.7.attn: Qwen2_5_VLVisionSdpaAttention (6.56M parameters)
- visual.blocks.7.attn.qkv: Linear (4.92M parameters)
- visual.blocks.7.attn.proj: Linear (1.64M parameters)
- visual.blocks.7.mlp: Qwen2_5_VLMLP (13.14M parameters)
- visual.blocks.7.mlp.gate_proj: Linear (4.38M parameters)
- visual.blocks.7.mlp.up_proj: Linear (4.38M parameters)
- visual.blocks.7.mlp.down_proj: Linear (4.38M parameters)
- visual.blocks.7.mlp.act_fn: SiLU (0 parameters)
- visual.blocks.8: Qwen2_5_VLVisionBlock (19.70M parameters)
- visual.blocks.8.norm1: Qwen2RMSNorm (1.28K parameters)
- visual.blocks.8.norm2: Qwen2RMSNorm (1.28K parameters)
- visual.blocks.8.attn: Qwen2_5_VLVisionSdpaAttention (6.56M parameters)
- visual.blocks.8.attn.qkv: Linear (4.92M parameters)
- visual.blocks.8.attn.proj: Linear (1.64M parameters)
- visual.blocks.8.mlp: Qwen2_5_VLMLP (13.14M parameters)
- visual.blocks.8.mlp.gate_proj: Linear (4.38M parameters)
- visual.blocks.8.mlp.up_proj: Linear (4.38M parameters)
- visual.blocks.8.mlp.down_proj: Linear (4.38M parameters)
- visual.blocks.8.mlp.act_fn: SiLU (0 parameters)
- visual.blocks.9: Qwen2_5_VLVisionBlock (19.70M parameters)
- visual.blocks.9.norm1: Qwen2RMSNorm (1.28K parameters)
- visual.blocks.9.norm2: Qwen2RMSNorm (1.28K parameters)
- visual.blocks.9.attn: Qwen2_5_VLVisionSdpaAttention (6.56M parameters)
- visual.blocks.9.attn.qkv: Linear (4.92M parameters)
- visual.blocks.9.attn.proj: Linear (1.64M parameters)
- visual.blocks.9.mlp: Qwen2_5_VLMLP (13.14M parameters)
- visual.blocks.9.mlp.gate_proj: Linear (4.38M parameters)
- visual.blocks.9.mlp.up_proj: Linear (4.38M parameters)
- visual.blocks.9.mlp.down_proj: Linear (4.38M parameters)
- visual.blocks.9.mlp.act_fn: SiLU (0 parameters)
- visual.blocks.10: Qwen2_5_VLVisionBlock (19.70M parameters)
- visual.blocks.10.norm1: Qwen2RMSNorm (1.28K parameters)
- visual.blocks.10.norm2: Qwen2RMSNorm (1.28K parameters)
- visual.blocks.10.attn: Qwen2_5_VLVisionSdpaAttention (6.56M parameters)
- visual.blocks.10.attn.qkv: Linear (4.92M parameters)
- visual.blocks.10.attn.proj: Linear (1.64M parameters)
- visual.blocks.10.mlp: Qwen2_5_VLMLP (13.14M parameters)
- visual.blocks.10.mlp.gate_proj: Linear (4.38M parameters)
- visual.blocks.10.mlp.up_proj: Linear (4.38M parameters)
- visual.blocks.10.mlp.down_proj: Linear (4.38M parameters)
- visual.blocks.10.mlp.act_fn: SiLU (0 parameters)
- visual.blocks.11: Qwen2_5_VLVisionBlock (19.70M parameters)
- visual.blocks.11.norm1: Qwen2RMSNorm (1.28K parameters)
- visual.blocks.11.norm2: Qwen2RMSNorm (1.28K parameters)
- visual.blocks.11.attn: Qwen2_5_VLVisionSdpaAttention (6.56M parameters)
- visual.blocks.11.attn.qkv: Linear (4.92M parameters)
- visual.blocks.11.attn.proj: Linear (1.64M parameters)
- visual.blocks.11.mlp: Qwen2_5_VLMLP (13.14M parameters)
- visual.blocks.11.mlp.gate_proj: Linear (4.38M parameters)
- visual.blocks.11.mlp.up_proj: Linear (4.38M parameters)
- visual.blocks.11.mlp.down_proj: Linear (4.38M parameters)
- visual.blocks.11.mlp.act_fn: SiLU (0 parameters)
- visual.blocks.12: Qwen2_5_VLVisionBlock (19.70M parameters)
- visual.blocks.12.norm1: Qwen2RMSNorm (1.28K parameters)
- visual.blocks.12.norm2: Qwen2RMSNorm (1.28K parameters)
- visual.blocks.12.attn: Qwen2_5_VLVisionSdpaAttention (6.56M parameters)
- visual.blocks.12.attn.qkv: Linear (4.92M parameters)
- visual.blocks.12.attn.proj: Linear (1.64M parameters)
- visual.blocks.12.mlp: Qwen2_5_VLMLP (13.14M parameters)
- visual.blocks.12.mlp.gate_proj: Linear (4.38M parameters)
- visual.blocks.12.mlp.up_proj: Linear (4.38M parameters)
- visual.blocks.12.mlp.down_proj: Linear (4.38M parameters)
- visual.blocks.12.mlp.act_fn: SiLU (0 parameters)
- visual.blocks.13: Qwen2_5_VLVisionBlock (19.70M parameters)
- visual.blocks.13.norm1: Qwen2RMSNorm (1.28K parameters)
- visual.blocks.13.norm2: Qwen2RMSNorm (1.28K parameters)
- visual.blocks.13.attn: Qwen2_5_VLVisionSdpaAttention (6.56M parameters)
- visual.blocks.13.attn.qkv: Linear (4.92M parameters)
- visual.blocks.13.attn.proj: Linear (1.64M parameters)
- visual.blocks.13.mlp: Qwen2_5_VLMLP (13.14M parameters)
- visual.blocks.13.mlp.gate_proj: Linear (4.38M parameters)
- visual.blocks.13.mlp.up_proj: Linear (4.38M parameters)
- visual.blocks.13.mlp.down_proj: Linear (4.38M parameters)
- visual.blocks.13.mlp.act_fn: SiLU (0 parameters)
- visual.blocks.14: Qwen2_5_VLVisionBlock (19.70M parameters)
- visual.blocks.14.norm1: Qwen2RMSNorm (1.28K parameters)
- visual.blocks.14.norm2: Qwen2RMSNorm (1.28K parameters)
- visual.blocks.14.attn: Qwen2_5_VLVisionSdpaAttention (6.56M parameters)
- visual.blocks.14.attn.qkv: Linear (4.92M parameters)
- visual.blocks.14.attn.proj: Linear (1.64M parameters)
- visual.blocks.14.mlp: Qwen2_5_VLMLP (13.14M parameters)
- visual.blocks.14.mlp.gate_proj: Linear (4.38M parameters)
- visual.blocks.14.mlp.up_proj: Linear (4.38M parameters)
- visual.blocks.14.mlp.down_proj: Linear (4.38M parameters)
- visual.blocks.14.mlp.act_fn: SiLU (0 parameters)
- visual.blocks.15: Qwen2_5_VLVisionBlock (19.70M parameters)
- visual.blocks.15.norm1: Qwen2RMSNorm (1.28K parameters)
- visual.blocks.15.norm2: Qwen2RMSNorm (1.28K parameters)
- visual.blocks.15.attn: Qwen2_5_VLVisionSdpaAttention (6.56M parameters)
- visual.blocks.15.attn.qkv: Linear (4.92M parameters)
- visual.blocks.15.attn.proj: Linear (1.64M parameters)
- visual.blocks.15.mlp: Qwen2_5_VLMLP (13.14M parameters)
- visual.blocks.15.mlp.gate_proj: Linear (4.38M parameters)
- visual.blocks.15.mlp.up_proj: Linear (4.38M parameters)
- visual.blocks.15.mlp.down_proj: Linear (4.38M parameters)
- visual.blocks.15.mlp.act_fn: SiLU (0 parameters)
- visual.blocks.16: Qwen2_5_VLVisionBlock (19.70M parameters)
- visual.blocks.16.norm1: Qwen2RMSNorm (1.28K parameters)
- visual.blocks.16.norm2: Qwen2RMSNorm (1.28K parameters)
- visual.blocks.16.attn: Qwen2_5_VLVisionSdpaAttention (6.56M parameters)
- visual.blocks.16.attn.qkv: Linear (4.92M parameters)
- visual.blocks.16.attn.proj: Linear (1.64M parameters)
- visual.blocks.16.mlp: Qwen2_5_VLMLP (13.14M parameters)
- visual.blocks.16.mlp.gate_proj: Linear (4.38M parameters)
- visual.blocks.16.mlp.up_proj: Linear (4.38M parameters)
- visual.blocks.16.mlp.down_proj: Linear (4.38M parameters)
- visual.blocks.16.mlp.act_fn: SiLU (0 parameters)
- visual.blocks.17: Qwen2_5_VLVisionBlock (19.70M parameters)
- visual.blocks.17.norm1: Qwen2RMSNorm (1.28K parameters)
- visual.blocks.17.norm2: Qwen2RMSNorm (1.28K parameters)
- visual.blocks.17.attn: Qwen2_5_VLVisionSdpaAttention (6.56M parameters)
- visual.blocks.17.attn.qkv: Linear (4.92M parameters)
- visual.blocks.17.attn.proj: Linear (1.64M parameters)
- visual.blocks.17.mlp: Qwen2_5_VLMLP (13.14M parameters)
- visual.blocks.17.mlp.gate_proj: Linear (4.38M parameters)
- visual.blocks.17.mlp.up_proj: Linear (4.38M parameters)
- visual.blocks.17.mlp.down_proj: Linear (4.38M parameters)
- visual.blocks.17.mlp.act_fn: SiLU (0 parameters)
- visual.blocks.18: Qwen2_5_VLVisionBlock (19.70M parameters)
- visual.blocks.18.norm1: Qwen2RMSNorm (1.28K parameters)
- visual.blocks.18.norm2: Qwen2RMSNorm (1.28K parameters)
- visual.blocks.18.attn: Qwen2_5_VLVisionSdpaAttention (6.56M parameters)
- visual.blocks.18.attn.qkv: Linear (4.92M parameters)
- visual.blocks.18.attn.proj: Linear (1.64M parameters)
- visual.blocks.18.mlp: Qwen2_5_VLMLP (13.14M parameters)
- visual.blocks.18.mlp.gate_proj: Linear (4.38M parameters)
- visual.blocks.18.mlp.up_proj: Linear (4.38M parameters)
- visual.blocks.18.mlp.down_proj: Linear (4.38M parameters)
- visual.blocks.18.mlp.act_fn: SiLU (0 parameters)
- visual.blocks.19: Qwen2_5_VLVisionBlock (19.70M parameters)
- visual.blocks.19.norm1: Qwen2RMSNorm (1.28K parameters)
- visual.blocks.19.norm2: Qwen2RMSNorm (1.28K parameters)
- visual.blocks.19.attn: Qwen2_5_VLVisionSdpaAttention (6.56M parameters)
- visual.blocks.19.attn.qkv: Linear (4.92M parameters)
- visual.blocks.19.attn.proj: Linear (1.64M parameters)
- visual.blocks.19.mlp: Qwen2_5_VLMLP (13.14M parameters)
- visual.blocks.19.mlp.gate_proj: Linear (4.38M parameters)
- visual.blocks.19.mlp.up_proj: Linear (4.38M parameters)
- visual.blocks.19.mlp.down_proj: Linear (4.38M parameters)
- visual.blocks.19.mlp.act_fn: SiLU (0 parameters)
- visual.blocks.20: Qwen2_5_VLVisionBlock (19.70M parameters)
- visual.blocks.20.norm1: Qwen2RMSNorm (1.28K parameters)
- visual.blocks.20.norm2: Qwen2RMSNorm (1.28K parameters)
- visual.blocks.20.attn: Qwen2_5_VLVisionSdpaAttention (6.56M parameters)
- visual.blocks.20.attn.qkv: Linear (4.92M parameters)
- visual.blocks.20.attn.proj: Linear (1.64M parameters)
- visual.blocks.20.mlp: Qwen2_5_VLMLP (13.14M parameters)
- visual.blocks.20.mlp.gate_proj: Linear (4.38M parameters)
- visual.blocks.20.mlp.up_proj: Linear (4.38M parameters)
- visual.blocks.20.mlp.down_proj: Linear (4.38M parameters)
- visual.blocks.20.mlp.act_fn: SiLU (0 parameters)
- visual.blocks.21: Qwen2_5_VLVisionBlock (19.70M parameters)
- visual.blocks.21.norm1: Qwen2RMSNorm (1.28K parameters)
- visual.blocks.21.norm2: Qwen2RMSNorm (1.28K parameters)
- visual.blocks.21.attn: Qwen2_5_VLVisionSdpaAttention (6.56M parameters)
- visual.blocks.21.attn.qkv: Linear (4.92M parameters)
- visual.blocks.21.attn.proj: Linear (1.64M parameters)
- visual.blocks.21.mlp: Qwen2_5_VLMLP (13.14M parameters)
- visual.blocks.21.mlp.gate_proj: Linear (4.38M parameters)
- visual.blocks.21.mlp.up_proj: Linear (4.38M parameters)
- visual.blocks.21.mlp.down_proj: Linear (4.38M parameters)
- visual.blocks.21.mlp.act_fn: SiLU (0 parameters)
- visual.blocks.22: Qwen2_5_VLVisionBlock (19.70M parameters)
- visual.blocks.22.norm1: Qwen2RMSNorm (1.28K parameters)
- visual.blocks.22.norm2: Qwen2RMSNorm (1.28K parameters)
- visual.blocks.22.attn: Qwen2_5_VLVisionSdpaAttention (6.56M parameters)
- visual.blocks.22.attn.qkv: Linear (4.92M parameters)
- visual.blocks.22.attn.proj: Linear (1.64M parameters)
- visual.blocks.22.mlp: Qwen2_5_VLMLP (13.14M parameters)
- visual.blocks.22.mlp.gate_proj: Linear (4.38M parameters)
- visual.blocks.22.mlp.up_proj: Linear (4.38M parameters)
- visual.blocks.22.mlp.down_proj: Linear (4.38M parameters)
- visual.blocks.22.mlp.act_fn: SiLU (0 parameters)
- visual.blocks.23: Qwen2_5_VLVisionBlock (19.70M parameters)
- visual.blocks.23.norm1: Qwen2RMSNorm (1.28K parameters)
- visual.blocks.23.norm2: Qwen2RMSNorm (1.28K parameters)
- visual.blocks.23.attn: Qwen2_5_VLVisionSdpaAttention (6.56M parameters)
- visual.blocks.23.attn.qkv: Linear (4.92M parameters)
- visual.blocks.23.attn.proj: Linear (1.64M parameters)
- visual.blocks.23.mlp: Qwen2_5_VLMLP (13.14M parameters)
- visual.blocks.23.mlp.gate_proj: Linear (4.38M parameters)
- visual.blocks.23.mlp.up_proj: Linear (4.38M parameters)
- visual.blocks.23.mlp.down_proj: Linear (4.38M parameters)
- visual.blocks.23.mlp.act_fn: SiLU (0 parameters)
- visual.blocks.24: Qwen2_5_VLVisionBlock (19.70M parameters)
- visual.blocks.24.norm1: Qwen2RMSNorm (1.28K parameters)
- visual.blocks.24.norm2: Qwen2RMSNorm (1.28K parameters)
- visual.blocks.24.attn: Qwen2_5_VLVisionSdpaAttention (6.56M parameters)
- visual.blocks.24.attn.qkv: Linear (4.92M parameters)
- visual.blocks.24.attn.proj: Linear (1.64M parameters)
- visual.blocks.24.mlp: Qwen2_5_VLMLP (13.14M parameters)
- visual.blocks.24.mlp.gate_proj: Linear (4.38M parameters)
- visual.blocks.24.mlp.up_proj: Linear (4.38M parameters)
- visual.blocks.24.mlp.down_proj: Linear (4.38M parameters)
- visual.blocks.24.mlp.act_fn: SiLU (0 parameters)
- visual.blocks.25: Qwen2_5_VLVisionBlock (19.70M parameters)
- visual.blocks.25.norm1: Qwen2RMSNorm (1.28K parameters)
- visual.blocks.25.norm2: Qwen2RMSNorm (1.28K parameters)
- visual.blocks.25.attn: Qwen2_5_VLVisionSdpaAttention (6.56M parameters)
- visual.blocks.25.attn.qkv: Linear (4.92M parameters)
- visual.blocks.25.attn.proj: Linear (1.64M parameters)
- visual.blocks.25.mlp: Qwen2_5_VLMLP (13.14M parameters)
- visual.blocks.25.mlp.gate_proj: Linear (4.38M parameters)
- visual.blocks.25.mlp.up_proj: Linear (4.38M parameters)
- visual.blocks.25.mlp.down_proj: Linear (4.38M parameters)
- visual.blocks.25.mlp.act_fn: SiLU (0 parameters)
- visual.blocks.26: Qwen2_5_VLVisionBlock (19.70M parameters)
- visual.blocks.26.norm1: Qwen2RMSNorm (1.28K parameters)
- visual.blocks.26.norm2: Qwen2RMSNorm (1.28K parameters)
- visual.blocks.26.attn: Qwen2_5_VLVisionSdpaAttention (6.56M parameters)
- visual.blocks.26.attn.qkv: Linear (4.92M parameters)
- visual.blocks.26.attn.proj: Linear (1.64M parameters)
- visual.blocks.26.mlp: Qwen2_5_VLMLP (13.14M parameters)
- visual.blocks.26.mlp.gate_proj: Linear (4.38M parameters)
- visual.blocks.26.mlp.up_proj: Linear (4.38M parameters)
- visual.blocks.26.mlp.down_proj: Linear (4.38M parameters)
- visual.blocks.26.mlp.act_fn: SiLU (0 parameters)
- visual.blocks.27: Qwen2_5_VLVisionBlock (19.70M parameters)
- visual.blocks.27.norm1: Qwen2RMSNorm (1.28K parameters)
- visual.blocks.27.norm2: Qwen2RMSNorm (1.28K parameters)
- visual.blocks.27.attn: Qwen2_5_VLVisionSdpaAttention (6.56M parameters)
- visual.blocks.27.attn.qkv: Linear (4.92M parameters)
- visual.blocks.27.attn.proj: Linear (1.64M parameters)
- visual.blocks.27.mlp: Qwen2_5_VLMLP (13.14M parameters)
- visual.blocks.27.mlp.gate_proj: Linear (4.38M parameters)
- visual.blocks.27.mlp.up_proj: Linear (4.38M parameters)
- visual.blocks.27.mlp.down_proj: Linear (4.38M parameters)
- visual.blocks.27.mlp.act_fn: SiLU (0 parameters)
- visual.blocks.28: Qwen2_5_VLVisionBlock (19.70M parameters)
- visual.blocks.28.norm1: Qwen2RMSNorm (1.28K parameters)
- visual.blocks.28.norm2: Qwen2RMSNorm (1.28K parameters)
- visual.blocks.28.attn: Qwen2_5_VLVisionSdpaAttention (6.56M parameters)
- visual.blocks.28.attn.qkv: Linear (4.92M parameters)
- visual.blocks.28.attn.proj: Linear (1.64M parameters)
- visual.blocks.28.mlp: Qwen2_5_VLMLP (13.14M parameters)
- visual.blocks.28.mlp.gate_proj: Linear (4.38M parameters)
- visual.blocks.28.mlp.up_proj: Linear (4.38M parameters)
- visual.blocks.28.mlp.down_proj: Linear (4.38M parameters)
- visual.blocks.28.mlp.act_fn: SiLU (0 parameters)
- visual.blocks.29: Qwen2_5_VLVisionBlock (19.70M parameters)
- visual.blocks.29.norm1: Qwen2RMSNorm (1.28K parameters)
- visual.blocks.29.norm2: Qwen2RMSNorm (1.28K parameters)
- visual.blocks.29.attn: Qwen2_5_VLVisionSdpaAttention (6.56M parameters)
- visual.blocks.29.attn.qkv: Linear (4.92M parameters)
- visual.blocks.29.attn.proj: Linear (1.64M parameters)
- visual.blocks.29.mlp: Qwen2_5_VLMLP (13.14M parameters)
- visual.blocks.29.mlp.gate_proj: Linear (4.38M parameters)
- visual.blocks.29.mlp.up_proj: Linear (4.38M parameters)
- visual.blocks.29.mlp.down_proj: Linear (4.38M parameters)
- visual.blocks.29.mlp.act_fn: SiLU (0 parameters)
- visual.blocks.30: Qwen2_5_VLVisionBlock (19.70M parameters)
- visual.blocks.30.norm1: Qwen2RMSNorm (1.28K parameters)
- visual.blocks.30.norm2: Qwen2RMSNorm (1.28K parameters)
- visual.blocks.30.attn: Qwen2_5_VLVisionSdpaAttention (6.56M parameters)
- visual.blocks.30.attn.qkv: Linear (4.92M parameters)
- visual.blocks.30.attn.proj: Linear (1.64M parameters)
- visual.blocks.30.mlp: Qwen2_5_VLMLP (13.14M parameters)
- visual.blocks.30.mlp.gate_proj: Linear (4.38M parameters)
- visual.blocks.30.mlp.up_proj: Linear (4.38M parameters)
- visual.blocks.30.mlp.down_proj: Linear (4.38M parameters)
- visual.blocks.30.mlp.act_fn: SiLU (0 parameters)
- visual.blocks.31: Qwen2_5_VLVisionBlock (19.70M parameters)
- visual.blocks.31.norm1: Qwen2RMSNorm (1.28K parameters)
- visual.blocks.31.norm2: Qwen2RMSNorm (1.28K parameters)
- visual.blocks.31.attn: Qwen2_5_VLVisionSdpaAttention (6.56M parameters)
- visual.blocks.31.attn.qkv: Linear (4.92M parameters)
- visual.blocks.31.attn.proj: Linear (1.64M parameters)
- visual.blocks.31.mlp: Qwen2_5_VLMLP (13.14M parameters)
- visual.blocks.31.mlp.gate_proj: Linear (4.38M parameters)
- visual.blocks.31.mlp.up_proj: Linear (4.38M parameters)
- visual.blocks.31.mlp.down_proj: Linear (4.38M parameters)
- visual.blocks.31.mlp.act_fn: SiLU (0 parameters)
- visual.merger: Qwen2_5_VLPatchMerger (44.57M parameters)
- visual.merger.ln_q: Qwen2RMSNorm (1.28K parameters)
- visual.merger.mlp: Sequential (44.57M parameters)
- visual.merger.mlp.0: Linear (26.22M parameters)
- visual.merger.mlp.1: GELU (0 parameters)
- visual.merger.mlp.2: Linear (18.35M parameters)

Model config:
- vision_config: Qwen2_5_VLVisionConfig {
  "depth": 32,
  "fullatt_block_indexes": [
    7,
    15,
    23,
    31
  ],
  "hidden_act": "silu",
  "hidden_size": 1280,
  "in_channels": 3,
  "in_chans": 3,
  "intermediate_size": 3420,
  "model_type": "qwen2_5_vl",
  "num_heads": 16,
  "out_hidden_size": 3584,
  "patch_size": 14,
  "spatial_merge_size": 2,
  "spatial_patch_size": 14,
  "temporal_patch_size": 2,
  "tokens_per_second": 2,
  "torch_dtype": "float16",
  "transformers_version": "4.51.3",
  "window_size": 112
}

- vocab_size: 152064
- max_position_embeddings: 128000
- hidden_size: 3584
- intermediate_size: 18944
- num_hidden_layers: 28
- num_attention_heads: 28
- use_sliding_window: False
- sliding_window: 32768
- max_window_layers: 28
- num_key_value_heads: 4
- hidden_act: silu
- initializer_range: 0.02
- rms_norm_eps: 1e-06
- use_cache: True
- rope_theta: 1000000.0
- attention_dropout: 0.0
- rope_scaling: {'type': 'default', 'mrope_section': [16, 24, 24], 'rope_type': 'default'}
- return_dict: True
- output_hidden_states: False
- output_attentions: False
- torchscript: False
- torch_dtype: torch.float16
- use_bfloat16: False
- tf_legacy_loss: False
- pruned_heads: {}
- tie_word_embeddings: False
- chunk_size_feed_forward: 0
- is_encoder_decoder: False
- is_decoder: False
- cross_attention_hidden_size: None
- add_cross_attention: False
- tie_encoder_decoder: False
- max_length: 20
- min_length: 0
- do_sample: False
- early_stopping: False
- num_beams: 1
- num_beam_groups: 1
- diversity_penalty: 0.0
- temperature: 1.0
- top_k: 50
- top_p: 1.0
- typical_p: 1.0
- repetition_penalty: 1.0
- length_penalty: 1.0
- no_repeat_ngram_size: 0
- encoder_no_repeat_ngram_size: 0
- bad_words_ids: None
- num_return_sequences: 1
- output_scores: False
- return_dict_in_generate: False
- forced_bos_token_id: None
- forced_eos_token_id: None
- remove_invalid_values: False
- exponential_decay_length_penalty: None
- suppress_tokens: None
- begin_suppress_tokens: None
- architectures: ['Qwen2_5_VLForConditionalGeneration']
- finetuning_task: None
- id2label: {0: 'LABEL_0', 1: 'LABEL_1'}
- label2id: {'LABEL_0': 0, 'LABEL_1': 1}
- tokenizer_class: None
- prefix: None
- bos_token_id: 151643
- pad_token_id: None
- eos_token_id: 151645
- sep_token_id: None
- decoder_start_token_id: None
- task_specific_params: None
- problem_type: None
- transformers_version: 4.41.2
- vision_start_token_id: 151652
- vision_end_token_id: 151653
- vision_token_id: 151654
- image_token_id: 151655
- video_token_id: 151656
- model_type: qwen2_5_vl

Analysis complete!
