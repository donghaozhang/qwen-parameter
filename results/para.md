Loading Qwen/Qwen2.5-VL-7B-Instruct for merger analysis...
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

Vision-Language Merger Flow:

    ┌───────────────┐      ┌──────────────┐
    │ Vision Model  │      │ Text Tokens  │
    └───────┬───────┘      └──────┬───────┘
            │                     │
            ▼                     ▼
    ┌───────────────┐      ┌──────────────┐
    │Vision Features│      │Token Embeddings│
    └───────┬───────┘      └──────┬───────┘
            │                     │
            ▼                     │
    ┌───────────────┐             │
    │Vision Projection│            │
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
    

Overall Model Architecture:
1. visual: Qwen2_5_VisionTransformerPretrainedModel (676.55M params)
2. model: Qwen2_5_VLModel (7.07B params)
3. lm_head: Linear (545.00M params)

Analysis complete!
