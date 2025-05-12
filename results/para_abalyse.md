Analyzing Qwen2.5-VL model parameters...
Loading Qwen/Qwen2.5-VL-7B-Instruct for parameter analysis...
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

Model architecture overview:
visual: Qwen2_5_VisionTransformerPretrainedModel
model: Qwen2_5_VLModel
lm_head: Linear

Analysis complete!
