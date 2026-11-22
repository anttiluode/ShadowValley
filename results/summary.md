| cell | solved (NMSE<0.01) | <0.001 | read: at / neg-at / away / other | median NMSE | failed look-away runs: median m_eff, beta, NMSE, Neon curve at m |
|---|---|---|---|---|---|
| A' attn, scalar cue | softmax | 9/20 | 9/20 | 14 / 0 / 6 / 0 | 0.0400 | 6 runs: m 4.69, β -1.31, NMSE 0.851, curve 0.798 |
| A' attn, scalar cue | differential | 13/20 | 12/20 | 10 / 7 / 1 / 2 | 0.0009 | 1 runs: m 1.65, β -0.57, NMSE 0.121, curve 0.963 |
| A' attn, scalar cue | cog | 9/20 | 9/20 | 10 / 7 / 1 / 2 | 0.0666 | 1 runs: m 1.47, β -0.48, NMSE 0.350, curve 0.968 |
| A' attn, scalar cue | flashlight | 20/20 | 20/20 | 20 / 0 / 0 / 0 | 0.0005 | – |
| A' attn, scalar cue | softmax, 2 heads | 10/20 | 8/20 | 16 / 0 / 3 / 1 | 0.0153 | 3 runs: m 3.70, β -0.73, NMSE 0.653, curve 0.877 |
| A' attn, scalar cue | softmax, 4x longer | 9/20 | 9/20 | 14 / 0 / 6 / 0 | 0.0283 | 6 runs: m 4.64, β -1.73, NMSE 0.834, curve 0.802 |
| A' attn, scalar cue | softmax, iid detail | 20/20 | 20/20 | 20 / 0 / 0 / 0 | 0.0002 | – |
| A' attn, scalar cue | softmax, reversed | 11/20 | 10/20 | 15 / 0 / 5 / 0 | 0.0014 | 5 runs: m 4.72, β 5.17, NMSE 0.804, curve 0.794 |
| A' attn, scalar cue | flashlight, reversed | 20/20 | 20/20 | 20 / 0 / 0 / 0 | 0.0005 | – |
| A' attn, scalar cue | signed teacher, reversed | 1/20 | 1/20 | 12 / 0 / 8 / 0 | 0.6297 | 8 runs: m 1.11, β -0.60, NMSE 0.785, curve 0.977 |
| A' attn, scalar cue | shuffled teacher | 8/20 | 7/20 | 12 / 0 / 7 / 1 | 0.2333 | 7 runs: m 4.37, β -1.21, NMSE 0.823, curve 0.828 |
| A' attn, scalar cue | random-table teacher | 0/20 | 0/20 | 9 / 0 / 11 / 0 | 0.8514 | 11 runs: m 2.07, β -0.15, NMSE 0.932, curve 0.950 |
| B' block, scalar cue | softmax | 5/20 | 1/20 | 19 / 0 / 0 / 1 | 0.0891 | – |
| B' block, scalar cue | differential | 6/20 | 1/20 | 7 / 7 / 0 / 6 | 0.0158 | – |
| B' block, scalar cue | cog | 9/20 | 0/20 | 14 / 4 / 1 / 1 | 0.0152 | – |
| B' block, scalar cue | flashlight | 9/20 | 3/20 | 20 / 0 / 0 / 0 | 0.0137 | – |
| B' block, scalar cue | random-table teacher | 0/20 | 0/20 | 5 / 0 / 15 / 0 | 0.7272 | 15 runs: m 2.33, β -0.17, NMSE 0.852, curve 0.941 |
| B' block, scalar cue | softmax, 2 heads | 14/20 | 4/20 | 20 / 0 / 0 / 0 | 0.0019 | – |
| A attn, one-hot cue | softmax | 14/20 | 12/20 | 12 / 0 / 8 / 0 | 0.0006 | 3 runs: m 6.53, β -7.61, NMSE 0.017, curve 0.367 |
| A attn, one-hot cue | differential | 20/20 | 18/20 | 8 / 8 / 3 / 1 | 0.0000 | – |
| A attn, one-hot cue | cog | 18/20 | 11/20 | 14 / 6 / 0 / 0 | 0.0001 | – |
| A attn, one-hot cue | flashlight | 20/20 | 20/20 | 20 / 0 / 0 / 0 | 0.0000 | – |
| B block, one-hot cue | softmax | 17/20 | 15/20 | 16 / 0 / 4 / 0 | 0.0001 | 2 runs: m 6.92, β -5.95, NMSE 0.013, curve 0.087 |
| B block, one-hot cue | differential | 20/20 | 19/20 | 11 / 9 / 0 / 0 | 0.0001 | – |
| B block, one-hot cue | cog | 20/20 | 14/20 | 11 / 9 / 0 / 0 | 0.0004 | – |
| B block, one-hot cue | flashlight | 20/20 | 20/20 | 20 / 0 / 0 / 0 | 0.0001 | – |
| C self-mean | softmax | 14/20 | 4/20 | 19 / 0 / 1 / 0 | 0.0048 | – |
| C self-mean | differential | 12/20 | 0/20 | 10 / 5 / 0 / 5 | 0.0072 | – |
| C self-mean | cog | 6/20 | 0/20 | 13 / 6 / 0 / 1 | 0.0125 | – |
| C self-mean | flashlight | 16/20 | 11/20 | 19 / 0 / 1 / 0 | 0.0008 | 1 runs: m 1.07, β -0.14, NMSE 0.625, curve 0.978 |
