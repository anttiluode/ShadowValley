| cell | solved | attention says (at / away) | causal read of solved runs: direct / complement / mixed | solved, attention "at" but causal complement | median noise gain vs ideal direct (solved) |
|---|---|---|---|---|---|
| A attn, one-hot cue | softmax | 14/20 | 9 / 5 | 9 / 5 / 0 | 0 | 1.02 |
| A attn, one-hot cue | flashlight | 20/20 | 20 / 0 | 20 / 0 / 0 | 0 | 1.00 |
| B block, one-hot cue | softmax | 17/20 | 15 / 2 | 15 / 2 / 0 | 0 | 1.00 |
| B block, one-hot cue | differential | 20/20 | 11 / 0 | 20 / 0 / 0 | 0 | 0.97 |
| B block, one-hot cue | cog | 20/20 | 11 / 0 | 20 / 0 / 0 | 0 | 1.01 |
| B' block, scalar cue | softmax | 5/20 | 5 / 0 | 5 / 0 / 0 | 0 | 1.01 |
| B' block, scalar cue | softmax, 2 heads | 14/20 | 14 / 0 | 13 / 0 / 1 | 0 | 1.01 |
| A' attn, scalar cue | softmax | 9/20 | 9 / 0 | 9 / 0 / 0 | 0 | 1.01 |
| C self-mean | softmax | 14/20 | 13 / 1 | 0 / 0 / 14 | 0 | 2.19 |
