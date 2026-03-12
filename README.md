# forgeformer

a minimal, weights-forged-by-hand transformer that adds two two-digit numbers — built to be fully inspectable at every internal step.

© aritro 'sortira' shome · [twitter](https://x.com/silicognition)

try out the demo to look inside the brain of the model [here](https://aritro.is-a.dev/forgeformer)

read the detailed blog post (docs basically) [here](https://silicognition.is-a.dev/post2.html)

see the explanation video by me [here](https://youtu.be/FnKLQJ5EIZ4)

---

## overview

forgeformer is a stripped-down transformer implementation designed for mechanistic interpretability. rather than training a model and reverse-engineering its behavior, the weights are constructed analytically — each matrix has a clear, human-understandable role in the computation. the goal is to make the internal mechanics of attention as transparent as possible.

the model takes two numbers (each between 1 and 99), tokenizes them into four digit tokens plus an [eos] token, and returns the sum through two sequential attention layers.

> **note:** not all standard transformer components are present. there are no feed-forward layers, no layer normalization, and no learned embeddings. the embedding and unembedding stages use deliberate "hacks" to keep the focus on the attention mechanism itself.

---

## architecture

| property | value |
|---|---|
| model dimension (`d_model`) | 3 |
| number of attention heads | 1 |
| number of layers | 2 |
| input range | 1 – 99 (both operands) |
| max result | 198 (hundreds digit supported via carry hack) |

### tokenization

each input pair `(a, b)` is expanded into a 5-token sequence:

```
[ tens(a),  units(a),  tens(b),  units(b),  [eos] ]
```

### embedding

each token maps to a 3-dimensional vector:

| dimension | meaning |
|---|---|
| `dim[0]` | accumulator — collects the running sum |
| `dim[1]` | digit value |
| `dim[2]` | positional sign: `+1` for even indices and eos, `−1` for odd indices |

### layer 1 — units place

the first attention head uses `wq1`, `wk1`, `wv1` to aggregate the units digits of both operands into the eos token's `dim[0]`. the positional sign encoding in `dim[2]` ensures the eos token attends to the relevant tokens.

### layer 2 — tens place

the second attention head uses `wq2`, `wk2`, `wv2` to aggregate the tens digits into the eos token's `dim[1]`.

### readout

the final answer is read from the eos token of the output matrix `x3`:

```
units_sum = x3[eos, 0]
tens_sum  = x3[eos, 1]
carry     = 1 if units_sum >= 10 else 0
result    = (tens_sum + carry) * 10 + (units_sum % 10)
```

---

## weight matrices

```python
wq1 = [[0,0,0], [0,0,0], [0,0, 10]]
wk1 = [[0,0,0], [0,0,0], [0,0,-10]]
wv1 = [[0,0,0], [2,0,0], [0,0,  0]]

wq2 = [[0,0,0], [0,0,0], [0,0, 10]]
wk2 = [[0,0,0], [0,0,0], [0,0, 10]]
wv2 = [[0,0,0], [0,3,0], [0,0,  0]]
```

the large values (±10) in `dim[2]` of the query/key matrices sharpen the attention distribution toward near-argmax behavior via the softmax. the value matrices route the digit information (`dim[1]`) into the accumulator slot (`dim[0]` or `dim[1]`) of the eos token.

---

## files

| file | description |
|---|---|
| `forgeformer.ipynb` | core python implementation using numpy; runs the forward pass and exposes all intermediate states |
| `forgeformer.html` | self-contained browser demo; runs the model in javascript and visualizes every matrix, attention heatmap, and activation in real time |

---

## usage

### python (notebook)

```python
ans, states = forgeformer([3, 7, 4, 6, -1])  # computes 37 + 46
print(ans)  # 83
```

`states` is a dictionary containing all intermediate matrices: `x1`, `q1`, `k1`, `v1`, `kq1`, `attn1`, `x2`, `q2`, `k2`, `v2`, `kq2`, `attn2`, and `output`.

### browser demo

open `forgeformer.html` in any modern browser. enter two numbers and click **run model** to step through the full forward pass with annotated visualizations of each stage.

---

## dependencies

- python notebook: `numpy`
- browser demo: none — fully self-contained, no external libraries

---

## limitations

- inputs must be integers between 1 and 99
- the sum may slightly overflow into the hundreds digit, handled by the manual carry logic rather than a learned readout head
- the embedding and unembedding are not learned; they are fixed by design to isolate the attention mechanism
