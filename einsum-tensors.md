In this article, we provide a cheatsheet for using [einsum](https://pytorch.org/docs/stable/generated/torch.einsum.html) to perform common operations in deep learning. We start from simple operations like rowsum that serve as building blocks. And gradually build up to defining multi-head attention with einsum. Where applicable I compare with more common ways of doing the same operations in PyTorch and the problem with these ways.

This post came about as I was implementing self-attention and comparing against llama3's open sourced code. llama3 code has model and data parallel to enable training large models on multiple GPUs. Like everything in PyTorch, there are mutiple ways to implement self-attention. See also, [this efficient implementation](https://github.com/facebookresearch/xformers/blob/95f085abc3ba8cfaa2527250b8e274a95b10f7fe/xformers/components/attention/core.py#L208). Here I focus on readability and exposition.

I like einsum because the syntax is pithy and self-documenting. I can easily visualize the shapes of the input and output tensors leading to easier coding and debugging. See these [visualizations](https://www.tensors.net/tutorial-1) and the [original proposal](https://nlp.seas.harvard.edu/NamedTensor) for more discussion.


```python
import torch
import torch.nn.functional as F
import math

a = torch.arange(6).reshape(2, 3)
a
```




    tensor([[0, 1, 2],
            [3, 4, 5]])



## Transpose


```python
torch.einsum('ij->ji', a)
```




    tensor([[0, 3],
            [1, 4],
            [2, 5]])



## Sum


```python
torch.einsum('ij->', a)
```




    tensor(15)




```python

```

## Rowsum (sum all rows, compress along rows) -> get a vector of length j


```python
torch.einsum('ij->j', a)
```




    tensor([3, 5, 7])




```python
torch.sum(a, dim=0)
```




    tensor([3, 5, 7])




```python

```

## Colsum (sum all cols, compress along cols) -> get a vector of length i


```python
torch.einsum('ij->i', a)
```




    tensor([ 3, 12])




```python
torch.sum(a, dim=1)
```




    tensor([ 3, 12])



## Matrix vector multiplication

This is a weighted sum of all columns, where column i is weighted by b[i], and similar to colsum compress all columns


```python
b = torch.arange(3)
b
```




    tensor([0, 1, 2])




```python
torch.einsum('ij, j ->i', a, b)
```




    tensor([ 5, 14])



## Matrix-Matrix multiplication

The index missing in 'ij, jk -> ik' on the output-string is j. This means we sum across j. To help visualize, do 'ij,j->i', k times. 


```python
c = torch.arange(12).reshape(3, 4)
c.shape
```




    torch.Size([3, 4])




```python
torch.einsum('ij, jk -> ik', [a, c])
```




    tensor([[20, 23, 26, 29],
            [56, 68, 80, 92]])



## Dot product


```python
#b.b
torch.einsum('j,j->',[b,b])
```




    tensor(5)



## softmax($w^T M_t$)

It is not obvious from reading this single line below what probs shape is. One has to rely either on comments or look at the input tensor dimensions, determine what the output shape is, and then remember it. Instead with einsum, it is clear the distribution is across j output nodes.


```python
w = torch.randn(5)
M_t = torch.randn(5, 7)
probs = F.softmax(w.T.matmul(M_t), dim=0) 

#einsum implementation
probs_e = F.softmax(torch.einsum("i,ij->j",[w,M_t]), dim=0)
```

## $W h$ -  matrix vector multiplication for a batch of vectors

Caveat: The goal here is to show equivalence of outputs from einsum, Linear and matmul. In practice, one would rather use efficient library implementations like Linear, especially when used as part of a larger network. 


```python
batch_size = 2
ip_dim = 3
op_dim = 5

h = torch.randn(batch_size, ip_dim) #a batch of 3-dimensional vectors
model = torch.nn.Linear(ip_dim, op_dim, bias = False)
w = model.weight
w.shape
```




    torch.Size([5, 3])




```python
batch_output = model(h)
batch_output
```




    tensor([[-0.1363, -0.3708,  0.4023, -0.0985, -0.0302],
            [-0.4795, -0.7544,  0.3758, -0.6922,  0.3838]], grad_fn=<MmBackward0>)




```python
torch.matmul(h, w.T)
```




    tensor([[-0.1363, -0.3708,  0.4023, -0.0985, -0.0302],
            [-0.4795, -0.7544,  0.3758, -0.6922,  0.3838]], grad_fn=<MmBackward0>)




```python
torch.einsum("ij,jk->ik", [h, w.T])
```




    tensor([[-0.1363, -0.3708,  0.4023, -0.0985, -0.0302],
            [-0.4795, -0.7544,  0.3758, -0.6922,  0.3838]],
           grad_fn=<ViewBackward0>)



## Linear projection for a batch of a sequence of vectors

Taking the above one step further. What if each "example" in our dataset is a sequence of items, and each item is a vector. e.g., In transformer-based NLP models, a sentence is a sequence of tokens (roughly) corresponding to subwords. Each token has a learned embedding. In self-attention mechanism, we first do a linear projection of the sequence of query, key and value token embeddings using $W_q$, $W_k$ and $W_v$ respectively. In this setting, ip_dim is the embedding dimension, and op_dim the head dimension.


```python
seq_length = 6
sequence_data = torch.randn(batch_size, seq_length, ip_dim)
model(sequence_data)
```




    tensor([[[-2.0220e-01,  1.1954e+00, -6.8526e-01, -1.2182e-01,  1.2170e-01],
             [ 2.8706e-01,  2.3963e-01, -5.1339e-01,  1.6451e-01,  5.5872e-02],
             [ 2.1695e-01, -3.8090e-01,  1.3676e-01,  1.7875e-01, -8.8499e-02],
             [-3.5328e-01,  7.5884e-01, -7.3428e-01, -5.1697e-01,  4.3955e-01],
             [ 5.5502e-01,  6.1513e-01, -9.2788e-01,  4.1818e-01,  1.1290e-03],
             [ 4.5112e-01,  3.7324e-01, -2.3062e-01,  5.7744e-01, -3.0762e-01]],
    
            [[ 2.4067e-01,  2.1685e-02, -5.0118e-01,  2.2909e-02,  1.6912e-01],
             [ 1.8257e-01, -1.0460e+00,  1.9180e-01, -1.1328e-01,  1.7044e-01],
             [-3.3498e-01, -4.3380e-02,  4.6705e-01, -1.6562e-01, -7.1225e-02],
             [-2.3848e-01, -1.9542e-01,  6.2850e-01, -2.2837e-02, -1.8702e-01],
             [-7.6096e-01,  3.5922e-01,  3.5882e-01, -5.7330e-01,  1.3388e-01],
             [ 4.1296e-01,  4.4924e-01,  2.5884e-01,  8.3529e-01, -6.4939e-01]]],
           grad_fn=<UnsafeViewBackward0>)




```python
Q = torch.einsum('ilj,jk->ilk',[sequence_data, w.T]) #output shape [batch size, seq length, op_dim] i.e., [2, 6, 5]
Q
```




    tensor([[[-2.0220e-01,  1.1954e+00, -6.8526e-01, -1.2182e-01,  1.2170e-01],
             [ 2.8706e-01,  2.3963e-01, -5.1339e-01,  1.6451e-01,  5.5872e-02],
             [ 2.1695e-01, -3.8090e-01,  1.3676e-01,  1.7875e-01, -8.8499e-02],
             [-3.5328e-01,  7.5884e-01, -7.3428e-01, -5.1697e-01,  4.3955e-01],
             [ 5.5502e-01,  6.1513e-01, -9.2788e-01,  4.1818e-01,  1.1290e-03],
             [ 4.5112e-01,  3.7324e-01, -2.3062e-01,  5.7744e-01, -3.0762e-01]],
    
            [[ 2.4067e-01,  2.1685e-02, -5.0118e-01,  2.2909e-02,  1.6912e-01],
             [ 1.8257e-01, -1.0460e+00,  1.9180e-01, -1.1328e-01,  1.7044e-01],
             [-3.3498e-01, -4.3380e-02,  4.6705e-01, -1.6562e-01, -7.1225e-02],
             [-2.3848e-01, -1.9542e-01,  6.2850e-01, -2.2837e-02, -1.8702e-01],
             [-7.6096e-01,  3.5922e-01,  3.5882e-01, -5.7330e-01,  1.3388e-01],
             [ 4.1296e-01,  4.4924e-01,  2.5884e-01,  8.3529e-01, -6.4939e-01]]],
           grad_fn=<ViewBackward0>)



## Now consider the self-attention mechanism across multiple attention heads

We want to process the same sequence with multiple attention heads, each first doing a linear projection with corresponding weights. But why do n_head multiplications, when we could do one giant multiplication and use GPUs more efficiently? So, we do a _single_ linear projection into n_head*head_dim, i.e. $4*5=20$. Then, we view results spit across n_heads (4). This allows us to compute scores using softmax independently for each head.


```python
n_heads= 4
head_dim = op_dim
wq = torch.nn.Linear(ip_dim, n_heads*op_dim, bias = False)

wk = torch.nn.Linear(ip_dim, n_heads*op_dim, bias = False)
wv = torch.nn.Linear(ip_dim, n_heads*op_dim, bias = False)
wo = torch.nn.Linear(n_heads*head_dim, ip_dim, bias = False)
```


```python
xq = wq(sequence_data)
xq.shape
```




    torch.Size([2, 6, 20])




```python
xq = xq.view(batch_size, seq_length, n_heads, head_dim)
xq.shape
```




    torch.Size([2, 6, 4, 5])



Similarly, compute keys and values.


```python
keys = wk(sequence_data)
values = wv(sequence_data)

keys = keys.view(batch_size, seq_length, n_heads, head_dim)
values = values.view(batch_size, seq_length, n_heads, head_dim)

# Rearrange tensors so we compute one set of scores per attention-head
xq = xq.transpose(1, 2)  #(bs, n_heads, seqlen, head_dim)
keys = keys.transpose(1, 2)  #(bs, n_heads, seqlen, head_dim)
values = values.transpose(1, 2)  #(bs, n_heads, seqlen, head_dim)
```

Compute scores for all pairs of tokens in the sequence. For matmul between xq and keys to produce a seq_len*seq_len set of scores, we need to transpose keys so it has shape #(bs, n_heads, head_dim, seq_len)


```python
scores = torch.matmul(xq, keys.transpose(2, 3)) / math.sqrt(head_dim)
scores = F.softmax(scores.float(), dim=-1).type_as(xq)
scores.shape
```




    torch.Size([2, 4, 6, 6])



Compute attention-weighted vector of the input sequence


```python
output = torch.matmul(scores, values)  # (bs, n_heads, seqlen, head_dim)
output.shape
```




    torch.Size([2, 4, 6, 5])



Concatenate outputs from multiple heads so we get an output tensor of shape (batch_size, seq_length, n_heads*head_dim) 


```python
output = output.transpose(1, 2).contiguous().view(batch_size, seq_length, -1)
output.shape
```




    torch.Size([2, 6, 20])



And one final linear projection


```python
wo(output).shape
```




    torch.Size([2, 6, 3])



## Einsum implementation of multi-head self-attention

The index that is missing in the output string is the one that is being compressed or summed over. So note in output_e computation below bhlj, bhjd -> bhld. Here we want to compress/sum over j. Perhaps it's easier to use the ellipsis notation, which brings the focus to lj, jd and now we can think again in 2 dimensions.


```python
xqq = wq(sequence_data).view(batch_size, seq_length, n_heads, head_dim)
kk  = wk(sequence_data).view(batch_size, seq_length, n_heads, head_dim)
vv  = wv(sequence_data).view(batch_size, seq_length, n_heads, head_dim)

xqq = torch.einsum('blhd->bhld',[xqq])
kk  = torch.einsum('blhd->bhld',[kk])
vv  = torch.einsum('blhd->bhld', [vv])

scores_e = torch.einsum("bhid, bhjd -> bhij", [xqq, kk]) / math.sqrt(head_dim)
scores_e = F.softmax(scores_e.float(), dim =-1).type_as(xq)

print(f"Scores shape: {scores_e.shape}")
output_e = torch.einsum("bhlj,bhjd->bhld", scores_e, vv) 
output_e = torch.einsum("bhld->blhd", [output_e]).contiguous().view(batch_size, seq_length, -1)
output == output_e
```

    Scores shape: torch.Size([2, 4, 6, 6])





    tensor([[[True, True, True, True, True, True, True, True, True, True, True,
              True, True, True, True, True, True, True, True, True],
             [True, True, True, True, True, True, True, True, True, True, True,
              True, True, True, True, True, True, True, True, True],
             [True, True, True, True, True, True, True, True, True, True, True,
              True, True, True, True, True, True, True, True, True],
             [True, True, True, True, True, True, True, True, True, True, True,
              True, True, True, True, True, True, True, True, True],
             [True, True, True, True, True, True, True, True, True, True, True,
              True, True, True, True, True, True, True, True, True],
             [True, True, True, True, True, True, True, True, True, True, True,
              True, True, True, True, True, True, True, True, True]],
    
            [[True, True, True, True, True, True, True, True, True, True, True,
              True, True, True, True, True, True, True, True, True],
             [True, True, True, True, True, True, True, True, True, True, True,
              True, True, True, True, True, True, True, True, True],
             [True, True, True, True, True, True, True, True, True, True, True,
              True, True, True, True, True, True, True, True, True],
             [True, True, True, True, True, True, True, True, True, True, True,
              True, True, True, True, True, True, True, True, True],
             [True, True, True, True, True, True, True, True, True, True, True,
              True, True, True, True, True, True, True, True, True],
             [True, True, True, True, True, True, True, True, True, True, True,
              True, True, True, True, True, True, True, True, True]]])



It is easier to see the crux of the operation with the ellipsis notation. We ignore the batch and head dimensions, and see we are summing across the j dimension.


```python
output_ee = torch.einsum("...lj,...jd->...ld", scores_e, vv) 
output_ee = torch.einsum("bhld->blhd", [output_ee]).contiguous().view(batch_size, seq_length, -1)
output == output_ee
```




    tensor([[[True, True, True, True, True, True, True, True, True, True, True,
              True, True, True, True, True, True, True, True, True],
             [True, True, True, True, True, True, True, True, True, True, True,
              True, True, True, True, True, True, True, True, True],
             [True, True, True, True, True, True, True, True, True, True, True,
              True, True, True, True, True, True, True, True, True],
             [True, True, True, True, True, True, True, True, True, True, True,
              True, True, True, True, True, True, True, True, True],
             [True, True, True, True, True, True, True, True, True, True, True,
              True, True, True, True, True, True, True, True, True],
             [True, True, True, True, True, True, True, True, True, True, True,
              True, True, True, True, True, True, True, True, True]],
    
            [[True, True, True, True, True, True, True, True, True, True, True,
              True, True, True, True, True, True, True, True, True],
             [True, True, True, True, True, True, True, True, True, True, True,
              True, True, True, True, True, True, True, True, True],
             [True, True, True, True, True, True, True, True, True, True, True,
              True, True, True, True, True, True, True, True, True],
             [True, True, True, True, True, True, True, True, True, True, True,
              True, True, True, True, True, True, True, True, True],
             [True, True, True, True, True, True, True, True, True, True, True,
              True, True, True, True, True, True, True, True, True],
             [True, True, True, True, True, True, True, True, True, True, True,
              True, True, True, True, True, True, True, True, True]]])



In conclusion, einsum is a nifty tool to implement and visualize almost any tensor operation. An even more versatile tool is einops, that I plan to cover in a subsequent post.
