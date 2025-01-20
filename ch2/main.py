import torch

inputs = torch.tensor(
  [[0.43, 0.15, 0.89], # Your     (x^1)
   [0.55, 0.87, 0.66], # journey  (x^2)
   [0.57, 0.85, 0.64], # starts   (x^3)
   [0.22, 0.58, 0.33], # with     (x^4)
   [0.77, 0.25, 0.10], # one      (x^5)
   [0.05, 0.80, 0.55]] # step     (x^6)
)

# query = inputs[1]
# attn_scores_2 = torch.empty(inputs.shape[0])
# print(attn_scores_2.shape)
# for i, x_i in enumerate(inputs):
#     attn_scores_2[i] = torch.dot(x_i, query)
# print(attn_scores_2)

# # Softmax function is a performant way to
# # enxure all attention weights are positive and
# # normalized. This allows us to use the attention
# # weight as a probability (higher weights greater importance)
# attn_weights_2_tmp = torch.softmax(attn_scores_2, dim=0)
# print('Attentionweights: ', attn_weights_2_tmp)
# print('Sum: ', attn_weights_2_tmp.sum())

# context_vec_2 = torch.zeros(query.shape)
# for i,x_i in enumerate(inputs):
#     context_vec_2 += attn_weights_2_tmp[i]*x_i
# print(context_vec_2)

# Get attention scores by using matix multiplication
attn_scores = inputs @ inputs.T

# normalize the attention weights
attn_weights = torch.softmax(attn_scores, dim=-1)

# Calculate the context vectors
all_context_vec = attn_weights @ inputs
print(all_context_vec)
