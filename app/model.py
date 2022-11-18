import torch
import torch.nn as nn
import numpy as np


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Training on [{}].'.format(device))


class CpuEmbedding(nn.Module):

    def __init__(self, num_embeddings, embed_dim):
        super(CpuEmbedding, self).__init__()

        self.weight = nn.Parameter(torch.zeros((num_embeddings, embed_dim)))
        nn.init.xavier_uniform_(self.weight.data)

    def forward(self, x):
        """
        :param x: shape (batch_size, num_fields)
        :return: shape (batch_size, num_fields, embedding_dim)
        """
        return self.weight[x]


class Embedding:

    def __new__(cls, num_embeddings, embed_dim):
        if torch.cuda.is_available():
            embedding = nn.Embedding(num_embeddings, embed_dim)
            nn.init.xavier_uniform_(embedding.weight.data)
            return embedding
        else:
            return CpuEmbedding(num_embeddings, embed_dim)


class FeaturesEmbedding(nn.Module):

    def __init__(self, field_dims, embed_dim):
        super(FeaturesEmbedding, self).__init__()
        self.embedding = Embedding(sum(field_dims), embed_dim)

        # e.g. field_dims = [2, 3, 4, 5], offsets = [0, 2, 5, 9]
        self.offsets = np.array(
            (0, *np.cumsum(field_dims)[:-1]), dtype=np.long)

    def forward(self, x):
        """
        :param x: shape (batch_size, num_fields)
        :return: shape (batch_size, num_fields, embedding_dim)
        """
        x = x + x.new_tensor(self.offsets)
        return self.embedding(x)


class EmbeddingsInteraction(nn.Module):

    def __init__(self):
        super(EmbeddingsInteraction, self).__init__()

    def forward(self, x):
        """
        :param x: shape (batch_size, num_fields, embedding_dim)
        :return: shape (batch_size, num_fields*(num_fields)//2, embedding_dim)
        """

        num_fields = x.shape[1]
        i1, i2 = [], []
        for i in range(num_fields):
            for j in range(i + 1, num_fields):
                i1.append(i)
                i2.append(j)
        interaction = torch.mul(x[:, i1], x[:, i2])

        return interaction


class AttentionNet(nn.Module):

    def __init__(self, embed_dim=4, t=4):
        super(AttentionNet, self).__init__()

        self.an = nn.Sequential(
            # (batch_size, num_crosses, t), num_crosses = num_fields*(num_fields-1)//2
            nn.Linear(embed_dim, t),
            nn.ReLU(),
            nn.Linear(t, 1, bias=False),  # (batch_size, num_crosses, 1)
            nn.Flatten(),  # (batch_size, num_crosses)
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        return self.an(x)


class AttentionalFactorizationMachine(nn.Module):

    def __init__(self, field_dims, embed_dim=4):
        super(AttentionalFactorizationMachine, self).__init__()

        num_fields = len(field_dims)

        self.w0 = nn.Parameter(torch.zeros((1, )))

        self.embed1 = FeaturesEmbedding(field_dims, 1)
        self.embed2 = FeaturesEmbedding(field_dims, embed_dim)
        self.interact = EmbeddingsInteraction()

        self.attention = AttentionNet(embed_dim)
        self.p = nn.Parameter(torch.zeros(embed_dim, ))
        nn.init.xavier_uniform_(self.p.unsqueeze(0).data)

    def forward(self, x):
        # x size: (batch_size, num_fields)
        # embed(x) size: (batch_size, num_fields, embed_dim)

        embeddings = self.embed2(x)
        interactions = self.interact(embeddings)

        att = self.attention(interactions)
        att_part = interactions.mul(
            att.unsqueeze(-1)).sum(dim=1).mul(self.p).sum(dim=1, keepdim=True)

        output = self.w0 + self.embed1(x).sum(dim=1) + att_part
#         output = torch.sigmoid(output)

        return output
