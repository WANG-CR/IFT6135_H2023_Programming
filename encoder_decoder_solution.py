import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class GRU(nn.Module):
    def __init__(
            self,
            input_size, 
            hidden_size
            ):
        super(GRU, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.w_ir = nn.Parameter(torch.empty(hidden_size, input_size))
        self.w_iz = nn.Parameter(torch.empty(hidden_size, input_size))
        self.w_in = nn.Parameter(torch.empty(hidden_size, input_size))

        self.b_ir = nn.Parameter(torch.empty(hidden_size))
        self.b_iz = nn.Parameter(torch.empty(hidden_size))
        self.b_in = nn.Parameter(torch.empty(hidden_size))

        self.w_hr = nn.Parameter(torch.empty(hidden_size, hidden_size))
        self.w_hz = nn.Parameter(torch.empty(hidden_size, hidden_size))
        self.w_hn = nn.Parameter(torch.empty(hidden_size, hidden_size))

        self.b_hr = nn.Parameter(torch.empty(hidden_size))
        self.b_hz = nn.Parameter(torch.empty(hidden_size))
        self.b_hn = nn.Parameter(torch.empty(hidden_size))
        for param in self.parameters():
            nn.init.uniform_(param, a=-(1/hidden_size)**0.5, b=(1/hidden_size)**0.5)

    def forward(self, inputs, hidden_states):
        """GRU.
        
        This is a Gated Recurrent Unit
        Parameters
        ----------
        inputs (`torch.FloatTensor` of shape `(batch_size, sequence_length, input_size)`)
          The input tensor containing the embedded sequences. input_size corresponds to embedding size.
          
        hidden_states (`torch.FloatTensor` of shape `(1, batch_size, hidden_size)`)
          The (initial) hidden state.
          
        Returns
        -------
        outputs (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`)
          A feature tensor encoding the input sentence. 
          
        hidden_states (`torch.FloatTensor` of shape `(1, batch_size, hidden_size)`)
          The final hidden state. 
        """
        # ==========================
        # TODO: Write your code here
        # ==========================
        x_ir = inputs@self.w_ir.transpose(0,1)
        x_iz = inputs@self.w_iz.transpose(0,1)
        x_in = inputs@self.w_in.transpose(0,1)
        # shape (batch_size, sequence_length, hidden_size)
        
        outputs = []
        h_t = hidden_states[0]
        for t in range(inputs.shape[1]):
            r_t = F.sigmoid(x_ir[:, t, :] + self.b_ir + h_t@self.w_hr.transpose(0,1) + self.b_hr)
            z_t = F.sigmoid(x_iz[:, t, :] + self.b_iz + h_t@self.w_hz.transpose(0,1) + self.b_hz)
            n_t = F.tanh(x_in[:, t, :] + self.b_in + r_t * (h_t@self.w_hn.transpose(0,1) + self.b_hn))
            h_t = (1 - z_t) * n_t + z_t * h_t
            outputs.append(h_t)
    
        return torch.stack(outputs).transpose(0,1), h_t.unsqueeze(0) 
        



class Attn(nn.Module):
    def __init__(
        self,
        hidden_size=256,
        dropout=0.0 # note, this is an extrenous argument
        ):
        super(Attn, self).__init__()
        self.hidden_size = hidden_size

        self.W = nn.Linear(hidden_size*2, hidden_size)
        self.V = nn.Linear(hidden_size, hidden_size) # in the forwards, after multiplying
                                                     # do a torch.sum(..., keepdim=True), its a linear operation

        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=1)


    def forward(self, inputs, hidden_states, mask = None):
        """Soft Attention mechanism.

        This is a one layer MLP network that implements Soft (i.e. Bahdanau) Attention with masking
        Parameters
        ----------
        inputs (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`)
            The input tensor containing the embedded sequences.

        hidden_states (`torch.FloatTensor` of shape `(num_layers, batch_size, hidden_size)`)
            The (initial) hidden state.

        mask ( optional `torch.LongTensor` of shape `(batch_size, sequence_length)`)
            The masked tensor containing the location of padding in the sequences.

        Returns
        -------
        outputs (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`)
            A feature tensor encoding the input sentence with attention applied.

        x_attn (`torch.FloatTensor` of shape `(batch_size, sequence_length, 1)`)
            The attention vector.
        """
        # ==========================
        # TODO: Write your code here
        # ==========================
        # tanh(self.W (inputs),self.V (hidden_states))
        # score = self.tanh(self.W(inputs) + self.V(hidden_states.transpose(0,1))).sum(dim=-1, keepdim=True) # shape (batchsize, sequencelength, hiddensize) => shape (batchsize, sequencelength, 1)
        # alpha = self.softmax(score)

        # query = torch.cat((hidden_states.sum(dim=0).unsqueeze(1).repeat(1, sequence_length, 1), inputs), dim=-1)
        print(f"input shape is {inputs.shape}")
        print(f"hidden states shape is {hidden_states.shape}")

        hidden_states = hidden_states.reshape(-1, 1, hidden_states.shape[-1]).repeat(1, inputs.shape[1], 1)
        inputs = inputs.repeat(hidden_states.shape[0], 1, 1)
        score = self.V(self.tanh(self.W(torch.cat([inputs,hidden_states], dim=-1)))).sum(dim=-1, keepdim=True) #(batchsize, sequencelength, 1)
        # alpha = self.softmax(score)
        if mask:
            # score = score.masked_fill(mask.unsqueeze(-1), -float('inf'))
            score = score.masked_fill(mask.unsqueeze(-1)==0, -float('inf'))
        alpha = self.softmax(score)
        outputs = inputs * alpha
        return outputs, alpha


class Encoder(nn.Module):
    def __init__(
        self,
        vocabulary_size=30522,
        embedding_size=256,
        hidden_size=256,
        num_layers=1,
        dropout=0.0
        ):
        super(Encoder, self).__init__()
        self.vocabulary_size = vocabulary_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.embedding = nn.Embedding(
            vocabulary_size, embedding_size, padding_idx=0,
        )

        self.dropout = nn.Dropout(p=dropout)
        self.rnn = nn.GRU(input_size=embedding_size, hidden_size=hidden_size,batch_first=True, bidirectional=True, dropout=dropout)

    def forward(self, inputs, hidden_states):
        """GRU Encoder.

        This is a Bidirectional Gated Recurrent Unit Encoder network
        Parameters
        ----------
        inputs (`torch.FloatTensor` of shape `(batch_size, sequence_length)`)
            The input tensor containing the token sequences.

        hidden_states(`torch.FloatTensor` of shape `(num_layers*2, batch_size, hidden_size)`)
            The (initial) hidden state for the bidrectional GRU.
            
        Returns
        -------
        outputs (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`)
            A feature tensor encoding the input sentence. 

        hidden_states (`torch.FloatTensor` of shape `(num_layers, batch_size, hidden_size)`)
            The final hidden state. 
        """
        # ==========================
        # TODO: Write your code here
        # ==========================
        embedding = self.dropout(self.embedding(inputs))
        
        # bidirectional GRU
        output, hidden = self.rnn(embedding, hidden_states)
        # output shape is [B, L, 2*H]
        # hidden shape is [2*1, B, H]

        B = output.shape[0]
        L = output.shape[1]
        # sum the bidirectional hidden states
        output = output.reshape(B, L, 2, -1).sum(dim=-2)
        hidden = hidden.reshape(-1, 2, B, self.hidden_size).sum(dim=1)
        print(f"output shape is {output.shape}, line188")
        print(f"hidden is {hidden.shape}, line189")
        return output, hidden

    def initial_states(self, batch_size, device=None):
        if device is None:
            device = next(self.parameters()).device
        shape = (self.num_layers*2, batch_size, self.hidden_size)
        # The initial state is a constant here, and is not a learnable parameter
        h_0 = torch.zeros(shape, dtype=torch.float, device=device)
        return h_0

# class Encoder(nn.Module):
#     def __init__(
#             self,
#             vocabulary_size=30522,
#             embedding_size=256,
#             hidden_size=256,
#             num_layers=1,
#             dropout=0.0
#     ):
#         super(Encoder, self).__init__()
#         self.vocabulary_size = vocabulary_size
#         self.embedding_size = embedding_size
#         self.hidden_size = hidden_size
#         self.num_layers = num_layers

#         self.embedding = nn.Embedding(
#             vocabulary_size, embedding_size, padding_idx=0,
#         )

#         self.dropout = nn.Dropout(p=dropout)
#         self.rnn = nn.GRU(
#             input_size=embedding_size,
#             hidden_size=hidden_size,
#             num_layers=num_layers,
#             batch_first=True,
#             bidirectional=True,
#             dropout=dropout
#         )

#     def forward(self, inputs, hidden_states):
#         """GRU Encoder.

#         This is a Bidirectional Gated Recurrent Unit Encoder network
#         Parameters
#         ----------
#         inputs (`torch.FloatTensor` of shape `(batch_size, sequence_length)`)
#             The input tensor containing the token sequences.

#         hidden_states(`torch.FloatTensor` of shape `(num_layers*2, batch_size, hidden_size)`)
#             The (initial) hidden state for the bidrectional GRU.

#         Returns
#         -------
#         outputs (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`)
#             A feature tensor encoding the input sentence.

#         hidden_states (`torch.FloatTensor` of shape `(num_layers, batch_size, hidden_size)`)
#             The final hidden state.
#         """
#         inputs = self.embedding(inputs)
#         inputs = self.dropout(inputs)
#         outputs, hidden_states = self.rnn(inputs, hidden_states)
#         outputs = outputs.reshape(outputs.shape[0], outputs.shape[1], 2, -1).sum(2)
#         hidden_states = hidden_states.reshape(-1, 2, hidden_states.shape[1], hidden_states.shape[2]).sum(1)
#         return outputs, hidden_states

#     def initial_states(self, batch_size, device=None):
#         if device is None:
#             device = next(self.parameters()).device
#         shape = (self.num_layers * 2, batch_size, self.hidden_size)
#         # The initial state is a constant here, and is not a learnable parameter
#         h_0 = torch.zeros(shape, dtype=torch.float, device=device)
#         return h_0

class DecoderAttn(nn.Module):
    def __init__(
        self,
        vocabulary_size=30522,
        embedding_size=256,
        hidden_size=256,
        num_layers=1,
        dropout=0.0, 
        ):

        super(DecoderAttn, self).__init__()
        self.vocabulary_size = vocabulary_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = nn.Dropout(p=dropout)

        self.rnn = nn.GRU(input_size=embedding_size, hidden_size=hidden_size,batch_first=True, bidirectional=False, dropout=dropout)
        
        self.mlp_attn = Attn(hidden_size, dropout)

    def forward(self, inputs, hidden_states, mask=None):
        """GRU Decoder network with Soft attention

        This is a Unidirectional Gated Recurrent Unit Decoder network
        Parameters
        ----------
        inputs (`torch.LongTensor` of shape `(batch_size, sequence_length, hidden_size)`)
            The input tensor containing the encoded input sequence.

        hidden_states(`torch.FloatTensor` of shape `(num_layers, batch_size, hidden_size)`)
            The (initial) hidden state for the unidirectional GRU.

        mask ( optional `torch.LongTensor` of shape `(batch_size, sequence_length)`)
            The masked tensor containing the location of padding in the sequences.

        Returns
        -------
        outputs (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`)
            A feature tensor encoding the input sentence. 

        hidden_states (`torch.FloatTensor` of shape `(num_layers, batch_size, hidden_size)`)
            The final hidden state. 
        """
        # ==========================
        # TODO: Write your code here
        # ==========================
        # hidden_states = self.dropout(hidden_states)
        inputs = self.dropout(inputs)
        attended_inputs, alpha = self.mlp_attn(inputs, hidden_states, mask)
        outputs, hidden_states = self.rnn(attended_inputs, hidden_states)
        print(f"outputs shape is {outputs.shape}")
        print(f"hidden_states shape is {hidden_states.shape}")
        return outputs, hidden_states
        
        
class EncoderDecoder(nn.Module):
    def __init__(
        self,
        vocabulary_size=30522,
        embedding_size=256,
        hidden_size=256,
        num_layers=1,
        dropout = 0.0,
        encoder_only=False
        ):
        super(EncoderDecoder, self).__init__()
        self.encoder_only = encoder_only
        self.encoder = Encoder(vocabulary_size, embedding_size, hidden_size,
                num_layers, dropout=dropout)
        if not encoder_only:
            self.decoder = DecoderAttn(vocabulary_size, embedding_size, hidden_size, num_layers, dropout=dropout)
        
    def forward(self, inputs, mask=None):
        """GRU Encoder-Decoder network with Soft attention.

        This is a Gated Recurrent Unit network for Sentiment Analysis. This
        module returns a decoded feature for classification. 
        Parameters
        ----------
        inputs (`torch.LongTensor` of shape `(batch_size, sequence_length)`)
            The input tensor containing the token sequences.

        mask (`torch.LongTensor` of shape `(batch_size, sequence_length)`)
            The masked tensor containing the location of padding in the sequences.

        Returns
        -------
        x (`torch.FloatTensor` of shape `(batch_size, hidden_size)`)
            A feature tensor encoding the input sentence. 

        hidden_states (`torch.FloatTensor` of shape `(num_layers, batch_size, hidden_size)`)
            The final hidden state. 
        """
        hidden_states = self.encoder.initial_states(inputs.shape[0])
        x, hidden_states = self.encoder(inputs, hidden_states)  # hidden_size shape is [num_layers, batch_size, hidden_size]
        if self.encoder_only:
          x = x[:, 0]
          return x, hidden_states
        x, hidden_states = self.decoder(x, hidden_states, mask) # input hidden_size shape should be [num_layers*2, batch_size, hidden_size]?
        x = x[:, 0]
        return x, hidden_states
