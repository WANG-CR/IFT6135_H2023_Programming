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
        inputs (`torch.FloatTensor` of shape `(batch_size, sequence_length, 2*hidden_size)`)
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
        score = self.tanh(self.W(inputs) + self.V(hidden_states.transpose(0,1))).sum(dim=-1, keepdim=True) # shape (batchsize, sequencelength, hiddensize) => shape (batchsize, hiddensize, 1)
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
        self.rnn = nn.GRU(input_size=embedding_size, hidden_size=hidden_size,batch_first=True, bidirectional=True)

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
        
        # sum the bidirectional hidden states
        output = output.reshape(output.shape[0], output.shape[1], -1, 2).sum(dim=-1)
        hidden = hidden.sum(dim=0)
        print(f"hidden states shape is {hidden_states.shape}")
        print(f"hidden shape is {hidden.shape}")
        return output, hidden

    def initial_states(self, batch_size, device=None):
        if device is None:
            device = next(self.parameters()).device
        shape = (self.num_layers*2, batch_size, self.hidden_size)
        # The initial state is a constant here, and is not a learnable parameter
        h_0 = torch.zeros(shape, dtype=torch.float, device=device)
        return h_0

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

        self.rnn = nn.GRU(input_size=embedding_size, hidden_size=hidden_size,batch_first=True, bidirectional=False)
        
        self.mlp_attn = Attn(hidden_size, dropout)

    def forward(self, inputs, hidden_states, mask=None):
        """GRU Decoder network with Soft attention

        This is a Unidirectional Gated Recurrent Unit Encoder network
        Parameters
        ----------
        inputs (`torch.LongTensor` of shape `(batch_size, sequence_length, hidden_size)`)
            The input tensor containing the encoded input sequence.

        hidden_states(`torch.FloatTensor` of shape `(num_layers*2, batch_size, hidden_size)`)
            The (initial) hidden state for the bidrectional GRU.

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
        pass
        
        
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
        x, hidden_states = self.encoder(inputs, hidden_states)
        if self.encoder_only:
          x = x[:, 0]
          return x, hidden_states
        x, hidden_states = self.decoder(x, hidden_states, mask)
        x = x[:, 0]
        return x, hidden_states
