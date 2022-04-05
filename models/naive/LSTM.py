import numpy as np
import torch
import torch.nn as nn


class LSTM(nn.Module):
    # An implementation of naive LSTM using Pytorch Linear layers and activations
    # You will need to complete the class forward function.

    def __init__(self, input_size, hidden_size):
        """ Init function for VanillaRNN class
            Args:
                input_size (int): the number of features in the inputs.
                hidden_size (int): the size of the hidden layer
            Returns:
                None
        """
        super(LSTM, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size

        # i_t, the input gate
        self.W_ii = nn.Parameter(torch.Tensor(input_size, hidden_size))
        self.b_ii = nn.Parameter(torch.Tensor(hidden_size))
        self.W_hi = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.b_hi = nn.Parameter(torch.Tensor(hidden_size))

        # f_t, the forget gate
        self.W_if = nn.Parameter(torch.Tensor(input_size, hidden_size))
        self.b_if = nn.Parameter(torch.Tensor(hidden_size))
        self.W_hf = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.b_hf = nn.Parameter(torch.Tensor(hidden_size))

        # g_t, the cell gate
        self.W_ig = nn.Parameter(torch.Tensor(input_size, hidden_size))
        self.b_ig = nn.Parameter(torch.Tensor(hidden_size))
        self.W_hg = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.b_hg = nn.Parameter(torch.Tensor(hidden_size))

        # o_t, the output gate
        self.W_io = nn.Parameter(torch.Tensor(input_size, hidden_size))
        self.b_io = nn.Parameter(torch.Tensor(hidden_size))
        self.W_ho = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.b_ho = nn.Parameter(torch.Tensor(hidden_size))

        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.init_hidden()

    def init_hidden(self):
        for p in self.parameters():
            if p.data.ndimension() >= 2:
                nn.init.xavier_uniform_(p.data)
            else:
                nn.init.zeros_(p.data)

    def forward(self, x: torch.Tensor, init_states=None):
        """Assumes x is of shape (batch, sequence, feature)"""

        ################################################################################
        # TODO:                                                                        #
        #   Implement the forward pass of LSTM. Please refer to the equations in the   #
        #   corresponding section of jupyter notebook. If init_states is None, you     #
        #   should initialize h_t and c_t to be zero vectors.                          #
        ################################################################################
        h_t, c_t = None, None
        
        if(init_states==None):
            h_t = torch.zeros(x.size(0), self.hidden_size)
            c_t = torch.zeros(x.size(0), self.hidden_size)
        else:
            h_t,c_t = init_states
        


        seq = x.size(1)

        for i in range(seq):
            x_t = x[:,i,:]
            # it = σ(Wiixt + bii + Whiht−1 + bhi)
            # ft =σ(Wifxt +bif +Whfht−1 +bhf)
            # gt = tanh(Wigxt + big + Whght−1 + bhg) 
            # ot = σ(Wioxt + bio + Whoht−1 + bho)
            # ct =ft ⊙ct−1 +it ⊙gt
            # ht = ot ⊙ tanh(ct)
            
            # print(gti.shape)

            ft = self.sigmoid(torch.matmul(x_t,self.W_if) + self.b_if + torch.matmul(h_t,self.W_hf) + self.b_hf)
            it = self.sigmoid( torch.matmul(x_t,self.W_ii) + self.b_ii + torch.matmul(h_t,self.W_hi) + self.b_hi)
            gti = (torch.matmul(x_t,self.W_ig) + self.b_ig + torch.matmul(h_t,self.W_hg) + self.b_hg)
            gt = self.tanh( gti)
            ot = self.sigmoid(torch.matmul(x_t,self.W_io) + self.b_io + torch.matmul(h_t,self.W_ho) + self.b_ho)
            c_t = ft * c_t + it*gt
            h_t = ot * self.tanh(c_t)
        ################################################################################
        #                              END OF YOUR CODE                                #
        ################################################################################
        return (h_t, c_t)

