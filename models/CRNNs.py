import pdb

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.model_utilities import ConvBlock, init_gru, init_layer, interpolate


class CRNN3(nn.Module):
    def __init__(self, class_num, pool_type='avg', pool_size=(1,4), pretrained_path=None):
        
        super().__init__()

        self.class_num = class_num
        self.pool_type = pool_type
        self.pool_size = pool_size
        
        self.conv_block1 = ConvBlock(in_channels=1, out_channels=64)
        self.conv_block2 = ConvBlock(in_channels=64, out_channels=128)
        self.conv_block3 = ConvBlock(in_channels=128, out_channels=256)
        self.conv_block4 = ConvBlock(in_channels=256, out_channels=512)

        self.gru = nn.GRU(input_size=512, hidden_size=256, 
            num_layers=1, batch_first=True, bidirectional=True)

        self.event_fc = nn.Linear(512, class_num, bias=True)

        # self.azimuth_fc = nn.Linear(512, class_num, bias=True)
        # self.elevation_fc = nn.Linear(512, class_num, bias=True)

        self.init_weights()

    def init_weights(self):

        init_gru(self.gru)
        init_layer(self.event_fc)
        # init_layer(self.azimuth_fc)
        # init_layer(self.elevation_fc)

    def forward(self, x):
        '''input: (batch_size, mic_channels, time_steps, mel_bins)'''

        x = self.conv_block1(x, self.pool_type, pool_size=self.pool_size)
        x = self.conv_block2(x, self.pool_type, pool_size=self.pool_size)
        x = self.conv_block3(x, self.pool_type, pool_size=self.pool_size)
        x = self.conv_block4(x, self.pool_type, pool_size=self.pool_size)
        '''(batch_size, feature_maps, time_steps, mel_bins)'''

        if self.pool_type == 'avg':
            x = torch.mean(x, dim=3)
        elif self.pool_type == 'max':
            (x, _) = torch.max(x, dim=3)
        '''(batch_size, feature_maps, time_steps)'''

        x = x.transpose(1,2)
        ''' (batch_size, time_steps, feature_maps):'''

        (x, _) = self.gru(x)

        x = self.event_fc(x)
        sigmoid_output = torch.sigmoid(x)
        softmax_output = torch.softmax(x)
        loss_output = torch.sum(sigmoid_output * softmax_output, dim=1) / torch.sum(softmax_output, dim=1)

        # azimuth_output = self.azimuth_fc(x)
        # elevation_output = self.elevation_fc(x)
        #loss_output
        '''(batch_size, class_num)'''
        #sigmoid_output
        '''(batch_size, time_steps, class_num)'''

        output = {
            'loss_output': loss_output,
            'inference_output': sigmoid_output
        }

        return output


class pretrained_CRNN10(CRNN10):

    def __init__(self, class_num, pool_type='avg', pool_size=(2,2), interp_ratio=16, pretrained_path=None):

        super().__init__(class_num, pool_type, pool_size, interp_ratio, pretrained_path)
        
        self.load_weights(pretrained_path)

    def load_weights(self, pretrained_path):

        model = CRNN10(self.class_num, self.pool_type, self.pool_size, self.interp_ratio)
        checkpoint = torch.load(pretrained_path, map_location=lambda storage, loc: storage)
        model.load_state_dict(checkpoint['model_state_dict'])

        self.conv_block1 = model.conv_block1
        self.conv_block2 = model.conv_block2
        self.conv_block3 = model.conv_block3
        self.conv_block4 = model.conv_block4

        init_gru(self.gru)
        init_layer(self.event_fc)
        init_layer(self.azimuth_fc)
        init_layer(self.elevation_fc)
