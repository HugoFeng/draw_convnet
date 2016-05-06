"""
Copyright (c) 2016, Gavin Weiguang Ding
All rights reserved.

Redistribution and use in source and binary forms, with or without
    modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
    list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
    this list of conditions and the following disclaimer in the documentation
    and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors
    may be used to endorse or promote products derived from this software
    without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
    AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
    IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
    ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
    LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
    CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
    SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
    INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
    CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
    ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
    POSSIBILITY OF SUCH DAMAGE.
"""


import os
import numpy as np
import matplotlib.pyplot as plt
plt.rcdefaults()
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.collections import PatchCollection


NumConvMax = 12
NumFcMax = 20
White = 1.
Light = 0.7
Medium = 0.5
Dark = 0.3
Black = 0.


def add_layer(patches, colors, size=(24, 24), num=5,
              top_left=[0, 0],
              loc_diff=[3, -3],
              ):
    # add a rectangle
    top_left = np.array(top_left)
    loc_diff = np.array(loc_diff)
    loc_start = top_left - np.array([0, size[-1]]) # lower left xy of a certain rectangle
    for ind in range(num):
        patches.append(Rectangle(loc_start + ind * loc_diff, size[-2], size[-1]))
        if ind % 2:
            colors.append(Medium)
        else:
            colors.append(Light)


def add_mapping(patches, colors, start_ratio, patch_size, ind_bgn,
                top_left_list, loc_diff_list, num_show_list, size_list):

    start_loc = top_left_list[ind_bgn] \
        + (num_show_list[ind_bgn] - 1) * np.array(loc_diff_list[ind_bgn]) \
        + np.array([start_ratio[-2] * size_list[ind_bgn][-2],
                    -start_ratio[-1] * size_list[ind_bgn][-1]])

    end_loc = top_left_list[ind_bgn + 1] \
        + (num_show_list[ind_bgn + 1] - 1) \
        * np.array(loc_diff_list[ind_bgn + 1]) \
        + np.array([(start_ratio[0] + .5 * patch_size[-2] / size_list[ind_bgn + 1][-2]) *
                    size_list[ind_bgn + 1][-2] - 0.5,
                    -(start_ratio[1] - .5 * patch_size[-1] / size_list[ind_bgn + 1][-1]) *
                    size_list[ind_bgn + 1][-1]])

    patches.append(Rectangle(start_loc, patch_size[-2], patch_size[-1]))
    colors.append(Dark)
    patches.append(Line2D([start_loc[0], end_loc[0]],
                          [start_loc[1], end_loc[1]]))
    colors.append(Black)
    patches.append(Line2D([start_loc[0] + patch_size[-2], end_loc[0]],
                          [start_loc[1], end_loc[1]]))
    colors.append(Black)
    patches.append(Line2D([start_loc[0], end_loc[0]],
                          [start_loc[1] + patch_size[-1], end_loc[1]]))
    colors.append(Black)
    patches.append(Line2D([start_loc[0] + patch_size[-2], end_loc[0]],
                          [start_loc[1] + patch_size[-1], end_loc[1]]))
    colors.append(Black)


def label(xy, text, xy_off=[0, 4]):
    plt.text(xy[0] + xy_off[0], xy[1] + xy_off[1], text,
             family='sans-serif', size=8)


if __name__ == '__main__':

    fc_unit_size = 2
    layer_width_min = 40

    patches = []
    colors = []

    fig, ax = plt.subplots()


    ############################
    # feature maps
    # settings: The 'kernel' or 'map' key may have one or two pairs of (w, h), such as the first 'kernel' (1, 3, 3, 9).
    #           The first pair (1, 3) is the actual network param used for text display,
    #           the second pair (3, 9) is the scaled pixel size only for plotting on large feature maps.
    #           Same thing applies to 'map' size, where size used for plotting may be smaller than the actual size.
    input_setting = {'map': (112, 112, 56, 56), 'map_num': 3}
    network_settings = [{'type': 'Convolution1_v', 'kernel': (1, 3, 3, 9), 'map': (112, 112, 56, 56), 'map_num': 5},
                        {'type': 'Convolution1_h', 'kernel': (3, 1, 9, 3), 'map': (112, 112, 56, 56), 'map_num': 16},
                        {'type': 'Max-pooling', 'kernel': (2, 2, 6, 6), 'map': (56, 56, 40, 40), 'map_num': 16},
                        {'type': 'Convolution2', 'kernel': (1, 3, 3, 9), 'map': (56, 56, 40, 40), 'map_num': 32},
                        {'type': 'Max-pooling', 'kernel': (2, 2, 6, 6), 'map': (28, 28, 32, 32), 'map_num': 32},
                        {'type': 'Convolution3', 'kernel': (3, 3, 9, 9), 'map': (28, 28, 32, 32), 'map_num': 64},
                        {'type': 'Max-pooling', 'kernel': (2, 2, 6, 6), 'map': (14, 14, 20, 20), 'map_num': 64},
                        {'type': 'Convolution4', 'kernel': (3, 3, 5, 5), 'map': (14, 14, 20, 20), 'map_num': 128},
                        {'type': 'Max-pooling', 'kernel': (2, 2), 'map': (7, 7), 'map_num': 128}]
    size_list = [input_setting['map']] + [setting['map'] for setting in network_settings]
    num_list = [input_setting['map_num']] + [setting['map_num'] for setting in network_settings]
    loc_diff_each_map = [3, -3]
    layer_width_padding = 50
    layer_width_list = [max(size[-2:]) + NumConvMax * loc_diff_each_map[0] for size in size_list]

    x_diff_list = [0] + [max(layer_width_min, width + layer_width_padding) for width in layer_width_list]
    text_list = ['Inputs'] + ['Feature\nmaps'] * (len(size_list) - 1)
    loc_diff_list = [loc_diff_each_map] * len(size_list)

    num_show_list = map(min, num_list, [NumConvMax] * len(num_list))
    top_left_list = np.c_[np.cumsum(x_diff_list), np.zeros(len(x_diff_list))]

    for ind in range(len(size_list)):
        add_layer(patches, colors, size=size_list[ind],
                  num=num_show_list[ind],
                  top_left=top_left_list[ind], loc_diff=loc_diff_list[ind])
        label(top_left_list[ind], text_list[ind] + '\n{}@{}x{}'.format(
            num_list[ind], size_list[ind][0], size_list[ind][1]))


    ############################
    # in between layers
    start_ratio_list = [[0.3, 0.4], [0.5, 0.8]] * ((len(network_settings) + 1)//2)
    patch_size_list = [setting['kernel'] for setting in network_settings]
    ind_bgn_list = range(len(patch_size_list))
    text_list = [setting['type'] for setting in network_settings]

    for ind in range(len(patch_size_list)):
        add_mapping(patches, colors, start_ratio_list[ind],
                    patch_size_list[ind], ind,
                    top_left_list, loc_diff_list, num_show_list, size_list)
        label(top_left_list[ind], text_list[ind] + '\n{}x{} kernel'.format(
            patch_size_list[ind][0], patch_size_list[ind][1]), xy_off=[26, -layer_width_list[ind]-25])


    ############################
    # fully connected layers
    network_settings = [{'type': 'Flatten\n',           'visual_size': fc_unit_size, 'num': 256},
                        {'type': 'Dropout\n0.5',        'visual_size': fc_unit_size, 'num': 1024},
                        {'type': 'Fully\nconnected',    'visual_size': fc_unit_size, 'num': 10}]
    size_list = [[setting['visual_size']]*2 for setting in network_settings]
    num_list = [setting['num'] for setting in network_settings]
    layer_width_min = 60
    num_show_list = map(min, num_list, [NumFcMax] * len(num_list))
    x_diff_list = [sum(x_diff_list)] + [layer_width_min] * (len(size_list) -1)
    top_left_list = np.c_[np.cumsum(x_diff_list), np.zeros(len(x_diff_list))]
    loc_diff_list = [[fc_unit_size, -fc_unit_size]] * len(top_left_list)
    text_list = ['Hidden\nunits'] * (len(size_list) - 1) + ['Outputs']

    for ind in range(len(size_list)):
        add_layer(patches, colors, size=size_list[ind], num=num_show_list[ind],
                  top_left=top_left_list[ind], loc_diff=loc_diff_list[ind])
        label(top_left_list[ind], text_list[ind] + '\n{}'.format(
            num_list[ind]))

    text_list = [setting['type'] for setting in network_settings]

    for ind in range(len(size_list)):
        label(top_left_list[ind], text_list[ind], xy_off=[-10, -65])

    ############################
    colors += [0, 1]
    collection = PatchCollection(patches, cmap=plt.cm.gray)
    collection.set_array(np.array(colors))
    ax.add_collection(collection)
    plt.tight_layout()
    plt.axis('equal')
    plt.axis('off')
    # plt.show() # showing the plot will change the layout of the fig as the window changes

    ############################
    # output settings: tweak dpi and fig size for different network depth
    Dpi = 150
    fig.set_size_inches(20, 7.5)

    fig_dir = './'
    fig_ext = '.png'
    fig.savefig(os.path.join(fig_dir, 'convnet_fig' + fig_ext),
                bbox_inches='tight', pad_inches=0, dpi=Dpi)
