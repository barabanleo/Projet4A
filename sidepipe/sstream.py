# -*- coding: utf-8 -*-
#
# This file is part of SIDEPIPE.
#
# SIDEPIPE is a python package for speaker verification.
# Home page: http://www-lium.univ-lemans.fr/sidekit/
#
# SIDEPIPE is a python package for acoustic parametrization for speaker verification.
# Home page: http://www-lium.univ-lemans.fr/sidekit/
#
# SIDEPIPE is free software: you can redistribute it and/or modify
# it under the terms of the GNU LLesser General Public License as
# published by the Free Software Foundation, either version 3 of the License,
# or (at your option) any later version.
#
# SIDEPIPE is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with SIDEPIPE.  If not, see <http://www.gnu.org/licenses/>.

"""
Copyright 2016 Anthony Larcher
"""

import os
import numpy as np
import scipy
import copy
import h5py
import sidekit
from sidepipe.utils import coroutine

class SStream(object):
    """ Il faut définir le contenu exact de la config pour qu'il soit bien clair
        si on a plusieurs coroutines du même type dans un même SStream alors
        elles partagent forcément la même config
    """
    valid_sinks = {"file_writer", "dump"}
    valid_sources = {"audio_reader"}
    valid_filters = {"pad_zeros", "framing", "demux", "pre_emphasis", "hamming", "fft", "spectrum", "dct", "tee", "vad", "rasta", "temporal_context", "normalize"}
    valid_non_sink = valid_sources.union(valid_filters)
    valid_elements = valid_sinks.union(valid_sources).union(valid_filters)

    def __init__(self, id, description=()):
        self.buffer_size=None

        self.id = id
        self.description = description
        self.elements = []

        self.output_id = 'default'
        self.padding_context = (0,0)

        self.input_filename=None
        self.input_format = None
        self.output_filename=None
        self.output_dir='./'
        self.output_ext='.h5'
        self.output_format="hdf5"
        self.show=''

        self.prefac = 0.97

        self.sampling_rate = 16000
        self.maxfreq = 8000
        self.lowfreq = 0
        self.widest_lowfreq=0
        self.widest_maxfreq=8000

        self.window_duration = 0.025
        self.window_shift = 0.01
        self._win_size = None
        self._win_shift = None
        self.drop_lowE_frames = False
        self.drop_alpha = 2.
        self.snr = 40

        self.nfft = None
        self.nlogfilt = 40
        self.nceps = 13
        self.log_energy = True
        self.context = (0,0)
        self.delta = 2
        self.delta_mode = 'delta'
        self.delta_window = 3  # window from -3 to +3 : size = 7
        self.delta_filter = [.25, .5, .25, 0, -.25, -.5, -.25]

        self.feat_norm = None
        self.feat_norm_all = False  # if True, normalize use all features to normalize, if False, use only the vad labels

    def set_options(self, stream_options_dic):
        """
        Set the options of the current stream from a dictionary

        :param stream_options_dic: the dictionary of options to set
        """
        if 'buffer_size' in stream_options_dic.keys():
            self.buffer_size = stream_options_dic['buffer_size']

        if 'output_id' in stream_options_dic.keys():
            self.output_id = stream_options_dic['output_id']

        if 'padding_context' in stream_options_dic.keys():
            self.padding_context = stream_options_dic['padding_context']

        if 'input_filename' in stream_options_dic.keys():
            self.input_filename = stream_options_dic['input_filename']
        if 'input_format' in stream_options_dic.keys():
            self.input_format = stream_options_dic['input_format']

        if 'output_dir' in stream_options_dic.keys():
            self.output_dir = stream_options_dic['output_dir']
        if 'output_ext' in stream_options_dic.keys():
            self.output_ext = stream_options_dic['output_ext']
        if 'output_format' in stream_options_dic.keys():
            self.output_format = stream_options_dic['output_format']
        #self.output_filename = self.output_dir + self.show + self.output_ext


        if 'prefac' in stream_options_dic.keys():
            self.prefac = stream_options_dic['prefac']

        if 'sampling_rate' in stream_options_dic.keys():
            self.sampling_rate = stream_options_dic['sampling_rate']
        if 'maxfreq' in stream_options_dic.keys():
            self.maxfreq = stream_options_dic['maxfreq']
        if 'lowfreq' in stream_options_dic.keys():
            self.lowfreq = stream_options_dic['lowfreq']
        if 'widest_lowfreq' in stream_options_dic.keys():
            self.widest_lowfreq = stream_options_dic['widest_lowfreq']
        if 'widest_maxfreq' in stream_options_dic.keys():
            self.widest_maxfreq = stream_options_dic['widest_maxfreq']

        if 'window_duration' in stream_options_dic.keys():
            self.window_duration = stream_options_dic['window_duration']
        if 'window_shift' in stream_options_dic.keys():
            self.window_shift = stream_options_dic['window_shift']
        if '_win_size' in stream_options_dic.keys():
            self._win_size = stream_options_dic['_win_size']
        if '_win_shift' in stream_options_dic.keys():
            self._win_shift = stream_options_dic['_win_shift']
        if 'drop_lowE_frames' in stream_options_dic.keys():
            self.drop_lowE_frames = stream_options_dic['drop_lowE_frames']
        if 'drop_alpha' in stream_options_dic.keys():
            self.drop_alpha = stream_options_dic['drop_alpha']
        if 'snr' in stream_options_dic.keys():
            self.snr = stream_options_dic['snr']

        if 'nfft' in stream_options_dic.keys():
            self.nfft = stream_options_dic['nfft']
        if 'nlogfilt' in stream_options_dic.keys():
            self.nlogfilt = stream_options_dic['nlogfilt']
        if 'nlinfilt' in stream_options_dic.keys():
            self.nlinfilt = stream_options_dic['nlinfilt']
        if 'nceps' in stream_options_dic.keys():
            self.nceps = stream_options_dic['nceps']
        if 'log_energy' in stream_options_dic.keys():
            self.log_energy = stream_options_dic['log_energy']
        if 'context' in stream_options_dic.keys():
            self.context = stream_options_dic['context']
        if 'delta' in stream_options_dic.keys():
            self.delta = stream_options_dic['delta']
        if 'delta_mode' in stream_options_dic.keys():
            self.delta_mode = stream_options_dic['delta_mode']
        if 'delta_window' in stream_options_dic.keys():
            self.delta_window = stream_options_dic['delta_window']
        if 'delta_filter' in stream_options_dic.keys():
            self.delta_filter = stream_options_dic['delta_filter']

        if 'feat_norm' in stream_options_dic.keys():
            self.feat_norm = stream_options_dic['feat_norm']
        if 'feat_norm_all' in stream_options_dic.keys():
            self.feat_norm_all = stream_options_dic['feat_norm_all']

    #def set_sampling_rate(self, fs):
    #    self.sampling_rate()

    def add_element(self, elt_type, followers=None):
        """
        :param elt_type: the type of element to add to the pipe, the element is added at the begining of the pipe
        as it should know its followers
        :param followers: a list of element that have already been creatd and initialized
        :return:
        """
        # on va gérer tous les cas d'elements possibles
        if elt_type == "file_writer":
            self.elements.insert(0, self.file_writer())
        if elt_type == "dump":
            self.elements.insert(0, self.dump())
        elif elt_type == "demux":
            assert len(followers) == 2, "Error: missing followers for demux"
            self.elements.insert(0, self.demux(first_channel=followers[0], second_channel=followers[1]))
        elif elt_type == "pad_zeros":
            self.elements.insert(0, self.pad_zeros(followers))
        elif elt_type == "tee":
            self.elements.insert(0, self.tee(followers))
        elif elt_type == "framing":
            self.elements.insert(0, self.framing(followers))
        elif elt_type == "pre_emphasis":
            self.elements.insert(0, self.pre_emphasis(followers))
        elif elt_type == "hamming":
            self.elements.insert(0, self.hamming(followers))
        elif elt_type == "fft":
            self.elements.insert(0, self.fft(followers))
        elif elt_type == "spectrum":
            self.elements.insert(0, self.spectrum(followers))
        elif elt_type == "dct":
            self.elements.insert(0, self.dct(followers))
        elif elt_type == "vad":
            self.elements.insert(0, self.vad(followers))
        elif elt_type == "rasta":
            self.elements.insert(0, self.rasta(followers))
        elif elt_type == "temporal_context":
            self.elements.insert(0, self.temporal_context(followers))
        elif elt_type == "normalize":
            self.elements.insert(0, self.normalize(followers))
        elif elt_type == "audio_reader":
            self.elements.insert(0, self.audio_reader(followers))

    def reset(self):
        """
        Remove all elements from the stream without modifying the configuration

        """
        del self.elements[:]
        self.buffer_size = None
        self.input_filename = None

    @coroutine
    def audio_reader(self, next_routine):
        if self.input_filename.endswith('.sph') or self.input_filename.endswith('.pcm')\
                or self.input_filename.endswith('.wav') or self.input_filename.endswith('.raw'):
            x, rate = sidekit.frontend.io.read_audio(self.input_filename, self.sampling_rate)

        # add random noise to avoid any issue due to zeros
        np.random.seed(0)
        if x.ndim == 1:
            x += 0.0001 * np.random.randn(x.shape[0])
        elif x.ndim == 2:
            x[:, 0] += 0.0001 * np.random.randn(x.shape[0])
            if x.shape[1] == 2:
                x[:, 1] += 0.0001 * np.random.randn(x.shape[0])

        (yield None)
        idx = 0
        next_routine.send(x)
        next_routine.close()

    @coroutine
    def framing(self, next_routine, pad='zeros'):
        """
        Receive the signal at once (mono channel) and frame it
        :param pad: can be zeros or edge
        """
        try:
            sig = yield
            dsize = sig.dtype.itemsize
            if sig.ndim == 1:
                sig = sig[:, np.newaxis]
            # Manage padding
            context = (self.context,) +  (sig.ndim - 1) * ((0,0),)

            self._win_size = int(self.window_duration * self.sampling_rate)
            self._win_shift = int(self.window_shift * self.sampling_rate)

            win_size = self._win_size + sum(self.context)
            shape = ((sig.shape[0] - self._win_size) / self._win_shift + 1, 1,
                    win_size, sig.shape[1])

            strides = tuple(map(lambda x: x*dsize, [self._win_shift * sig.shape[1], 1, sig.shape[1], 1]))
            if pad == 'zeros':
                sliding_sig = np.lib.stride_tricks.as_strided(np.lib.pad(sig,
                                                                     context,
                                                                     'constant',
                                                                     constant_values=(0,)),
                                                            shape=shape,
                                                            strides=strides).squeeze()
            elif pad == 'edge':
                sliding_sig = np.lib.stride_tricks.as_strided(np.lib.pad(sig,
                                                                     context,
                                                                     'edge'),
                                                            shape=shape,
                                                            strides=strides).squeeze()

            # Apply pre-emphasis filtering if required
            sliding_sig = sliding_sig - np.c_[sliding_sig[..., :1], sliding_sig[..., :-1]] * self.prefac
            # Compute logEnergy and send it as first coefficient
            if self.log_energy:
                energy = np.log((sliding_sig**2).sum(axis=1))
                energy = energy[:,np.newaxis]
            else:
                energy = np.empty((sliding_sig.shape[0],1))

            idx = 0
            # If buffer_size is None, process the entire file at once
            if self.buffer_size is None:
                self.buffer_size = sliding_sig.shape[0]

            # Drop low energy frames if required
            if not self.drop_lowE_frames:
                vad = np.ones(sliding_sig.shape[0], dtype='bool')
            elif self.drop_lowE_frames == 'energy':
                vad = sidekit.frontend.vad.vad_energy(np.log((sliding_sig**2).sum(axis=1)),
                                                      distrib_nb=3,
                                                      nb_train_it=8,
                                                      flooring=0.0001,
                                                      ceiling=1.0,
                                                      alpha=self.drop_alpha)[0]
            elif self.drop_lowE_frames == 'snr':
                vad = sidekit.frontend.vad.vad_snr(sig,
                                                   self.snr,
                                                   fs=self.sampling_rate,
                                                   shift=self.window_shift,
                                                   nwin=self._win_size)
            # Apply smoothing on the labels
            vad = sidekit.frontend.vad.label_fusion(vad)
            while idx < sliding_sig.shape[0]:
                next_routine.send((sliding_sig[idx:idx+self.buffer_size, :].squeeze(), energy, vad))
                idx += self.buffer_size
                yield None

        except GeneratorExit:
            next_routine.close()

    @coroutine
    def demux(self, first_channel, second_channel=None):
        """
        Routine qui sépare les canaux
        """
        try:
            while True:
                sig = yield
                if sig.squeeze().ndim == 1:
                    assert second_channel is None, "Error: too many followers in {}".format(self.input_filename)
                    first_channel.send(sig.squeeze())
                elif sig.squeeze().shape[1] == 2:
                    assert second_channel is not None, "Error: missing second follower"
                    first_channel.send(sig[:,0].squeeze())
                    second_channel.send(sig[:,1].squeeze())
        except GeneratorExit:
            first_channel.close()
            if second_channel is not None:
                second_channel.close()

    @coroutine
    def pad_zeros(self, next_routine):
        try:
            while True:
                sig = yield
                context = (sig.ndim - 1) * ((0,0),) +  (self.padding_context,)
                next_routine.send(np.lib.pad(sig, context, 'constant', constant_values=(0,)))
        except GeneratorExit:
            next_routine.close()

    @coroutine
    def file_writer(self):
        collect_sig = []
        collect_energy = []
        collect_vad = []
        try:
            while True:
                x = yield
                if isinstance(x, tuple):
                    (x, energy, vad) = x
                collect_sig.append(x)
                collect_energy.append(energy)
                collect_vad.append(vad)
        except GeneratorExit:
            dir_name = os.path.dirname(self.output_filename)
            if not os.path.exists(dir_name) and (dir_name is not ''):
                os.makedirs(dir_name)
            with h5py.File(self.output_filename, 'a') as fh:
                sidekit.frontend.io.write_hdf5(show = self.show,
                                               fh = fh,
                                               cep = np.vstack(collect_sig),
                                               cep_mean = None,
                                               cep_std = None,
                                               energy = np.vstack(collect_energy),
                                               energy_mean = None,
                                               energy_std = None,
                                               fb = None,
                                               fb_mean = None,
                                               fb_std = None,
                                               bnf = None,
                                               bnf_mean = None,
                                               bnf_std = None,
                                               label = np.vstack(collect_vad))

    @coroutine
    def dump(self):
        try:
            while True:
                _ = yield
        except GeneratorExit:
            pass

    @coroutine
    def hamming(self, next_routine):
        try:
            while True:
                (sig, energy, vad) = yield
                #next_routine.send((sig * np.hamming(self._win_size)[np.newaxis, :], vad))
                next_routine.send((sig * np.hamming(self._win_size), energy, vad))
        except GeneratorExit:
            next_routine.close()

    @coroutine
    def fft(self, next_routine):
        """
        Send the FFT of the signal ans the energy as first row of the output
	"""
        # Set FFT parameters (number of points)
        self._win_size = int(self.window_duration * self.sampling_rate)
        self.nfft = 2 ** int(np.ceil(np.log2(self._win_size)))
        try:
            while True:
                (sig, energy, vad) = yield
                fft = np.fft.rfft(sig, self.nfft, axis=-1)
                next_routine.send((fft.real**2 + fft.imag**2 , energy, vad))
        except GeneratorExit:
            next_routine.close()

    @coroutine
    def spectrum(self, next_routine):

        """
        If True, the log_energy is added as first component of the output

        :param next_routine:
        :return: the spectrum and a VAD label
        """

        self._win_size = int(self.window_duration * self.sampling_rate)
        self.nfft = 2**int(np.ceil(np.log2(self._win_size)))
        fbank = sidekit.frontend.trfbank(self.sampling_rate,
                                         self.nfft,
                                         lowfreq=self.lowfreq,
                                         maxfreq=self.maxfreq,
                                         nlinfilt=self.nlinfilt,
                                         nlogfilt=self.nlogfilt)[0]

        #if self.widest_lowfreq is None:
        #    self.widest_lowfreq = self.lowfreq
        #if self.widest_maxfreq is None:
        #    self.widest_maxfreq = self.maxfreq
        #fbank = sidekit.frontend.mel_filter_bank(self.sampling_rate, self.nfft,
        #                                         lowfreq=self.lowfreq, maxfreq=self.maxfreq,
        #                                         widest_nlogfilt=self.nlogfilt,
        #                                         widest_lowfreq=self.widest_lowfreq, widest_maxfreq=self.widest_maxfreq)[0]
        try:
            while True:
                (sig, energy, vad) = yield
                next_routine.send((np.log(np.dot(sig, fbank.T)), energy, vad))
        except GeneratorExit:
            next_routine.close()

    @coroutine
    def dct(self, next_routine):
        """
        WARNING:  if the log_energy is set to True, then the input content the log_energy
         Don't keep CO
         Just pass the VAD
        :param next_routine:
        :return:
        """
        try:
            while True:
                (fb, energy, vad) = yield
                next_routine.send((scipy.fftpack.realtransforms.dct(fb, type=2, norm='ortho', axis=-1)[:, 1:self.nceps + 1], energy, vad))
        except GeneratorExit:
            next_routine.close()


    @coroutine
    def tee(self, next_routine_list):
        """
        Needs to deepcopy each element otherwise shared by the branches
        :param next_routine_list: a list of output coroutines to feed
        """
        try:
            while True:
                input = yield
                for ncr in next_routine_list:
                    ncr.send(copy.deepcopy(input))
        except GeneratorExit:
            for ncr in next_routine_list:
                ncr.close()

    def rasta():
        pass


    # La version ci-dessous intègre un buffer qui permet de calculer les dérivés en flux
    # Cette version n'est pas complète car elle n'intègre pas le calcul des double dérivées
    #@coroutine
    #def temporal_context(self, next_routine):
    #    """ could be delta or sdc
    #    """
    #    try:
    #        if self.delta_mode == 'delta':
    #
    #            # Create the buffer
    #            window_length = 2 * self.delta_window + 1
    #            idx = self.delta_window
    #            frame_buffer = np.zeros((window_length, self.nceps))
    #            vad_buffer = np.zeros(window_length, dtype='bool')
    #            delta = np.zeros(self.nceps)
    #            while True:
    #                (frame_buffer[idx], vad_buffer[idx]) = yield
    #                delta += (frame_buffer[(idx + 2) % window_length]
    #                        - frame_buffer[(idx - 2) % window_length])
    #                next_routine.send((np.hstack((frame_buffer[idx], delta)), vad_buffer[idx]))
    #                idx = (idx + 1) % window_length  # a verifier
    #
    #    except GeneratorExit:
    #        #for ii in range(self.delta_window):
    #        #    delta -= frame_buffer[(idx - 2) % window_length]
    #        #    next_routine.send((np.hstack((frame_buffer[idx], delta)), vad_buffer[idx]))
    #        #    idx = (idx + 1) % window_length   # a verifier
    #        next_routine.close()

    # La version ci-dessous de temporal_context, ne gère absolument pas le flux mais
    # calcule les dérivés premières et secondes sur l'ensemble du fichier en une seule fois
    @coroutine
    def temporal_context(self, next_routine):

        try:
            while True:
                (cep, energy, vad) = yield
                if self.delta in [1, 2]:
                    delta = sidekit.frontend.features.compute_delta(cep,
                                                                    win= self.delta_window,
                                                                    filt=self.delta_filter)
                    delta_E = sidekit.frontend.features.compute_delta(energy,
                                                                    win= self.delta_window,
                                                                    filt=self.delta_filter)
                if self.delta == 2:
                    double_delta = sidekit.frontend.features.compute_delta(delta,
                                                                           win= self.delta_window,
                                                                           filt=self.delta_filter)
                    double_delta_E = sidekit.frontend.features.compute_delta(delta_E,
                                                                           win= self.delta_window,
                                                                           filt=self.delta_filter)
                else:
                    double_delta = np.empty((delta.shape[0], 0))
                    double_delta_E = np.empty((delta_E.shape[0], 0))

                cep = np.column_stack((cep, delta, double_delta))
                energy = np.column_stack((energy, delta_E, double_delta_E))
                next_routine.send((cep, energy, vad))
        except GeneratorExit:
            next_routine.close()


    # La version ci-dessous est pensée pour fonctionner en flux et gère un buffer interne
    # Elle n'est pas complète
    #@coroutine
    #def normalize(self, next_routine):
    #    """ could be sdc, cms or cmvn
    #    We wait 3 seconds before starting the normalization
    #    """
    #    try:
    #        # create the buffer for a window of 3 seconds
    #        frame_buffer =  np.empty((int(3 / self.window_shift) , self.nceps * (self.delta + 1)))
    #        vad_buffer = np.zeros((int(3 / self.window_shift)), dtype='bool')
    #        received_frame = 0
    #
    #        while received_frame < 300:
    #            frame_buffer[received_frame], vad_buffer[received_frame] = yield
    #            received_frame += 1
    #
    #        # Compute normalization parameters for CMVN
    #        """ Normalize only with speech frames
    #        """
    #        mu = frame_buffer[vad_buffer, :].mean(axis=0)
    #        stdev = frame_buffer[vad_buffer, :].std(axis=0)
    #
    #        # Apply normalization and send all 300 frames before processing new ones
    #        frame_buffer -= mu
    #        frame_buffer /= stdev
    #
    #        for ii in range(min(300, received_frame)):
    #            next_routine.send((frame_buffer[ii], vad_buffer[ii]))
    #
    #        while True:
    #            (frame, vad) = yield
    #            next_routine.send(((frame - mu)/stdev, vad))
    #
    #    except GeneratorExit:
    #        next_routine.close()

    # La version ci-dessous de normalize, ne gère absolument pas le flux mais
    # normalise les trames sur l'ensemble du fichier en une seule fois
    @coroutine
    def normalize(self, next_routine):
        try:
            while True:
                (cep, energy, vad) = yield
                if self.feat_norm == 'cms':
                        sidekit.frontend.normfeat.cms(cep, vad + self.feat_norm_all)
                        sidekit.frontend.normfeat.cms(energy, vad + self.feat_norm_all)
                elif self.feat_norm == 'cmvn':
                    sidekit.frontend.normfeat.cmvn(cep, vad + self.feat_norm_all)
                    sidekit.frontend.normfeat.cmvn(energy, vad + self.feat_norm_all)
                elif self.feat_norm == 'stg':
                    sidekit.frontend.normfeat.stg(cep, vad + self.feat_norm_all)
                    sidekit.frontend.normfeat.stg(energy, vad + self.feat_norm_all)
                elif self.feat_norm == 'cmvn_sliding':
                    sidekit.frontend.normfeat.cep_sliding_norm(cep, win=301, center=True, reduce=True)
                    sidekit.frontend.normfeat.cep_sliding_norm(energy, win=301, center=True, reduce=True)
                elif self.feat_norm == 'cms_sliding':
                    sidekit.frontend.normfeat.cep_sliding_norm(cep, win=301, center=True, reduce=False)
                    sidekit.frontend.normfeat.cep_sliding_norm(energy, win=301, center=True, reduce=False)
            next_routine.send((cep, energy, vad))

        except GeneratorExit:
            next_routine.send((cep, energy, vad))
            next_routine.close()





# POUR l'affichage des graphes, regarder le module graphviz

