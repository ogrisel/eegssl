# eegssl

Experiments on Self-Supervised Learning on EEG data

This is a partial and independent replication of the "relative 
positioning" (RP) pretext task of:

- *Self-supervised representation learning from electroencephalography signals*
  Hubert Banville, Isabela Albuquerque, Aapo Hyvärinen, Graeme Moffat,
  Denis-Alexander Engemann, Alexandre Gramfort

  https://arxiv.org/abs/1911.05419

This experiment uses the dataset from:

- *Analysis of a sleep-dependent neuronal feedback loop: the slow-wave
  microcontinuity of the EEG.* B Kemp, AH Zwinderman, B Tuk, HAC Kamphuisen,
  JJL Oberyé.  IEEE-BME 47(9):1185-1194 (2000).

  https://physionet.org/content/sleep-edfx/1.0.0/

Some code (data loading in particular) is taken from the following mne-python
tutorial:

-  https://mne.tools/dev/auto_tutorials/sample-datasets/plot_sleep.html
