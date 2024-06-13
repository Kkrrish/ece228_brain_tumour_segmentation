# ece228_brain_tumour_segmentation

Applying Machine Learning techniques on Medical Imaging tasks has always been a challenging problem. Brain Tumour detection using MRI scans is one such application which has a huge demand and potential upsides from automation, but is still not a solved problem. In this project, we explore 2 major CNN-based approaches to Brain Tumour detection, 1. A pixel-by-pixel Cascading CNN, which trains on focused mini-patches of MRIs, 2. An image-based UNet with Attention, which trains on complete MRI scans and predicts tumour classes. We make some enhancements in the training process and explore the trade-offs of each of these models in order to get the best possible classification results.

References:
[1] Mohammad Havaei, Axel Davy, David Warde-Farley, Antoine Biard, Aaron Courville, Yoshua
Bengio, Chris Pal, Pierre-Marc Jodoin, and Hugo Larochelle. Brain tumor segmentation with
deep neural networks. Medical Image Analysis, 35:18–31, January 2017.
[2] Ozan Oktay, Jo Schlemper, Loic Le Folgoc, Matthew Lee, Mattias Heinrich, Kazunari Misawa,
Kensaku Mori, Steven McDonagh, Nils Y Hammerla, Bernhard Kainz, Ben Glocker, and Daniel
Rueckert. Attention u-net: Learning where to look for the pancreas, 2018.
[3] Marcel Prastawa, Elizabeth Bullitt, Sean Ho, and Guido Gerig. A brain tumor segmentation
framework based on outlier detection. Medical image analysis, 8(3):275–283, 2004.
[4] Olaf Ronneberger, Philipp Fischer, and Thomas Brox. U-net: Convolutional networks for
biomedical image segmentation, 2015.
[5] SMIR. Brats 2013 dataset. https://www.smir.ch/BRATS/Start2013.
[6] GitHub Source code. Achyth Esthuri, Devanshi Panchal, Krish Mehta, Prashil Parekh. https://github.com/Kkrrish/ece228_brain_tumour_segmentation.
[7] Nagesh Subbanna, Doina Precup, and Tal Arbel. Iterative multilevel mrf leveraging context and
voxel information for brain tumour segmentation in mri. In Proceedings of the IEEE conference
on computer vision and pattern recognition, pages 400–405, 2014.
[8] Nagesh K Subbanna, Doina Precup, D Louis Collins, and Tal Arbel. Hierarchical probabilistic
gabor and mrf segmentation of brain tumours in mri volumes. In Medical Image Computing and
Computer-Assisted Intervention–MICCAI 2013: 16th International Conference, Nagoya, Japan,
September 22-26, 2013, Proceedings, Part I 16, pages 751–758. Springer, 2013.
