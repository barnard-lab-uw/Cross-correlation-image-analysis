# Cross-correlation-image-analysis

This package is built around trying to align a series of similar frames to study the changes as a function of time. Our primary use-case is for shearing analysis wherein we are probing changes in the MOKE signal of a CrI3 sample as we apply shear deformations.

We have found that the scipy phase cross correlation/image shift combination do not simply work in a single iteration to get perfect alignment. However, by iterating over several steps it appears to converge. In the end, we generate colormaps comparing frames to a reference frame to codify the changes.

In the included files, as is, you would need to download frames from Morgan's dropbox folder under the subfolder "2023_05_29_push4" and also change the "source_path" accordingly in the ipython notebook.
