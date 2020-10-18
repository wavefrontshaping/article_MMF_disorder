# Code sample and supporting data for the paper
# [*Learning and avoiding disorder in multimode fibers*](https://arxiv.org)
# **M. W. Matthès, Y. Bromberg, J. de Rosny and S. M. Popoff**

## /Data
- `param.json`: json file containing the parameters of the experiment
- `TM_modes_X.npz`: transmission matrix in the mode basis after correction for the deformation <img src="https://render.githubusercontent.com/render/math?math=\Delta x = X \mu m">
- `TM5_0.npy` and `TM5_1.npy`: full transmission matrix in the pixel basis for the reference stat of the system, i.e. no deformation (<img src="https://render.githubusercontent.com/render/math?math=\Delta x = 0 \mu m">).
Because of the 100 Mo restriction of Github, the file is split into two, can be recombined with:
```python
part1 = np.load('TM5_0.npy')
part2 = np.load('TM5_1.npy')
TM_ref_pix = np.concatenate([part1, part2], axis = 0)
```
- `TM52_0.npy` and `TM52_1.npy`: full transmission matrix in the pixel basis for the maximum deformation (<img src="https://render.githubusercontent.com/render/math?math=\Delta x = 70 \mu m">).
- `conversion_matrices.npz`: contains the matrices `modes_in` and `modes_out` mode matrices computed for a system without aberration. They are generated using [Generate_theoretical_modes/Generate_modes.ipynb](./Generate_theoretical_modes/Generate_modes.ipynb).
- `modes_in_after_correction.npy`: Change of basis matrix between the input mode basis and the input pixel basis after aberration correction optimization.
- `modes_out_after_correction.npy`: Change of basis matrix between the output mode basis and the output pixel basis after aberration correction optimization.
- `mask_near_degenerate.npy`: a mask of the same size as the mode basis TM that represents the blocks of quasi-degenerate modes. 