# Code examples and supporting data for the paper
# [*Learning and avoiding disorder in multimode fibers*](https://arxiv.org/abs/2010.14813)
## **M. W. Matthès, Y. Bromberg, J. de Rosny and S. M. Popoff**


```
@misc{2010.14813,
author = {Maxime W. Matthès and Yaron Bromberg and Julien de Rosny and Sébastien M. Popoff},
yitle = {Learning and avoiding disorder in multimode fibers},
year = {2020},
archivePrefix = "arXiv",
eprint = {arXiv:2010.14813},
url = {https://arxiv.org/abs/2010.14813},
}
```

**Global requirements:**
- Numpy
- Matplotlib

## /Data
Contain the raw and processed data required to generate the figures and to run the demo codes. 
- `param.json`: json file containing the parameters of the experiment
- `TM_modes_X.npz`: transmission matrix in the mode basis after correction for the deformation <img src="https://render.githubusercontent.com/render/math?math=\Delta x = X \mu m">
- `TM5_0.npy` and `TM5_1.npy`: full transmission matrix in the pixel basis for no deformation (<img src="https://render.githubusercontent.com/render/math?math=\Delta x = -40 \mu m">).
Because of the 100 Mo restriction of Github, the file is split into two, can be recombined with:
```python
import numpy as np
part1 = np.load('TM5_0.npy')
part2 = np.load('TM5_1.npy')
TM_ref_pix = np.concatenate([part1, part2], axis = 0)
```
- `TM17_0.npy` and `TM17_1.npy`: full transmission matrix in the pixel basis for no deformation (<img src="https://render.githubusercontent.com/render/math?math=\Delta x = 0 \mu m">)
- `TM25_0.npy` and `TM25_1.npy`: full transmission matrix in the pixel basis for <img src="https://render.githubusercontent.com/render/math?math=\Delta x = 16 \mu m">
- `TM35_0.npy` and `TM35_1.npy`: full transmission matrix in the pixel basis for <img src="https://render.githubusercontent.com/render/math?math=\Delta x = 36 \mu m">
- `TM50_0.npy` and `TM50_1.npy`: full transmission matrix in the pixel basis for <img src="https://render.githubusercontent.com/render/math?math=\Delta x = 66 \mu m">
- `TM52_0.npy` and `TM52_1.npy`: full transmission matrix in the pixel basis for the maximum deformation (<img src="https://render.githubusercontent.com/render/math?math=\Delta x = 70 \mu m">).
- `conversion_matrices.npz`: contains the matrices `modes_in` and `modes_out` mode matrices computed for a system without aberration. They are generated using [Generate_theoretical_modes/Generate_modes.ipynb](./Generate_theoretical_modes/Generate_modes.ipynb).
- `modes_in_after_correction.npy`: Change of basis matrix between the input mode basis and the input pixel basis after aberration correction optimization.
- `modes_out_after_correction.npy`: Change of basis matrix between the output mode basis and the output pixel basis after aberration correction optimization.
- `mask_near_degenerate.npy`: a mask of the same size as the mode basis TM that represents the blocks of quasi-degenerate modes. 
- `TM_XX_optimization_results.npz` with XX = 17, 25, 35 and 50. Results of the aberration correction relative to the corresponding full pixel basis transmission matrix `TMXX`.

## /Generate_theoretical_modes

Calculation of the theoretical fiber modes.

**Requires:** [pyMMF](https://github.com/wavefrontshaping/pyMMF)

See section 2.1 of the Supplementary Information.

- [Generate_theoretical_modes/Generate_modes.ipynb](./Generate_theoretical_modes/Generate_modes.ipynb): 
Jupyter notebook containing an sample code to compute the theoretical mode profiles knowing the properties of the multimode fiber.
- [Generate_theoretical_modes/functions.py](./Generate_theoretical_modes/functions.py): 
Some useful functions to generate the plots.

## /Aberration_correction

See section 2.2 of the Supplementary Information.

**Requires:** [PyTorch](https://www.pytorch.org)

- [Aberration_correction/Demo_correction_aberration.ipynb](./Aberration_correction/Demo_correction_aberration.ipynb): 
Demo code to use the aberration correction model based on PyTorch framework.
It requires a TM measured in the pixel basis and the theoretical modes.
It learns the aberrations and misalignments of the optical system and compensate for them. It outputs a TM in the basis of the fiber modes.

- [Aberration_correction/Compare_optimization_results.ipynb](./Aberration_correction/Compare_optimization_results.ipynb): 
Code to compare the results of the optimization for different values of the deformation applied. 
Uses the `TM_XX_optimization_results.npz` data files. 
Corresponds to the results presented in the Section S5 of the Supplementary Information of manuscript.

- [Aberration_correction/functions.py](./Aberration_correction/functions.py): 
Some useful functions to generate the plots.

- [PyTorchAberrations/aberration_models.py](./Aberration_correction/PyTorchAberrations/aberration_models.py):
The PyTorch model to apply a set of aberrations and a deformation to the change of basis matrices.

- [PyTorchAberrations/aberration_layers.py](./Aberration_correction/PyTorchAberrations/aberration_layers.py):
Individual custom layers corresponding to each aberration/deformation we can introduce in the model.

- [PyTorchAberrations/aberration_functions.py](./Aberration_correction/PyTorchAberrations/aberration_functions.py):
Useful PyTorch functions, in particular to handle complex linear operations using two layers for the real and imaginary parts.

## /Analysis
Processing of the results and creation of the plots.

* [Analysis_Deformation.ipynb](./Analysis/Analysis_Deformation.ipynb):
Data processing and creation of the figures in the main text of the article.

* [Figures_SI.ipynb](./Analysis/Figures_SI.ipynb):
Data processing and creation of the figures for the Supplementary Information.

* [functions.py](./Analysis/functions.py):
Some useful functions to generate the plots.

## /Layout

Generation of input mask on the digital micro-mirror device (DMD)


See section 1.2 of the Supplementary Information.


**Requires:** [SLMLayout](https://github.com/wavefrontshaping/Layout)

* [Demo_layout.ipynb](./Layout/Demo_layout.ipynb):
A sample code to generate input patterns to send on the DMD.

## More information

Visit our website on [Wavefrontshaping.net](https://wavefrontshaping.net) for more information, codes and tutorials.





