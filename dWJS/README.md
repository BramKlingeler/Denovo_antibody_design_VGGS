# discrete Walk-Jump Sampling (dWJS) for Heavy Chain sequences only

This is repository is the adapted code of the open source repository for [discrete Walk-Jump Sampling](https://arxiv.org/abs/2306.12360) developed by [ncfrey](https://github.com/ncfrey), [djberenberg](https://github.com/djberenberg), [kleinhenz](https://github.com/kleinhenz), and [saeedsaremi](https://github.com/saeedsaremi), from [Prescient Design, a Genentech accelerator.](https://gene.com/prescient) which can be found in (https://github.com/prescient-design/walk-jump)

## Note
This code has been edited from the original repository to only sample Heavy Chain sequences. Configuration in src/walkjump/hydra_config/ gave the best results for both lowest training loss and highest sampling results.


## Setup
Assuming you have [miniconda](https://docs.conda.io/en/latest/miniconda.html) installed, clone the repository, navigate inside, and run:
```bash
./scripts/install.sh
```

## Training
The entrypoint `walkjump_train` is the main driver for training and accepts parameters using Hydra syntax.
The available parameters for configuration can be found by running `train` --help or by looking in the `src/walkjump/hydra_config` directory

## Sampling
The entrypoint `walkjump_sample` is the main driver for training and accepts parameters using Hydra syntax.
The available parameters for configuration can be found by running `sample` --help or by looking in the `src/walkjump/hydra_config` directory


## License
Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at https://www.apache.org/licenses/LICENSE-2.0.

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.


## Citations
If you use the code and/or model, please cite the paper of Frey et al. the dWJS:
```
@article{frey2023protein,
      title={Protein Discovery with Discrete Walk-Jump Sampling},
      author={Nathan C. Frey and Daniel Berenberg and Karina Zadorozhny and Joseph Kleinhenz and Julien Lafrance-Vanasse and Isidro Hotzel and Yan Wu and Stephen Ra and Richard Bonneau and Kyunghyun Cho and Andreas Loukas and Vladimir Gligorijevic and Saeed Saremi},
      year={2023},
      eprint={2306.12360},
      archivePrefix={arXiv},
      primaryClass={q-bio.BM}
}
```
