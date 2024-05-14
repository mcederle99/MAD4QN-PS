# WORK IN PROGRESS
# A Distributed Approach to Autonomous Intersection Management via Multi-Agent Reinforcement Learning
## Matteo Cederle, Marco Fabris and Gian Antonio Susto
#### Department of Information Engineering, University of Padova, 35131 Padua via Gradenigo 6/B, Italy

![](docs/_static/amco.png)

### Abstract
Autonomous intersection management (AIM) poses significant challenges due to the intricate nature of real-world traffic scenarios and the need for a highly expensive centralised server in charge of simultaneously controlling all the vehicles. This study addresses such issues by proposing a novel distributed approach to AIM utilizing multi-agent reinforcement learning (MARL). We show that by leveraging the 3D surround view technology for advanced assistance systems, autonomous vehicles can accurately navigate intersection scenarios without needing any centralised controller. The contributions of this paper thus include a MARL-based algorithm for the autonomous management of a 4-way intersection and also the introduction of a new strategy called prioritised scenario replay for improved training efficacy. 
We validate our approach as an innovative alternative to conventional centralised AIM techniques, ensuring the full reproducibility of our results. Specifically, experiments conducted in virtual environments using the SMARTS platform highlight its superiority over benchmarks across various metrics.

## How to train and evaluate the algorithm
1. Clone the repository
2. [Set up SMARTS (Scalable Multi-Agent Reinforcement Learning Training School)](https://smarts.readthedocs.io/en/latest/setup.html)

   [SMARTS repository](https://github.com/huawei-noah/SMARTS), [SMARTS paper](https://arxiv.org/abs/2010.09776)

   ![](docs/_static/smarts_envision.gif)
3. Train the agents: 
   ```
   scl run main.py --headless
   ```
4. After the agents have been trained you can evaluate the results and visualize the simulation:
   ```
   scl run --envision main.py --load_checkpoint
   ```
5. To collect the data needed to evaluate the metrics presented in the paper run:
   ```
   ./data_collection.sh
   ```
   for MAD4QN-PS. For the random policy run instead:
   ```
   ./data_collection_random.sh
   ```
   The data for the other baselines are stored in the folder baselines
6. To plot the results, use the notebook plots.ipynb

## Cite this work
If you our work interesting for your research, please cite the [paper](). In BibTeX format:

```bibtex
@misc{NOSTRO PAPER,
    title={SMARTS: Scalable Multi-Agent Reinforcement Learning Training School for Autonomous Driving},
    author={Ming Zhou and Jun Luo and Julian Villella and Yaodong Yang and David Rusu and Jiayu Miao and Weinan Zhang and Montgomery Alban and Iman Fadakar and Zheng Chen and Aurora Chongxi Huang and Ying Wen and Kimia Hassanzadeh and Daniel Graves and Dong Chen and Zhengbang Zhu and Nhat Nguyen and Mohamed Elsayed and Kun Shao and Sanjeevan Ahilan and Baokuan Zhang and Jiannan Wu and Zhengang Fu and Kasra Rezaee and Peyman Yadmellat and Mohsen Rohani and Nicolas Perez Nieves and Yihan Ni and Seyedershad Banijamali and Alexander Cowen Rivers and Zheng Tian and Daniel Palenicek and Haitham bou Ammar and Hongbo Zhang and Wulong Liu and Jianye Hao and Jun Wang},
    url={https://arxiv.org/abs/2010.09776},
    primaryClass={cs.MA},
    booktitle={Proceedings of the 4th Conference on Robot Learning (CoRL)},
    year={2020},
    month={11}
}
```
