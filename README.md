# Learning from an Exploring Demonstrator: Optimal Reward Estimation for Bandits

This repository provides implementation for experiments as described in the paper:

[Learning from an Exploring Demonstrator: Optimal Reward Estimation for Bandits](https://arxiv.org/abs/2106.14866)  
[Wenshuo Guo](https://people.eecs.berkeley.edu/~wguo/), [Kumar Krishna Agrawal](https://people.eecs.berkeley.edu/~krishna/), [Aditya Grover](https://aditya-grover.github.io), [Vidya Muthukumar](https://vmuthukumar.ece.gatech.edu), [Ashwin Pananjady](https://sites.gatech.edu/ashwin-pananjady/)


## Requirements

```
pip install -e . -r requirements.txt
```

## Experiments
Experiments are organized in notebooks. Please see example usage
```
jupyter-lab two_arms.ipynb
```

To cite this paper:
```
@article{guo2021exploringdemonstrator,
    title={Learning from an Exploring Demonstrator: Optimal Reward Estimation for Bandits},
    author={Wenshuo Guo, Kumar Krishna Agrawal, Aditya Grover, Vidya Muthukumar, Ashwin Pananjady},
    journal={arXiv preprint arXiv:2106.14866}.
    year={2021}
}
```
  
## Acknowledgements
Notebook for the error-landscape in the battery dataset builds significantly on [https://github.com/chueh-ermon/battery-fast-charging-optimization](https://github.com/chueh-ermon/battery-fast-charging-optimization/tree/master/figures/fig3)