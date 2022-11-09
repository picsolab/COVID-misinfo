# Predicting Twitter Users' Susceptibility to COVID Misinformation

This repository contains a PyTorch implementation of our deep learning method to predict a Twitter user's susceptibility to COVID-19 misinformation, presented in our ICWSM 2022 paper https://ojs.aaai.org/index.php/ICWSM/article/view/19353.


# Abstract

Though significant efforts such as removing false claims and promoting reliable sources have been increased to combat COVID-19 misinfodemic, it remains an unsolved societal challenge if lacking a proper understanding of susceptible online users, i.e., those who are likely to be attracted by, believe and spread misinformation. This study attempts to answer who constitutes the population vulnerable to the online misinformation in the pandemic, and what are the robust features and short-term behavior signals that distinguish susceptible users from others. Using a 6-month longitudinal user panel on Twitter collected from a geopolitically diverse network-stratified samples in the US, we distinguish different types of users, ranging from social bots to humans with various level of engagement with COVID-related misinformation. We then identify users' online features and situational predictors that correlate with their susceptibility to COVID-19 misinformation. This work brings unique contributions: First, contrary to the prior studies on bot influence, our analysis shows that social bots' contribution to misinformation sharing was surprisingly low, and human-like users' misinformation behaviors exhibit heterogeneity and temporal variability. While the sharing of misinformation was highly concentrated, the risk of occasionally sharing misinformation for average users remained alarmingly high. Second, our findings highlight the political sensitivity activeness and responsiveness to emotionally-charged content among susceptible users. Third, we demonstrate a feasible solution to efficiently predict users' transient susceptibility solely based on their short-term news consumption and exposure from their networks. Our work has an implication in designing effective intervention mechanism to mitigate the misinformation dissipation.

# Run
You might run the code using 3 options: to predict a user's future susceptibility, via (1) `ego,` using the user's own past activities, (2) `ngh`, based on a user's neighbors' past activitities, or (3) `both`, using both. The file `output` will keep a record of all code running logs, as well as evaluation performance (accuracy, AUC, precision, recall, F1).
```
nohup python run.py --filename data.json --verbose ego >> output
nohup python run.py --filename data.json --verbose ngh >> output
nohup python run.py --filename data.json --verbose both >> output
```

# Citation

If you use any of the resources provided on this page, please cite the following paper.
```
@inproceedings{teng2022characterizing,
  title={Characterizing User Susceptibility to COVID-19 Misinformation on Twitter},
  author={Teng, Xian and Lin, Yu-Ru and Chung, Wen-Ting and Li, Ang and Kovashka, Adriana},
  booktitle={Proceedings of the International AAAI Conference on Web and Social Media},
  volume={16},
  pages={1005--1016},
  year={2022}
}
```
