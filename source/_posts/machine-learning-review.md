---
title: machine learning review
date: 2019-01-25 14:20:24
categories:
- Machine Learning
tags:
- Machine Learning
mathjax: True
---

## Bayesian
### Bayes's Theorem
$ P(A|B) = \frac{P(B|A)P(A)}{P(B)} $
prior: $P(\omega)$
likelihood: $P(x|\omega)$
posterior: $P(\omega_i|x) = \frac{P(x|\omega_i)P(\omega_i)}{P(x)} = \frac{P(x|\omega_i)P(\omega_i)}{\sum_{j=1}^k P(x|\omega_j)P(\omega_j)} $

**Optimal Bayes Decision Rule: minimize the probability of error.**
&ensp;&ensp;&ensp;&ensp;if $P(\omega_1|x) > P(\omega_2|x)$ then True state of nature $=\omega_1$;
&ensp;&ensp;&ensp;&ensp;if $P(\omega_1|x) < P(\omega_2|x)$ then True state of nature $=\omega_2$.
> Prove: For a particular $x$, 
&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;$P(error|x) = P(\omega_1|x)$ if \omegae decide $\omega_2$;
&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;$P(error|x) = P(\omega_2|x)$ if we decide $\omega_1$.
Bayes Decision Rule:Decide $\omega_1$ if $P(\omega_1|x) > P(\omega_2|x)$;otherwise decide $\omega_2$.
Therefore: $P(error|x) = min[P(\omega_1|x),P(\omega_2|x)]$.
The unconditional error $P(error)$ obtained by integration over all $P(error|x)$.

### Bayesian Decision Theory
c state of nature: $\{\omega_1,,\omega_2,\cdots,\omega_c\}$
a possible actions: $\{\alpha_1,\alpha_2,\cdots,\alpha_a\}$
the loss for taking action $\alpha_i$ when the true state of nature is $\omega_j$: $\lambda(\alpha_i|\omega_j)$
$R(\alpha_i|x) = \sum_{j=1}^{c}\lambda(\alpha_i|\omega_j)P(\omega_j|x)$
Select the action for which the conditional risk $R(\alpha_i|x)$ is minimum.
Bayes Risk: $R = \sum_{over x} R(\alpha_i|x)$.
- Example 1:
action $\alpha_1$: deciding $\omega_1$
action $\alpha_2$: deciding $\omega_2$
$\lambda_{ij} = \lambda(\alpha_i|\omega_j)$
$R(\alpha_1|x) = \lambda_{11}P(\omega_1|x) + \lambda_{12}P(\omega_2|x)$
$R(\alpha_2|x) = \lambda_{21}P(\omega_1|x) + \lambda_{22}P(\omega_2|x)$
if $R(\alpha_1|x) < R(\alpha_2|x)$ , action $\alpha_1$ is taken: deciding $\omega_1$.
- Example 2:
Suppose $\lambda\left(\alpha_{i} | \omega_{j}\right) =
\begin{cases}
0& \text{i = j}\\\\
1& \text{i != j}
\end{cases}$
Conditional risk
$  R\left(\alpha_{i} | x\right)=\sum_{j=1}^{c} \lambda\left(\alpha_{i} | \omega_{j}\right) P\left(\omega_{j} | x\right)  =\sum_{j \neq i} P\left(\omega_{j} | x\right)=1-P\left(\omega_{i} | x\right)  $
Minimizing the risk $\longrightarrow$ Maximizing the posterior $P(\omega_i|x)$.
So we have the discriminant function(max. discriminant corresponds to min. risk):
$ g_{i}(x)=-R\left(\alpha_{i} | x\right) $
$\Longleftrightarrow$
$ g_{i}(x)=P\left(\omega_{i} | x\right) $
$  g_{i}(x)=P(x | \omega_{i}) P\left(\omega_{i}\right)  $
$  g_{i}(x)=\ln P(x | \omega_{i})+\ln P\left(\omega_{i}\right)  $
Set of discriminant functions: $ g_{i}(x), i=1, \cdots, c $
Classifier assigns a feature vector $x$ to class $\omega_i$ if: $ g_{i}(x)>g_{j}(x), \quad \forall j \neq i $

Binary classification $\longrightarrow$ Multi‐class classfication
- One vs. One
$N$ class, design $\frac{N(N-1)}{2}$ classifiers, denote for result.
- One vs. Rest
design $N$ classifiers, choose the one which prediction is positive.
- ECOC (Error‐Correcting Output Codes)
The code consisting of the labels predicted by these classifiers is compared with each line, and the one with the smallest distance between codes is the result.

|         | f1     |  f2    |  f3  |
| --------| -----: | :----: |:----:|
| c1      | -1     |   1    |  -1  |
| c2      | 1      |   -1   |  -1  |
| c3      | -1     |   1    |   1  |


