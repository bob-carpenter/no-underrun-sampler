# No-Underrun Sampler (NURS)

This is the reference implementation of the No-Underrun Sampler
(NURS), a gradient-free Markov chain Monte Carlo (MCMC) algorithm.
NURS is an implementable form of the [Hit-and-Run Sampler](https://www.numdam.org/item/JSFS_2007__148_4_5_0/) (HR) based on
ideas from the
[No-U-Turn Sampler](https://en.wikipedia.org/wiki/Hamiltonian_Monte_Carlo#No_U-Turn_Sampler)
(NUTS), which is in turn based on ideas from [Slice Sampling](https://en.wikipedia.org/wiki/Slice_sampling).

The motivations for NURS is to provide a superior alternative to Gibbs
in mixing time (linear vs. quadratic in condition number) that can be
implemented in parallel and requires only a log density function (not
a gradient).

## Reference implementations

* NURS:  `python/nurs/nurs.py`
* NURS with step size adaptation:  `python/nurs/nurs_step_adapt.py`

## Running the built-in examples

```python
cd python/nurs
python3 examples.py
```

In `examples.py`, Each example is formatted as a function with calls at the end.
Comment out tests to speed up testing specific examples.

## References

#### NURS

* Nawaf Bou-Rabee, Bob Carpenter, Sifan Liu, Stefan Oberdörster. 2025.
[The No-Underrun Sampler: A locally adaptive, gradient free, MCMC
method](https://arxiv.org/abs/2501.18548v2). *arXiv* 2501.18548 v2.

#### Background

* Andersen, Hans C.; Diaconis, Persi. 2007.  [Hit and run as a unifying
  device](https://www.numdam.org/item/JSFS_2007__148_4_5_0/). *Journal
  de la Société Française de Statistique* 148(4):5--28.

* Matthew D. Hoffman and Andrew
  Gelman.  2014. [The No-U-Turn Sampler: Adaptively setting path lengths in Hamiltonian Monte Carlo](https://www.jmlr.org/papers/volume15/hoffman14a/hoffman14a.pdf). *Journal
  of Machine Learning Research* 15(2014):1593-1623.

* Neal, Radford M. 2003. [Slice sampling](https://projecteuclid.org/journals/annals-of-statistics/volume-31/issue-3/Slice-sampling/10.1214/aos/1056562461.full). *Annals of Statistics* 31(3): 705–767.


