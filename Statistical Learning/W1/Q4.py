# Sampling from a Distribution # I acknowledge that I got help from online resources for this part

#The finite case:

#Let p be a discrete probability distribution over a set {x1,x2, ..., xm} with m items. To generate a sample X∼p:
# 1: Compute the probability vector [p(x1),…,p(xm)]
# 2: Compute the cumulative probability vector [F(x0),F(x1),…,F(xm)], where F(xi):=∑ij=1p(xj), and F(x0):=0
# 3: Divide the unit interval [0,1] into intervals Ij:=(F(xj−1),F(xj)]  (j=1,…,n)
# 4: Sample a uniform random number U∼[0,1]
# 5: Find j such that U∈Ij
# 6: Return xj

# That this algorithm samples from the correct distribution is a consequence of the following computation:
# Pr[U∈Ij]=Pr[F(xj−1)<U≤F(xj))]=F(xj)−F(xj−1)=p(xj)
# takes an integer parameter n (indicating the set of partitions Πn from which we wish to sample)
# and returns the corresponding function k↦kn/(k!eBn)The rest of the code below implements the simulation procedure described in steps 3–6.



import numpy
from mpmath import bell
from mpmath import e
from mpmath import factorial
from mpmath import power


def make_pdf(n):
    """Return a Dobinski probability p(k) := k^n/(k!eB_n)."""

    def pdf(k):
        numer = power(k, n)
        denom = factorial(k) * e * bell(n)
        return numer / denom

    return pdf


def make_cdf(pdf):
    """Return cumulative probability function for pdf."""

    def cdf(k):
        return sum(pdf(j) for j in xrange(k + 1))

    return cdf


def find_interval(u, cdf):
    """Find k such that u falls in I_k of given cdf."""
    k = 1
    while True:
        (left, right) = (cdf(k - 1), cdf(k))
        if left & lt; u & lt;= right:
            return k
        k += 1


def simulate(cdf, rng):
    """Simulate from pdf using rng::numpy.random.RandomState."""
    u = rng.uniform()
    return find_interval(u, cdf)