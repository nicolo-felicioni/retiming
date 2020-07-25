# Retiming synchronous circuitry implementation

This repository contains a Python 3 implementation of the paper [Retiming synchronous circuitry](https://people.eecs.berkeley.edu/~alanmi/courses/2008_290A/papers/leiserson_tech86.pdf) by Leiserson and Saxe.
The first part of the readme (until the section "Algorithms to be implemented" included) is a summary of the original paper with an highlight to the theoretical details relevant to the implementation. The second part is about the actual implementation.

## The problem

### Introduction
The goal is to decrease as much as possible the clock period of a synchronous circuit by means of a technique called _retiming_, that does not increase latency. One example is the following of the digital correlator, taken from the original paper. The correlator takes a stream of bits x<sub>0</sub>, x<sub>1</sub>, x<sub>2</sub>, ... as input and compares it with a fixed-length pattern          a<sub>0</sub>, a<sub>1</sub>, ... a<sub>k</sub>. After receiving each input x<sub>i</sub> (i &ge; k), the correlation produces as output the number of matches:

<center> y<sub>i</sub> = Σ<sup>k</sup><sub>j=0</sub>  δ(x<sub>i − j</sub>, a<sub>j</sub>) , </center>

where δ(x, y)=1 if x=y, and  δ(x, y)=0 otherwise (comparison function).
The following figure shows a design of a simple correlator for k = 3.
<p align="center">
    <img width="10%" src="https://seeklogo.com/images/T/twitter-logo-A84FE9258E-seeklogo.com.png" />
</p>
Suppose that each adder has a propagation delay of 7 esec (a fictitious amount of time), and each comparator has a delay of 3 esec. Then the clock period must be at least 24 esec, i.e. the time for a signal to propagate from the register on the connection labeled A through one comparator and three adders. 
Better performance can be reached by removing register on connection A and inserting a new register on connection B, as shown in the following figure.
<p align="center">
    <img width="10%" src="https://seeklogo.com/images/T/twitter-logo-A84FE9258E-seeklogo.com.png" />
</p>
The clock period now is decreased by 7 exec. 

**Retiming** is the technique of inserting and deleting registers in such a way as to preserve function, and it can be used to produce faster circuits.


### Preliminaries

Circuits are modelled as graphs, as shown in the figure below, where there is the graph of the first correlator. 
<p align="center">
    <img width="10%" src="https://seeklogo.com/images/T/twitter-logo-A84FE9258E-seeklogo.com.png" />
</p>

Each vertex is weighted with its propagation delay *d(v)*, while each edge is weighted with its register count *w(e)*.
In the following they are called **delay** and **weight**.
A simple path p is a path from a vertex to another that contains no vertex twice. The notion of delay *d*(*p)* and weight *w*(*p)* for paths as the sum of delays of the vertices in the path and the sum of weights of the edges in the path.

#### Constraints for physical feasibility
* **D1.** d(v) nonnegative for each vertex
* **W1.** w(v) nonnegative integer for each edge
* **W2.** In any directed cycle of G, there exists at least one edge e with w\(e\)&gt;0.

#### CP algorithm
This is an algorithm that computes the clock period of a circuit. The clock period is defined as:
<center>cp(G)=max{d(p) : w(p)=0} </center>

**Algorithm CP:**
1. Let G0 be the G subgraph with only those edges e with w(e)=0
2. By **W2.**, G0 is acyclic. Perform a topological sort such that vertices are totally ordered with respect to the edges (if u -> v, then u<v). 
3. Visit vertices in that order, for each v compute Δ(v) as follows:
	* If no in_edge(v), Δ(v):=d(v)
	* otherwise, Δ(v):=d(v)+max{Δ(u): u is incoming to v with edge e and w(e)=0}
4.  cp(G) = max<sub>v</sub> Δ(v)

Running time is O(|E|).

### Retiming

<p align="center">
    <img width="10%" src="https://seeklogo.com/images/T/twitter-logo-A84FE9258E-seeklogo.com.png" />
</p>

A retiming is a function r that assigns to each vertex an integer (positive or negative) r(v). It specifies a transformation to a new graph G<sub>r</sub> that has the same vertices, edges and delays, but different weights such that for each edge e that links u to v,  w<sub>r</sub>(e)=w(e)+r(v)-r(u).

#### Condition for a legal retiming
It can be proved that a retiming r is legal if G<sub>r</sub> satisfies W1.

## Algorithms to be implemented

#### WD
Algorithm to compute the matrices W and D (shape: |V| x |V| ), defined like this:
* W(u,v) = min{w(p): u -p-> v} (minimum number of registers in a path between u and v)
* D(u,v) = max{d(p): u-p->v and w(p)=W(u,v)}
The theoretical details of this algorithm are not relevant for the implementation, for more about the implementation details see the doc.
#### OPT1
Algorithm that given a circuit G, returns the optimal cp and the corresponding retiming r.
First, it is important to say that the following **conditions for a legal retiming** can be proved:
1. r(u)-r(v) &le; w(e) for every edge u-e->v AND
2.  r(u)-r(v) &le; w(e) for all vertices u,v such that D(u,v) &gt; c.

**OPT1 algorithm:**
1. Compute W and D with WD algorithm
2. Sort the elements in the range of D
3. Binary search among the elements of D for the minimum achievable cp, to test whether each potential clock period c is feasible, apply the Bellman-Ford algorithm to determine if the conditions for a legal retiming can be satisfied.
4. For the minimum achievable clock period, use the values for the r(v) found by BF.

**BF algorithm:**


#### FEAS
#### OPT2


## Algorithms implementation


## Test creation

### Random retiming legal by construction

## Assessment


## Requirements
In this implementation were used:
* **Python**: version 3.7.4. 
* **Pip**: version 20.1.1.

It is suggested to use a [python virtual environment](https://docs.python.org/3/library/venv.html).

After activating the virtual environment, install python dependecies with the following bash command:
```shell script
pip install -r requirements.txt
```
