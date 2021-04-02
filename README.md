# mlfaker

[Put CI Badge Here]

**Note that this is a stub README that contains boilerplate for many of the common operations done inside an ML repo. You should customize it appropriately for your specific project**

Quickly and easily mock data in pandas for various ML applications

# Setup

## Basics

Clone the code to your machine using the standard Git clone command. If you have SSH keys setup the command is:

```bash
git clone git@github.com:manifoldai/mlfaker.git
```


## Project goal

This project serves two primary goals:

1. Make it easy to mock data for accelerate development for collaborative ML teams
   (think about mocking APIs before the backend is built)
2. Generate structured data where the user provides the causal structure between
   variables
   
   
### 1. Mock data

We tend to find ourselves mocking tabular data in a ML projects. Here are few scenarios:

1. Tests
2. Waiting on other developers to build ETL process while building out other pieces
3. Test a model on noise (target independent of features) or basic linear data (y
   linearly dependent on features)

All of these tasks are straightforward in pandas, but time consuming and repetitive.
This project aims to make these tasks one-liners


### 2. Generate data from causal structure

Thinking of data from generative processes and having a playground to generate data has
immense value in ML. Let's say you have a conceptual model (i.e. a causal DAG) of your
system. What does data generated from this look like? Does this look like my data? What
happens if I have a collider in my system? My linear model says this effect is positive,
does that match the generative process? Is conditioning on other covariates leading me
to draw the wrong conclusions? These are important questions, and ones we hope you're
asking yourself..  Building simple models and testing them is the key to gaining
intuition and to understand what's actually going on.

This project aims to provide a nice user-face for engineers and scientists to mock data
from a casual DAG:

DAG represented as a matrix -> Casual DAG -> generated data
