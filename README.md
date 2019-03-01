# Supervisor comparison

A repo that creates plots and metrics for your supervisors. A supervisor can be seen as a detector if a sample should be considered out of distribution. 

## Detailed description
The idea with this repo is to set up a pre-defined environment, where the user quickly can get access to trained models and datasets. With these datasets you can either train new models, but also use them to test outlier detection through the supervisors. 

A supervisor is a descriminator that decides whether or not an input sample belongs to an in-distribution (Something that my network should be able to handle), or an out-distribution (something that my network was not trained to handle). 

## Required toolboxes
Repo is developed with python 3.6 and torch.
```
seaborn torch numpy matplotlib
```


## Installing and using

```
git clone https://github.com/jenshenriksson/sv-comparison.git
```

```
python example_supervisor.py
```

## Aknowledgements
This is done for the WASP Software and Systems course, that teaches proper ways of versoning, refactoring and testing.  
