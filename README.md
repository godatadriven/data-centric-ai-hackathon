# Data Centric AI hackathon

There is a lot of buzz around data centric AI, but we're not an authority in it... yet! Let’s boost our skills in this
hackathon.

If this hackathon is a success, we might spin it into a code breakfast, workshop, training, competition a la Dropblocks,
etc., so please participate and give me feedback on the hackathon!

I'll bring some starter code and ideas to get you going fast.

Note: The hackathon is not limited to DS/MLE -- as long as you like coding and optimisation, you should be good to go!

## Background

The example problem of the hackathon is to simulate something that is taken from a real problem we have at Schiphol in
the Deep Turnaround project. There we have a huge data set with annotations for the model to train on. However, some
annotations need to become more specific: instead of predicting that a luggage loading vehicle is present, we want to be
able to discern whether the vehicle is active at the front or at the rear of the plane. We can of course migrate the
annotations to a new version and start training the model with the new labels, but initially we have very limited amount
of labels in the new data set.

Unfortunately we cannot share the images from Schiphol. The proposal is to simulate the problem using a super simple
dataset: MNIST digits. We modify the labels in MNIST in the following way. The 7s and the 1s are mapped to a combined
class: C. A model is trained that predicts the labels for MNIST. This model is not able to distinguish 7s from 1s. The
goal now is to train a new model that can distinguish 7s from 1s, but with as little as possible annotations needed.

Because we map original labels to a common class, there is no need to do actual annotations. We just provide a
annotation function (i.e. get the original label, not the combined class C) that can be applied to a specific sample.
The goal of the hackathon is to achieve a 90% accuracy of discerning 7s from 1s with as least as possible applications
of the unmapping function.

Some ideas that the participant might explore are:

* Iteratively fitting a model purely on 7s and 1s and use that model in an active-learning setting
* Clustering in latent space
* Transfer learning
* other smart tricks

## Setting up

```shell
python3.10 -m venv venv
source venv/bin/activate
pip install -e .
```

Start Jupyter notebook (an example notebook is given in `notebooks/example-naive-annotations.ipynb`)

```shell
jupyter lab .
```

## Rules

Cheating is easy! Just download the original MNIST data set or download a pre-fitted model are two examples. Keep in
mind that the goal of the hackathon is to get experience with data centric AI. If one of your solutions feels like 
cheating, it is cheating 🙂

## Some ideas

* Use an unsupervised learning technique such as clustering. Use this clustering as an initial method for getting the initial labels.
* Use an existing tool to help you with it. Consider installing [`bulk`](https://github.com/koaning/bulk).
* Go nuts: implement [active learning](https://www.datacamp.com/tutorial/active-learning) yourself.



