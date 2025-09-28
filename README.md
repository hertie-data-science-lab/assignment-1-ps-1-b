[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/WPCxMZMM)
# Deep Learning E1394 - Problem Set 1

## Downloading the assignment
Clone the assigment repository which was created for your team. 
```
git clone https://github.com/hertie-data-science-lab/<repository-name>
```
If you feel more comfortable using git UI tools instead of the command line, feel free to use them.

## Dependency management

We recommend to use conda and pip for dependency management and setup an isolated conda environment for every Python assignment and project. This helps to cleanly management dependencies and makes it easier for other people to reproduce your results.

A conda environment with the specified dependencies can be created with:
```
conda create --name <some-name> --file requirements.txt
```

## Submitting your work

Once the deadline has passed, you are not able to push to your repository anymore or update it in any other way and all changes on the main branch will be considered for grading. In other words, there is no "submit" button which you need to press. You just need to ensure that all your work has been pushed / uploaded to GitHub before the deadline.

Uploading a new or edited file to GitHub involves three steps:
1. Add the file to the staging area:
    ```
    git add <file to be added>
    ```
2. Create a commit:
    ```
    git commit -m <commit msg - briefly describe your change>
    ```
3. Push the local commit to GitHub:
    ```
    git push
    ```

To avoid conflicts when multiple people are working on the same file, we recommend to push your changes as frequently as possible and always pull the latest changes from GitHub before you start working. Also, as a nice teammate, test your changes before you push them and use meaninful commit messages to make it easy to understand your changes.

## Asking for help

If you have important questions or believe to have spotted an error, please open an issue on your repository and mention the teaching assistant with @chiara-fb .

## Useful Links
* Git best practices: https://gist.github.com/luismts/495d982e8c5b1a0ced4a57cf3d93cf60
* How to write a good git commit message: https://cbea.ms/git-commit/
* Book "Dive into Deep Learning": https://d2l.ai/
* Keras model training and evaluation: https://www.tensorflow.org/guide/keras/train_and_evaluate
* Scheduled or adaptive learning rate: https://towardsdatascience.com/learning-rate-schedules-and-adaptive-learning-rate-methods-for-deep-learning-2c8f433990d1
* Tutorial on pushing and submitting work with GitHub classroom: https://www.youtube.com/watch?v=jXpT8eOzzCM
* Neural network from scratch:
    * https://towardsdatascience.com/math-neural-network-from-scratch-in-python-d6da9f29ce65
    * https://pythonalgos.com/create-a-neural-network-from-scratch-in-python-3/
    * https://github.com/casperbh96/Neural-Network-From-Scratch/blob/master/NN_From_Scratch.ipynb
    * https://www.codingame.com/playgrounds/59631/neural-network-xor-example-from-scratch-no-libs
