import optuna


def hyperparameter_tuning(network_class, hyperparameter_config, x_train, y_train, x_val, y_val):
    """
    Perform hyperparameter tuning using Optuna.

    :param network_class: The class of the network (e.g., Network, ResNetwork).
    :param hyperparameter_config: A dictionary containing hyperparameter ranges.
    :param x_train, y_train: Training data and labels.
    :param x_val, y_val: Validation data and labels.
    :return: Best hyperparameters found by Optuna.
    """
    # Create an Optuna study for maximizing the validation accuracy
    study = optuna.create_study(direction='maximize')
    optuna.logging.set_verbosity(optuna.logging.ERROR)

    # Start the optimization process
    study.optimize(
        lambda trial: _objective(trial, network_class, hyperparameter_config, x_train, y_train, x_val, y_val),
        n_trials=100)

    # Return the best hyperparameters found by Optuna
    return study.best_params


def _objective(trial, network_class, hyperparameter_config, x_train, y_train, x_val, y_val):
    """
    Objective function for Optuna that dynamically uses hyperparameters from the provided config.

    :param trial: Optuna trial object.
    :param network_class: The class of the network (e.g., Network, ResNetwork).
    :param hyperparameter_config: A dictionary containing hyperparameter ranges.
    :param x_train, y_train, x_val, y_val: Training and validation data.
    :return: Validation accuracy for the given trial.
    """
    # Use hyperparameter configuration
    learning_rate = trial.suggest_float('learning_rate', hyperparameter_config['learning_rate']['low'],
                                        hyperparameter_config['learning_rate']['high'])

    epochs = trial.suggest_int('epochs', hyperparameter_config['epochs']['low'],
                               hyperparameter_config['epochs']['high'])

    hidden_layer_1 = trial.suggest_int('hidden_layer_1', hyperparameter_config['hidden_layer_1']['low'],
                                       hyperparameter_config['hidden_layer_1']['high'])

    hidden_layer_2 = trial.suggest_int('hidden_layer_2', hyperparameter_config['hidden_layer_2']['low'],
                                       hyperparameter_config['hidden_layer_2']['high'])

    # Define the network with these hyperparameters
    nn = network_class(
        sizes=[x_train.shape[1], hidden_layer_1, hidden_layer_2, y_train.shape[1]],
        learning_rate=learning_rate,
        epochs=epochs
    )

    # Train the network using the training data and validation data
    nn.fit(x_train, y_train, x_val, y_val, verbose=False)

    # Evaluate the model on the validation set
    val_accuracy = nn.compute_accuracy(x_val, y_val)

    return val_accuracy  # Return the validation accuracy to optimize
