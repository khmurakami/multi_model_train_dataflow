#!/usr/bin/env python
# coding=utf-8

import json

def read_parameters_json(json_file):

    """Read the parameters for training a model from a JSON file

    Args:
        json_file (json): configuration of the model you want to train

    Return:
        params (dict): all the read paramters from the JSON file

    """

    # Open file with the ability to write to the file
    with open(file_name, "w") as data_file:
        params = json.dump(raw_json, data_file, indent=4, sort_keys=True)

    return params

def read_parameters_yaml(yaml_file):

    pass

def return_json_file(raw_json, file_name):

    """Returns nicely formated json file as .json. Used for debugging.

    Args:
        param raw_json(dict):    Takes in a json dict.
        param file_name(string): Name of the file name you want to write too.
    Return:
        True (boolean):          Return True if the function finished properly
                                 (for now)
    """

    # Open file with the ability to write to the file
    with open(file_name, "w") as data_file:
        json.dump(raw_json, data_file, indent=4, sort_keys=True)

    return True
