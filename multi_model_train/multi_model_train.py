#!/usr/bin/env python
# coding=utf-8

# This file handles the parallelization of training using Apache Beam and
# Google Cloud Platform Dataflow

# Handles Python 2 for Apache Beam
from __future__ import unicode_literals

# System Libraries
import itertools
import uuid
import json
import random
import string

# Third Party Libraries
import apache_beam as beam
import tensorflow as tf

# Third Party imports
from sklearn.model_selection import cross_validate
from apache_beam.io import WriteToText

# My Libraries
from utils import read_parameters_json


class MultiModelTrain():

    def __init__(self, json_file="None"):

        if json_file:

            # Get all the parameters of the JSON configuration file
            self.params = read_parameters_json(json_file)
            self.options = beam.options.pipeline_options.PipelineOptions()

            # Google Cloud possible options
            self.google_cloud_options = self.options.view_as(beam.options.pipeline_options.GoogleCloudOptions)
            self.google_cloud_options.project = params['google_cloud_project_name']
            self.google_cloud_options.job_name = params['google_cloud_job_name']
            self.google_cloud_options.staging_location = parmas['google_cloud_bucket_staging']
            self.google_cloud_options.temp_location = params['google_cloud_bucket_temp']

            # Apache Beam Worker Options
            self.worker_options = self.options.view_as(beam.options.pipeline_options.WorkerOptions)
            self.worker_options.max_num_workers = params['google_dataflow_max_workers']
            self.worker_options.num_workers = params['google_dataflow_workers']
            self.worker_options.disk_size_gb = params['google_dataflow_disk_gb']
            self.options.view_as(beam.options.pipeline_options.StandardOptions).runner = params['google_dataflow_runner']
            self.p = beam.Pipeline(options=self.options)

    def set_parameters(self, params):

        """Create a Beam object and write to Google Big Query

        """

        (self.p | 'init' >> beam.Create(params)
           | 'train' >> beam.Map(self.set_train)
           | 'output' >> beam.io.WriteToBigQuery(
                                    "multi-model-dataflow:dataset.example_set",
                                    schema='accuracy:FLOAT, loss:FLOAT, model_id:STRING, param:STRING',
                                    write_disposition='WRITE_APPEND',
                                    create_disposition='CREATE_IF_NEEDED')
         )

    def set_train(self, param):

        """Set the training to be used in Beam

        Args:

        Return:

        """

        import tensorflow as tf
        from sklearn.model_selection import train_test_split

        model_id = self.randomString()

        # Load iris dataset
        print("Loading Iris Data")
        iris = tf.contrib.learn.datasets.base.load_iris()
        train_x, test_x, train_y, test_y = train_test_split(
            iris.data, iris.target, test_size=params["test_size"], random_state=0
        )

        # https://www.tensorflow.org/get_started/tflearn
        feature_columns = [tf.contrib.layers.real_valued_column("", dimension=4)]
        classifier = tf.contrib.learn.DNNClassifier(feature_columns=feature_columns,
                                                    hidden_units=params["hidden_units"],
                                                    dropout=params["dropout"],
                                                    n_classes=3,
                                                    model_dir="gs://multi_model_bucket/models/%s"% model_id)

        print("Training Model Now")
        classifier.fit(x=train_x,
                       y=train_y,
                       steps=param["steps"],
                       batch_size=params["batch_size"])

        result = classifier.evaluate(x=test_x, y=test_y)

        ret = {
               "accuracy": float(result["accuracy"]),
               "loss": float(result["loss"]),
               "model_id": str(model_id),
               "param": json.dumps(param)
               }

        return ret

    def run_training(self):

        """Run the Beam

        """

        print("Sending the Process to Cloud")
        self.p.run()

    def randomString(self, stringLength=10):

        """Generate a random string of fixed length

         """

        import string
        import random

        letters = string.ascii_lowercase
        return ''.join(random.choice(letters) for i in range(stringLength))


    def dict_product(self, param_grid):

        """Create all the difference configurations of params a model can be trained in

        Args:
            param_grid (list): A list of all the possible params

        Return:
            param_permutations (dict): A dictionary of all the possible permutations


        """

        param_permutations = (dict(zip(param_grid, x)) for x in itertools.product(*param_grid.values()))

        return param_permutations


    def all_parameters(self):

        """All the parameters you want to permuatate

        """

        # Create a Dictionary
        param_grid = {
                      'hidden_units': [[10, 20, 10], [20, 40, 20], [100, 200, 100]],
                      'dropout': [0.1, 0.2, 0.5, 0.8],
                      'steps': [20000, 50000, 100000]
                     }

        params = list(self.dict_product(param_grid))

        return params


if __name__ == "__main__":

    test = MultiModelTrain()
    params = test.all_parameters()
    test.set_parameters(params)
    test.run_training()
