#!/usr/bin/env python
# -*- coding: utf-8 -*-


# System Libraries
import itertools
import uuid
import json

# Third Party Libraries
import apache_beam as beam
import tensorflow as tf

# Third Party imports
from sklearn import cross_validation


def dict_product(param_grid):

    """Create all the difference configurations of params a model can be trained in

    Args:
        param_grid (list): A list of all the possible params

    Return:


    """

    param_permutations = (dict(itertools.izip(param, x)) for x in itertools.product(*param.itervalues()))

    return param_permutations


def all_parameters():

    """All the parameters you want to permuatate

    """

    # Create a Dictionary
    param_grid = {'hidden_units': [[10, 20, 10], [20, 40, 20], [100, 200, 100]],
                  'dropout': [0.1, 0.2, 0.5, 0.8],
                  'steps': [20000, 50000, 100000]}

    params = list(dict_product(param_grid))

    return params


class MultiModelTrain():

    def __init__(self):

        self.options = beam.utils.pipeline_options.PipelineOptions()

        # Google Cloud possible options
        self.google_cloud_options = self.options.view_as(beam.utils.pipeline_options.GoogleCloudOptions)
        self.google_cloud_options.project = '{PROJECTID}'
        self.google_cloud_options.job_name = 'tensorflow-gs'
        self.google_cloud_options.staging_location = 'gs://{BUCKET_NAME}/binaries'
        self.google_cloud_options.temp_location = 'gs://{BUCKET_NAME}/temp'

        # Apache Beam Worker Optiosn
        self.worker_options = self.options.view_as(beam.utils.pipeline_options.WorkerOptions)
        self.worker_options.max_num_workers = 6
        self.worker_options.num_workers = 6
        self.worker_options.disk_size_gb = 20
        # worker_options.machine_type = 'n1-standard-16'

        # options.view_as(beam.utils.pipeline_options.StandardOptions).runner = 'DirectRunner'
        self.options.view_as(beam.utils.pipeline_options.StandardOptions).runner = 'DataflowRunner'

        self.p = beam.Pipeline(options=self.options)

    def set_parameters(self, params):

        """Create a Beam object and write to Google Big Query

        """

        (self.p | 'init' >> beam.Create(params)
           | 'train' >> beam.Map(self.set_train)
           | 'output' >> beam.Write(beam.io.BigQuerySink('project:dataset.table',
                                    schema="accuracy:FLOAT, loss:FLOAT, model_id:STRING, param:STRING",
                                    write_disposition=beam.io.BigQueryDisposition.WRITE_APPEND,
                                    create_disposition=beam.io.BigQueryDisposition.CREATE_IF_NEEDED))
         )

    def set_train(self, param):

        """Set the training to be used in Beam

        """

        model_id = str(uuid.uuid4())

        # Load iris dataset
        iris = tf.contrib.learn.datasets.base.load_iris()
        train_x, test_x, train_y, test_y = cross_validation.train_test_split(
            iris.data, iris.target, test_size=0.2, random_state=0
        )

        # https://www.tensorflow.org/get_started/tflearn
        feature_columns = [tf.contrib.layers.real_valued_column("", dimension=4)]
        classifier = tf.contrib.learn.DNNClassifier(feature_columns=feature_columns,
                                                    hidden_units=param['hidden_units'],
                                                    dropout=param['dropout'],
                                                    n_classes=3,
                                                    model_dir='gs://{BUCKET_NAME}/models/%s'% model_id)
        classifier.fit(x=train_x,
                       y=train_y,
                       steps=param['steps'],
                       batch_size=50)
        result = classifier.evaluate(x=test_x, y=test_y)

        ret = {'accuracy': float(result['accuracy']),
               'loss': float(result['loss']),
               'model_id': model_id,
               'param': json.dumps(param)}

        return ret

    def run_training(self):

        """Run the Beam

        """

        self.p.run()


if __name__ == "__main__":

    test = MultiModelTrain()
    params = all_parameters()
    test.set_parameters(params)
    test.run_training()
