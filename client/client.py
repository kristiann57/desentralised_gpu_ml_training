import grpc
import task_pb2 as pb2
import task_pb2_grpc as pb2_grpc
import pickle

from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier

import subprocess
import sys
import os

## ------------- Serialization and Deserializing the task data -------------- ##

def serialize_model(model):
    """
    Serialize the trained model to a byte array using pickle.
    """
    return pickle.dumps(model)

def deserialize_model(serialized_model):
    """
    Deserialize the model from byte array using pickle.
    """
    return pickle.loads(serialized_model)

## ------------- Machine Learning Training Function ------------ ##

def train_model(data, target):
    """
    Train a machine learning model using the Iris dataset.
    """
    model = RandomForestClassifier(n_estimators=100)
    model.fit(data, target)
    return model

## -------------- Additional TensorFlow-specific functions -------------- ##


def run_tensorflow_task():
    """ Run the TensorFlow task and return the path to the saved model. """
    result = subprocess.run([sys.executable, 'tensorflow_task.py'], capture_output=True, text=True)
    if result.stderr:
        print("Error:", result.stderr)
    return result.stdout.strip()  # This now contains the path to the saved model

## ------------- Client Class for ML Service Interaction ------------ ##

class MLClient(object):
    def __init__(self, host='localhost', port='50051'):
        self.channel = grpc.insecure_channel(f'{host}:{port}')
        self.stub = pb2_grpc.ModelServiceStub(self.channel)

    def register_node(self, node_id):
        """
        Register this node with the server.
        """
        node_info = pb2.NodeInfo(node_id=node_id)
        return self.stub.RegisterNode(node_info)

    def unregister_node(self, node_id):
        """
        Unregister this node from the server.
        """
        node_info = pb2.NodeInfo(node_id=node_id)
        return self.stub.UnregisterNode(node_info)

    def request_task(self, node_id):
        """
        Request a new task from the server for this node.
        """
        node_info = pb2.NodeInfo(node_id=node_id)
        return self.stub.RequestTask(node_info)

    def report_result(self, task_id, result_data):
        """
        Report the result of a task back to the server.
        """
        task_result = pb2.TaskResult(task_id=task_id, result=result_data)
        return self.stub.ReportResult(task_result)

    def process_task(self, node_id, task):
        """Process a received task based on its type."""
        if task.type == 'tensorflow':
            # Run TensorFlow task and get the model zip path
            model_zip_path = run_tensorflow_task()
            
            try:
                # Read the zip file content
                with open(model_zip_path, 'rb') as file:
                    model_data = file.read()
                return model_data
            except IOError as e:
                print(f"Failed to read model data from {model_zip_path}: {str(e)}")
                return None
        else:
            # Handle other types of tasks
            iris_data = load_iris()
            model = train_model(iris_data.data, iris_data.target)
            serialized_model = serialize_model(model)
            return serialized_model

    def close(self):
        """
        Close the gRPC channel when done.
        """
        self.channel.close()

def run():
    client = MLClient()
    node_id = "node_test_123"

    try:
        client.register_node(node_id)
        task_response = client.request_task(node_id)
        print("Received task:", task_response)
        
        result_data = client.process_task(node_id, task_response)
        
        print("Reporting result...")
        result_response = client.report_result(task_response.task_id, result_data)
        print("Result reported:", result_response)

    finally:
        client.unregister_node(node_id)
        client.close()

if __name__ == '__main__':
    run()