import grpc
import task_pb2 as pb2
import task_pb2_grpc as pb2_grpc
import pickle
import tensorflow as tf
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import numpy as np

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
def load_data():
    data = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.2, random_state=42)
    encoder = OneHotEncoder(sparse=False)
    y_train = encoder.fit_transform(y_train[:, np.newaxis])
    y_test = encoder.transform(y_test[:, np.newaxis])
    return X_train, X_test, y_train, y_test

def build_model():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(10, activation='relu', input_shape=(4,)),
        tf.keras.layers.Dense(10, activation='relu'),
        tf.keras.layers.Dense(3, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def tensorflow_train_model():
    X_train, X_test, y_train, y_test = load_data()
    model = build_model()
    model.fit(X_train, y_train, epochs=50, verbose=0)
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test Accuracy: {accuracy * 100:.2f}%")
    return model

## ------------- Client Class for ML Service Interaction ------------ ##

class MLClient(object):
    def __init__(self, host='localhost', port='50051'):
        """
        Initialize the ML client with a gRPC channel.
        """
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
            # Assuming task data is suitable for TensorFlow processing
            model = tensorflow_train_model()
            serialized_model = serialize_model(model)
            return serialized_model
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