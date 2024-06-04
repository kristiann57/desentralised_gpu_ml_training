# Import necessary modules from the gRPC package and the generated protocol buffer code.
import grpc
import task_pb2 as pb2
import task_pb2_grpc as pb2_grpc



# ML imports
import pickle
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier


## ------------- Serialization and Deserializing the task data -------------- #

def serialize_model(model):
    """
    Serialize the trained model to a byte array using joblib.
    """
    return pickle.dumps(model)

def deserialize_model(serialized_model):
    """
    Deserialize the model from byte array.
    """
    return pickle.loads(serialized_model)
## ------------- ML algorithm ------------ ##

def train_model(parameters):
    """
    Train a machine learning model using the Iris dataset.
    Here 'parameters' could be used to customize the model training,
    but for simplicity, we'll ignore them in this example.
    """
    data = load_iris()
    model = RandomForestClassifier(n_estimators=100)
    model.fit(data.data, data.target)
    return model



class MLClient(object):
    def __init__(self, host='localhost', port='50051'):
        # the host and port should match those of the server
        self.channel = grpc.insecure_channel(f'{host}:{port}')
        self.stub = pb2_grpc.ModelServiceStub(self.channel)

    def process_model(self, model_id, parameters):
        # this method should be called with the model data
        model_data = pb2.ModelData(model_id=model_id, parameters=parameters)
        return self.stub.ProcessModel(model_data)

    def register_node(self, node_id):
        # Call the RegisterNode RPC method
        node_info = pb2.NodeInfo(node_id=node_id)
        return self.stub.RegisterNode(node_info)

    def unregister_node(self, node_id):
        # call the unregister RCP method
        node_info = pb2.NodeInfo(node_id=node_id)
        return self.stub.UnregisterNode(node_info)

    def request_task(self, node_id):
        # call the request task RPC method
        node_info = pb2.NodeInfo(node_id=node_id)
        return self.stub.RequestTask(node_info)

    def report_result(self, task_id, result_data):
        # Call the ReportResult RPC method to report the result of a task
        task_result = pb2.TaskResult(task_id=task_id, result=result_data)
        return self.stub.ReportResult(task_result)

    @staticmethod
    def process_task(task):
        # process internally the ML learnign task
        model = train_model(task.parameters)
        serialized_model = serialize_model(model)
        return serialized_model


    def close(self):
        self.channel.close()

def run():
    # Establish a channel to a server. The 'insecure_channel' method creates a channel without encryption,
    # which is fine for development but not for production. Here, it connects to the server running on
    # 'localhost' at port '50051'.
    client = MLClient()
    model_id = 'unique_model_id'  # replace with your actual model id
    parameters = b'your_model_parameters'  # this should be the serialized model parameters

    try:
        # create a random ID:
        node_id = "node_test_papa"
        register_response = client.register_node(node_id)

        print("Register Response: ", register_response)
        response = client.process_model(model_id, parameters)
        print(f"Server responded with results: {response.results}")

        # Request a task from the server
        task = client.request_task(node_id)

        # debugging --------
        print(task)
        # Process the task
        # ... (process the task using machine learning libraries, etc.)
        result_data = client.process_task(task)

        # Report the result back to the server
        results_acknowledgement = client.report_result(task.task_id, result_data)

        print(results_acknowledgement)
    finally:
        client.close()


# This block ensures that the run() function is called when the script is executed directly,
# but not when it is imported as a module in another script.
if __name__ == '__main__':
    run()





# # Function to register with server's registry
# def register_with_server(stub):
#     # Generate a unique identifier for this client
#     client_id = str(uuid.uuid4())

#     response = stub.RegisterClient(task_pb2.ClientInfo(id=client_id))

#     if response.success:
#         print(f"Registered with the server: {response.message}")
#     else:
#         print(f"Failed to register with the server: {response.message}")

#     return client_id

# # Unregister with the reserver's registry
# def unregister_with_server(stub, client_id):
#     response = stub.UnregisterClient(task_pb2.ClientInfo(id=client_id))
#     if response.success:
#         print(f"Unregistered from the server: {response.message}")
#     else:
#         print(f"Failed to unregister from the server: {response.message}")

