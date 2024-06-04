# Generated by the gRPC Python protocol compiler plugin. DO NOT EDIT!
"""Client and server classes corresponding to protobuf-defined services."""
import grpc

import task_pb2 as task__pb2


class ModelServiceStub(object):
    """Missing associated documentation comment in .proto file."""

    def __init__(self, channel):
        """Constructor.

        Args:
            channel: A grpc.Channel.
        """
        self.ProcessModel = channel.unary_unary(
                '/ml.ModelService/ProcessModel',
                request_serializer=task__pb2.ModelData.SerializeToString,
                response_deserializer=task__pb2.ModelResult.FromString,
                )
        self.RegisterNode = channel.unary_unary(
                '/ml.ModelService/RegisterNode',
                request_serializer=task__pb2.NodeInfo.SerializeToString,
                response_deserializer=task__pb2.Acknowledgment.FromString,
                )
        self.UnregisterNode = channel.unary_unary(
                '/ml.ModelService/UnregisterNode',
                request_serializer=task__pb2.NodeInfo.SerializeToString,
                response_deserializer=task__pb2.Acknowledgment.FromString,
                )
        self.RequestTask = channel.unary_unary(
                '/ml.ModelService/RequestTask',
                request_serializer=task__pb2.NodeInfo.SerializeToString,
                response_deserializer=task__pb2.Task.FromString,
                )
        self.ReportResult = channel.unary_unary(
                '/ml.ModelService/ReportResult',
                request_serializer=task__pb2.TaskResult.SerializeToString,
                response_deserializer=task__pb2.Acknowledgment.FromString,
                )
        self.UpdateNodeStatus = channel.unary_unary(
                '/ml.ModelService/UpdateNodeStatus',
                request_serializer=task__pb2.NodeStatusUpdate.SerializeToString,
                response_deserializer=task__pb2.Acknowledgment.FromString,
                )


class ModelServiceServicer(object):
    """Missing associated documentation comment in .proto file."""

    def ProcessModel(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def RegisterNode(self, request, context):
        """New RPC methods for node management
        """
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def UnregisterNode(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def RequestTask(self, request, context):
        """New RPC method for worker nodes to request tasks
        """
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def ReportResult(self, request, context):
        """New RPC method for worker nodes to report results
        """
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def UpdateNodeStatus(self, request, context):
        """New RPC method for updating node status
        """
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')


def add_ModelServiceServicer_to_server(servicer, server):
    rpc_method_handlers = {
            'ProcessModel': grpc.unary_unary_rpc_method_handler(
                    servicer.ProcessModel,
                    request_deserializer=task__pb2.ModelData.FromString,
                    response_serializer=task__pb2.ModelResult.SerializeToString,
            ),
            'RegisterNode': grpc.unary_unary_rpc_method_handler(
                    servicer.RegisterNode,
                    request_deserializer=task__pb2.NodeInfo.FromString,
                    response_serializer=task__pb2.Acknowledgment.SerializeToString,
            ),
            'UnregisterNode': grpc.unary_unary_rpc_method_handler(
                    servicer.UnregisterNode,
                    request_deserializer=task__pb2.NodeInfo.FromString,
                    response_serializer=task__pb2.Acknowledgment.SerializeToString,
            ),
            'RequestTask': grpc.unary_unary_rpc_method_handler(
                    servicer.RequestTask,
                    request_deserializer=task__pb2.NodeInfo.FromString,
                    response_serializer=task__pb2.Task.SerializeToString,
            ),
            'ReportResult': grpc.unary_unary_rpc_method_handler(
                    servicer.ReportResult,
                    request_deserializer=task__pb2.TaskResult.FromString,
                    response_serializer=task__pb2.Acknowledgment.SerializeToString,
            ),
            'UpdateNodeStatus': grpc.unary_unary_rpc_method_handler(
                    servicer.UpdateNodeStatus,
                    request_deserializer=task__pb2.NodeStatusUpdate.FromString,
                    response_serializer=task__pb2.Acknowledgment.SerializeToString,
            ),
    }
    generic_handler = grpc.method_handlers_generic_handler(
            'ml.ModelService', rpc_method_handlers)
    server.add_generic_rpc_handlers((generic_handler,))


 # This class is part of an EXPERIMENTAL API.
class ModelService(object):
    """Missing associated documentation comment in .proto file."""

    @staticmethod
    def ProcessModel(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/ml.ModelService/ProcessModel',
            task__pb2.ModelData.SerializeToString,
            task__pb2.ModelResult.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def RegisterNode(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/ml.ModelService/RegisterNode',
            task__pb2.NodeInfo.SerializeToString,
            task__pb2.Acknowledgment.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def UnregisterNode(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/ml.ModelService/UnregisterNode',
            task__pb2.NodeInfo.SerializeToString,
            task__pb2.Acknowledgment.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def RequestTask(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/ml.ModelService/RequestTask',
            task__pb2.NodeInfo.SerializeToString,
            task__pb2.Task.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def ReportResult(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/ml.ModelService/ReportResult',
            task__pb2.TaskResult.SerializeToString,
            task__pb2.Acknowledgment.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def UpdateNodeStatus(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/ml.ModelService/UpdateNodeStatus',
            task__pb2.NodeStatusUpdate.SerializeToString,
            task__pb2.Acknowledgment.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)
