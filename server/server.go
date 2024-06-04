package main

import (
	"context"
	"errors"
	"log"
	"net"
	"os"

	pb "GO_GPU_DEMO/proto"

	"GO_GPU_DEMO/registry"

	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
)

// server struct implements the TaskServiceServer interface generated from the protobuf definition.
type server struct {
	pb.UnimplementedModelServiceServer
	registry *registry.NodeRegistry
}

// task queues for the workers to process
var task1 = &pb.Task{TaskId: "1", ModelId: "model", Type: "tensorflow", Parameters: []byte("task param here"), ModelData: []byte("model data here")}
var taskQueue = []*pb.Task{task1}

// --------------------- NODE REGISTRATION --------------------- ///

// RegisterNode adds a new node to the registry.
func (s *server) RegisterNode(ctx context.Context, info *pb.NodeInfo) (*pb.Acknowledgment, error) {
	// register the node with the registry
	s.registry.RegisterNode(info.NodeId)

	log.Printf("Node registered: %s", info.NodeId)

	return &pb.Acknowledgment{Success: true}, nil
}

// UnregisterNode removes a node from the registry.
func (s *server) UnregisterNode(ctx context.Context, info *pb.NodeInfo) (*pb.Acknowledgment, error) {
	// unregiter method
	s.registry.UnregisterNode(info.NodeId)

	log.Printf("Node unregistered: %s", info.NodeId)
	return &pb.Acknowledgment{
		Success: true,
		Message: "Node unregistered successfully",
	}, nil
}

// --------------------- NODE STATUS --------------------- ///
// UpdateNodeStatus is an RPC method that receives status updates from nodes.
// This is a new function based on the newly defined protobuf message NodeStatusUpdate.
func (s *server) UpdateNodeStatus(ctx context.Context, req *pb.NodeStatusUpdate) (*pb.Acknowledgment, error) {
	load := int(req.Load)                       // Convert int32 to int as Go uses int in the NodeRegistry.
	s.registry.UpdateNodeLoad(req.NodeId, load) // Update node load and check-in time.

	log.Printf("Updated status for node %s: load %d", req.NodeId, load)
	return &pb.Acknowledgment{Success: true, Message: "Node status updated successfully"}, nil
}

// Modified task allocation logic considering node load
func (s *server) AllocateNode() string {
	return s.registry.AllocateNode()
}

// --------------------- PROCESS MODEL --------------------- ///

func (s *server) ProcessModel(ctx context.Context, req *pb.ModelData) (*pb.ModelResult, error) {
	// allocate a node to process the task
	nodeID := s.registry.AllocateNode()
	if nodeID == "" {
		return nil, status.Errorf(codes.ResourceExhausted, "No available nodes to process the model")
	}

	log.Printf("Model assigned to node %s", nodeID)

	// Simulate processing and return a result
	resultData := req.Parameters // Placeholder for actual processing logic
	return &pb.ModelResult{ModelId: req.ModelId, Results: resultData}, nil

}

/// --------------------- TASK REQUESTING --------------------- ///

// RequestTask requests a task from the server.
func (s *server) RequestTask(ctx context.Context, info *pb.NodeInfo) (*pb.Task, error) {
	task, err := s.getTaskFromQueue()
	if err != nil {
		return nil, status.Errorf(codes.NotFound, "No tasks available")
	}
	return task, nil
}

// --------------------- GET TASK FROM QUEUE --------------------- ///

func (s *server) getTaskFromQueue() (*pb.Task, error) {
	// Implementation of your logic to get a task from the queue
	// Placeholder:
	if len(taskQueue) == 0 {
		return nil, errors.New("no tasks available")
	}
	task := taskQueue[0]      // Get the first task
	taskQueue = taskQueue[1:] // Remove the task from the queue
	return task, nil
}

/// --------------------- TASK REPORTING --------------------- ///

// ReportResult is called by worker nodes to report the result of a task.
func (s *server) ReportResult(ctx context.Context, result *pb.TaskResult) (*pb.Acknowledgment, error) {
	// Logic to handle the reported result.
	// For now, just log the result.
	log.Printf("Received result for task: %s", result.TaskId)
	log.Printf("Result: %s", result.Result)
	return &pb.Acknowledgment{Success: true, Message: "Result received successfully"}, nil
}

// / --------------------- SAVE OUT --------------------- ///
func saveResult(result *pb.TaskResult) {
	// Simple example to save the result data to a file
	fileName := "result_" + result.TaskId + ".model"
	file, err := os.Create(fileName)
	if err != nil {
		log.Fatalf("Failed to create file: %v", err)
		return
	}
	defer file.Close()

	_, err = file.Write(result.Result)
	if err != nil {
		log.Fatalf("Failed to write to file: %v", err)
	}
}

/// --------------------- MAIN FUNCTION --------------------- ///

func main() {
	// initialising instance of registry
	var reg = registry.NewRegistry()

	// Attempt to listen on TCP port 50051. This is the network port that your server will open on the host machine
	// to listen for incoming connections.
	lis, err := net.Listen("tcp", ":50051")
	if err != nil {
		// If there is an error in opening the port (e.g., the port is already in use), log the error and exit.
		// The `Fatalf` function of the `log` package logs the error message and then calls `os.Exit(1)`.
		log.Fatalf("Failed to listen: %v", err)
	}

	// Create a new gRPC server. This server will handle requests as per the gRPC protocols.
	s := grpc.NewServer()

	// Register the server with the TaskService gRPC service server. This is generated from the protobuf definition.
	// It tells the gRPC library that our server (created above) will be handling requests defined in the TaskService.
	pb.RegisterModelServiceServer(s, &server{registry: reg})

	// Log a message indicating that the server is now listening for requests. It logs the address on which the server is listening.
	log.Printf("Server listening at %v", lis.Addr())

	// Start serving incoming connections from clients. The `Serve` method will block indefinitely unless an error occurs,
	// meaning it will continually listen for and handle new incoming connections.
	if err := s.Serve(lis); err != nil {
		// If there is an error while serving, log the error and exit.
		log.Fatalf("Failed to serve: %v", err)
	}
}
