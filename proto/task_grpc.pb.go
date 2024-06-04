// Code generated by protoc-gen-go-grpc. DO NOT EDIT.
// versions:
// - protoc-gen-go-grpc v1.3.0
// - protoc             v3.12.4
// source: task.proto

package proto

import (
	context "context"
	grpc "google.golang.org/grpc"
	codes "google.golang.org/grpc/codes"
	status "google.golang.org/grpc/status"
)

// This is a compile-time assertion to ensure that this generated file
// is compatible with the grpc package it is being compiled against.
// Requires gRPC-Go v1.32.0 or later.
const _ = grpc.SupportPackageIsVersion7

const (
	ModelService_ProcessModel_FullMethodName     = "/ml.ModelService/ProcessModel"
	ModelService_RegisterNode_FullMethodName     = "/ml.ModelService/RegisterNode"
	ModelService_UnregisterNode_FullMethodName   = "/ml.ModelService/UnregisterNode"
	ModelService_RequestTask_FullMethodName      = "/ml.ModelService/RequestTask"
	ModelService_ReportResult_FullMethodName     = "/ml.ModelService/ReportResult"
	ModelService_UpdateNodeStatus_FullMethodName = "/ml.ModelService/UpdateNodeStatus"
)

// ModelServiceClient is the client API for ModelService service.
//
// For semantics around ctx use and closing/ending streaming RPCs, please refer to https://pkg.go.dev/google.golang.org/grpc/?tab=doc#ClientConn.NewStream.
type ModelServiceClient interface {
	ProcessModel(ctx context.Context, in *ModelData, opts ...grpc.CallOption) (*ModelResult, error)
	// New RPC methods for node management
	RegisterNode(ctx context.Context, in *NodeInfo, opts ...grpc.CallOption) (*Acknowledgment, error)
	UnregisterNode(ctx context.Context, in *NodeInfo, opts ...grpc.CallOption) (*Acknowledgment, error)
	// New RPC method for worker nodes to request tasks
	RequestTask(ctx context.Context, in *NodeInfo, opts ...grpc.CallOption) (*Task, error)
	// New RPC method for worker nodes to report results
	ReportResult(ctx context.Context, in *TaskResult, opts ...grpc.CallOption) (*Acknowledgment, error)
	// New RPC method for updating node status
	UpdateNodeStatus(ctx context.Context, in *NodeStatusUpdate, opts ...grpc.CallOption) (*Acknowledgment, error)
}

type modelServiceClient struct {
	cc grpc.ClientConnInterface
}

func NewModelServiceClient(cc grpc.ClientConnInterface) ModelServiceClient {
	return &modelServiceClient{cc}
}

func (c *modelServiceClient) ProcessModel(ctx context.Context, in *ModelData, opts ...grpc.CallOption) (*ModelResult, error) {
	out := new(ModelResult)
	err := c.cc.Invoke(ctx, ModelService_ProcessModel_FullMethodName, in, out, opts...)
	if err != nil {
		return nil, err
	}
	return out, nil
}

func (c *modelServiceClient) RegisterNode(ctx context.Context, in *NodeInfo, opts ...grpc.CallOption) (*Acknowledgment, error) {
	out := new(Acknowledgment)
	err := c.cc.Invoke(ctx, ModelService_RegisterNode_FullMethodName, in, out, opts...)
	if err != nil {
		return nil, err
	}
	return out, nil
}

func (c *modelServiceClient) UnregisterNode(ctx context.Context, in *NodeInfo, opts ...grpc.CallOption) (*Acknowledgment, error) {
	out := new(Acknowledgment)
	err := c.cc.Invoke(ctx, ModelService_UnregisterNode_FullMethodName, in, out, opts...)
	if err != nil {
		return nil, err
	}
	return out, nil
}

func (c *modelServiceClient) RequestTask(ctx context.Context, in *NodeInfo, opts ...grpc.CallOption) (*Task, error) {
	out := new(Task)
	err := c.cc.Invoke(ctx, ModelService_RequestTask_FullMethodName, in, out, opts...)
	if err != nil {
		return nil, err
	}
	return out, nil
}

func (c *modelServiceClient) ReportResult(ctx context.Context, in *TaskResult, opts ...grpc.CallOption) (*Acknowledgment, error) {
	out := new(Acknowledgment)
	err := c.cc.Invoke(ctx, ModelService_ReportResult_FullMethodName, in, out, opts...)
	if err != nil {
		return nil, err
	}
	return out, nil
}

func (c *modelServiceClient) UpdateNodeStatus(ctx context.Context, in *NodeStatusUpdate, opts ...grpc.CallOption) (*Acknowledgment, error) {
	out := new(Acknowledgment)
	err := c.cc.Invoke(ctx, ModelService_UpdateNodeStatus_FullMethodName, in, out, opts...)
	if err != nil {
		return nil, err
	}
	return out, nil
}

// ModelServiceServer is the server API for ModelService service.
// All implementations must embed UnimplementedModelServiceServer
// for forward compatibility
type ModelServiceServer interface {
	ProcessModel(context.Context, *ModelData) (*ModelResult, error)
	// New RPC methods for node management
	RegisterNode(context.Context, *NodeInfo) (*Acknowledgment, error)
	UnregisterNode(context.Context, *NodeInfo) (*Acknowledgment, error)
	// New RPC method for worker nodes to request tasks
	RequestTask(context.Context, *NodeInfo) (*Task, error)
	// New RPC method for worker nodes to report results
	ReportResult(context.Context, *TaskResult) (*Acknowledgment, error)
	// New RPC method for updating node status
	UpdateNodeStatus(context.Context, *NodeStatusUpdate) (*Acknowledgment, error)
	mustEmbedUnimplementedModelServiceServer()
}

// UnimplementedModelServiceServer must be embedded to have forward compatible implementations.
type UnimplementedModelServiceServer struct {
}

func (UnimplementedModelServiceServer) ProcessModel(context.Context, *ModelData) (*ModelResult, error) {
	return nil, status.Errorf(codes.Unimplemented, "method ProcessModel not implemented")
}
func (UnimplementedModelServiceServer) RegisterNode(context.Context, *NodeInfo) (*Acknowledgment, error) {
	return nil, status.Errorf(codes.Unimplemented, "method RegisterNode not implemented")
}
func (UnimplementedModelServiceServer) UnregisterNode(context.Context, *NodeInfo) (*Acknowledgment, error) {
	return nil, status.Errorf(codes.Unimplemented, "method UnregisterNode not implemented")
}
func (UnimplementedModelServiceServer) RequestTask(context.Context, *NodeInfo) (*Task, error) {
	return nil, status.Errorf(codes.Unimplemented, "method RequestTask not implemented")
}
func (UnimplementedModelServiceServer) ReportResult(context.Context, *TaskResult) (*Acknowledgment, error) {
	return nil, status.Errorf(codes.Unimplemented, "method ReportResult not implemented")
}
func (UnimplementedModelServiceServer) UpdateNodeStatus(context.Context, *NodeStatusUpdate) (*Acknowledgment, error) {
	return nil, status.Errorf(codes.Unimplemented, "method UpdateNodeStatus not implemented")
}
func (UnimplementedModelServiceServer) mustEmbedUnimplementedModelServiceServer() {}

// UnsafeModelServiceServer may be embedded to opt out of forward compatibility for this service.
// Use of this interface is not recommended, as added methods to ModelServiceServer will
// result in compilation errors.
type UnsafeModelServiceServer interface {
	mustEmbedUnimplementedModelServiceServer()
}

func RegisterModelServiceServer(s grpc.ServiceRegistrar, srv ModelServiceServer) {
	s.RegisterService(&ModelService_ServiceDesc, srv)
}

func _ModelService_ProcessModel_Handler(srv interface{}, ctx context.Context, dec func(interface{}) error, interceptor grpc.UnaryServerInterceptor) (interface{}, error) {
	in := new(ModelData)
	if err := dec(in); err != nil {
		return nil, err
	}
	if interceptor == nil {
		return srv.(ModelServiceServer).ProcessModel(ctx, in)
	}
	info := &grpc.UnaryServerInfo{
		Server:     srv,
		FullMethod: ModelService_ProcessModel_FullMethodName,
	}
	handler := func(ctx context.Context, req interface{}) (interface{}, error) {
		return srv.(ModelServiceServer).ProcessModel(ctx, req.(*ModelData))
	}
	return interceptor(ctx, in, info, handler)
}

func _ModelService_RegisterNode_Handler(srv interface{}, ctx context.Context, dec func(interface{}) error, interceptor grpc.UnaryServerInterceptor) (interface{}, error) {
	in := new(NodeInfo)
	if err := dec(in); err != nil {
		return nil, err
	}
	if interceptor == nil {
		return srv.(ModelServiceServer).RegisterNode(ctx, in)
	}
	info := &grpc.UnaryServerInfo{
		Server:     srv,
		FullMethod: ModelService_RegisterNode_FullMethodName,
	}
	handler := func(ctx context.Context, req interface{}) (interface{}, error) {
		return srv.(ModelServiceServer).RegisterNode(ctx, req.(*NodeInfo))
	}
	return interceptor(ctx, in, info, handler)
}

func _ModelService_UnregisterNode_Handler(srv interface{}, ctx context.Context, dec func(interface{}) error, interceptor grpc.UnaryServerInterceptor) (interface{}, error) {
	in := new(NodeInfo)
	if err := dec(in); err != nil {
		return nil, err
	}
	if interceptor == nil {
		return srv.(ModelServiceServer).UnregisterNode(ctx, in)
	}
	info := &grpc.UnaryServerInfo{
		Server:     srv,
		FullMethod: ModelService_UnregisterNode_FullMethodName,
	}
	handler := func(ctx context.Context, req interface{}) (interface{}, error) {
		return srv.(ModelServiceServer).UnregisterNode(ctx, req.(*NodeInfo))
	}
	return interceptor(ctx, in, info, handler)
}

func _ModelService_RequestTask_Handler(srv interface{}, ctx context.Context, dec func(interface{}) error, interceptor grpc.UnaryServerInterceptor) (interface{}, error) {
	in := new(NodeInfo)
	if err := dec(in); err != nil {
		return nil, err
	}
	if interceptor == nil {
		return srv.(ModelServiceServer).RequestTask(ctx, in)
	}
	info := &grpc.UnaryServerInfo{
		Server:     srv,
		FullMethod: ModelService_RequestTask_FullMethodName,
	}
	handler := func(ctx context.Context, req interface{}) (interface{}, error) {
		return srv.(ModelServiceServer).RequestTask(ctx, req.(*NodeInfo))
	}
	return interceptor(ctx, in, info, handler)
}

func _ModelService_ReportResult_Handler(srv interface{}, ctx context.Context, dec func(interface{}) error, interceptor grpc.UnaryServerInterceptor) (interface{}, error) {
	in := new(TaskResult)
	if err := dec(in); err != nil {
		return nil, err
	}
	if interceptor == nil {
		return srv.(ModelServiceServer).ReportResult(ctx, in)
	}
	info := &grpc.UnaryServerInfo{
		Server:     srv,
		FullMethod: ModelService_ReportResult_FullMethodName,
	}
	handler := func(ctx context.Context, req interface{}) (interface{}, error) {
		return srv.(ModelServiceServer).ReportResult(ctx, req.(*TaskResult))
	}
	return interceptor(ctx, in, info, handler)
}

func _ModelService_UpdateNodeStatus_Handler(srv interface{}, ctx context.Context, dec func(interface{}) error, interceptor grpc.UnaryServerInterceptor) (interface{}, error) {
	in := new(NodeStatusUpdate)
	if err := dec(in); err != nil {
		return nil, err
	}
	if interceptor == nil {
		return srv.(ModelServiceServer).UpdateNodeStatus(ctx, in)
	}
	info := &grpc.UnaryServerInfo{
		Server:     srv,
		FullMethod: ModelService_UpdateNodeStatus_FullMethodName,
	}
	handler := func(ctx context.Context, req interface{}) (interface{}, error) {
		return srv.(ModelServiceServer).UpdateNodeStatus(ctx, req.(*NodeStatusUpdate))
	}
	return interceptor(ctx, in, info, handler)
}

// ModelService_ServiceDesc is the grpc.ServiceDesc for ModelService service.
// It's only intended for direct use with grpc.RegisterService,
// and not to be introspected or modified (even as a copy)
var ModelService_ServiceDesc = grpc.ServiceDesc{
	ServiceName: "ml.ModelService",
	HandlerType: (*ModelServiceServer)(nil),
	Methods: []grpc.MethodDesc{
		{
			MethodName: "ProcessModel",
			Handler:    _ModelService_ProcessModel_Handler,
		},
		{
			MethodName: "RegisterNode",
			Handler:    _ModelService_RegisterNode_Handler,
		},
		{
			MethodName: "UnregisterNode",
			Handler:    _ModelService_UnregisterNode_Handler,
		},
		{
			MethodName: "RequestTask",
			Handler:    _ModelService_RequestTask_Handler,
		},
		{
			MethodName: "ReportResult",
			Handler:    _ModelService_ReportResult_Handler,
		},
		{
			MethodName: "UpdateNodeStatus",
			Handler:    _ModelService_UpdateNodeStatus_Handler,
		},
	},
	Streams:  []grpc.StreamDesc{},
	Metadata: "task.proto",
}