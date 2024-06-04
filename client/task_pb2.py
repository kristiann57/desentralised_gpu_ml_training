# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: task.proto
# Protobuf Python Version: 4.25.1
"""Generated protocol buffer code."""
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import symbol_database as _symbol_database
from google.protobuf.internal import builder as _builder
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n\ntask.proto\x12\x02ml\"1\n\tModelData\x12\x10\n\x08model_id\x18\x01 \x01(\t\x12\x12\n\nparameters\x18\x02 \x01(\x0c\"0\n\x0bModelResult\x12\x10\n\x08model_id\x18\x01 \x01(\t\x12\x0f\n\x07results\x18\x02 \x01(\x0c\"\x1b\n\x08NodeInfo\x12\x0f\n\x07node_id\x18\x01 \x01(\t\"1\n\x10NodeStatusUpdate\x12\x0f\n\x07node_id\x18\x01 \x01(\t\x12\x0c\n\x04load\x18\x02 \x01(\x05\"2\n\x0e\x41\x63knowledgment\x12\x0f\n\x07success\x18\x01 \x01(\x08\x12\x0f\n\x07message\x18\x02 \x01(\t\"_\n\x04Task\x12\x0f\n\x07task_id\x18\x01 \x01(\t\x12\x10\n\x08model_id\x18\x02 \x01(\t\x12\x0c\n\x04type\x18\x03 \x01(\t\x12\x12\n\nparameters\x18\x04 \x01(\x0c\x12\x12\n\nmodel_data\x18\x05 \x01(\x0c\"-\n\nTaskResult\x12\x0f\n\x07task_id\x18\x01 \x01(\t\x12\x0e\n\x06result\x18\x02 \x01(\x0c\x32\xbd\x02\n\x0cModelService\x12.\n\x0cProcessModel\x12\r.ml.ModelData\x1a\x0f.ml.ModelResult\x12\x30\n\x0cRegisterNode\x12\x0c.ml.NodeInfo\x1a\x12.ml.Acknowledgment\x12\x32\n\x0eUnregisterNode\x12\x0c.ml.NodeInfo\x1a\x12.ml.Acknowledgment\x12%\n\x0bRequestTask\x12\x0c.ml.NodeInfo\x1a\x08.ml.Task\x12\x32\n\x0cReportResult\x12\x0e.ml.TaskResult\x1a\x12.ml.Acknowledgment\x12<\n\x10UpdateNodeStatus\x12\x14.ml.NodeStatusUpdate\x1a\x12.ml.AcknowledgmentB\x13Z\x11GO_GPU_DEMO/protob\x06proto3')

_globals = globals()
_builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, _globals)
_builder.BuildTopDescriptorsAndMessages(DESCRIPTOR, 'task_pb2', _globals)
if _descriptor._USE_C_DESCRIPTORS == False:
  _globals['DESCRIPTOR']._options = None
  _globals['DESCRIPTOR']._serialized_options = b'Z\021GO_GPU_DEMO/proto'
  _globals['_MODELDATA']._serialized_start=18
  _globals['_MODELDATA']._serialized_end=67
  _globals['_MODELRESULT']._serialized_start=69
  _globals['_MODELRESULT']._serialized_end=117
  _globals['_NODEINFO']._serialized_start=119
  _globals['_NODEINFO']._serialized_end=146
  _globals['_NODESTATUSUPDATE']._serialized_start=148
  _globals['_NODESTATUSUPDATE']._serialized_end=197
  _globals['_ACKNOWLEDGMENT']._serialized_start=199
  _globals['_ACKNOWLEDGMENT']._serialized_end=249
  _globals['_TASK']._serialized_start=251
  _globals['_TASK']._serialized_end=346
  _globals['_TASKRESULT']._serialized_start=348
  _globals['_TASKRESULT']._serialized_end=393
  _globals['_MODELSERVICE']._serialized_start=396
  _globals['_MODELSERVICE']._serialized_end=713
# @@protoc_insertion_point(module_scope)