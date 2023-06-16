import ml_metadata as mlmd
from ml_metadata.metadata_store import metadata_store
from ml_metadata.proto import metadata_store_pb2

MODEL_DIR = './server/saved_model'
MODEL_VER = 1

connection_config = metadata_store_pb2.ConnectionConfig()
connection_config.mysql.host = '127.0.0.1'
connection_config.mysql.port = 3306
connection_config.mysql.database = 'birdsClassifier'
connection_config.mysql.user = 'root'
connection_config.mysql.password = 'ROOT'
store = metadata_store.MetadataStore(connection_config)

model_type = metadata_store_pb2.ArtifactType()
model_type.name = "SavedModel"
model_type.properties["version"] = metadata_store_pb2.INT
model_type.properties["name"] = metadata_store_pb2.STRING
model_type_id = store.put_artifact_type(model_type)

model_artifact = metadata_store_pb2.Artifact()
model_artifact.uri = '{}/00{}/saved_model.pb'.format(MODEL_DIR, MODEL_VER)
model_artifact.properties["version"].int_value = MODEL_VER
model_artifact.properties["name"].string_value = 'birdsClassifier'
model_artifact.type_id = model_type_id
[model_artifact_id] = store.put_artifacts([model_artifact])