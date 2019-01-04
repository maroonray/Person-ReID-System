import os
import tensorflow as tf


def load_gait_model(model_dir):
    graph_name = [name for name in os.listdir(model_dir) if 'meta' in name][0]
    model_id = graph_name.split('.')[0]
    graph_path = os.path.join(model_dir, graph_name)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    saver = tf.train.import_meta_graph(graph_path)
    model_save_path = os.path.join(model_dir, model_id)
    saver.restore(sess, model_save_path)
    graph = tf.get_default_graph()
    embeds = graph.get_tensor_by_name('RNN/map/TensorArrayStack/TensorArrayGatherV3:0')
    return sess, embeds
