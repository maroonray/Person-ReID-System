from __future__ import print_function
import threading
import multi_thread as mt


if __name__ == '__main__':
    run_mode = 'multi-thread'
    rtsp_path = "rtsp://admin:admin123@10.1.83.133/cam/realmonitor?channel=7&subtype=0"

    model_dir = '/home/yul-j/Desktop/Demos/MixedModel/models'
    openpose_model = "/home/yul-j/Desktop/Demos/OpenPose/models/"
    embeddings_dir = '/home/yul-j/Desktop/Demos/MixedModel/embeddings'

    if run_mode == 'multi-thread':
        t1 = threading.Thread(target=mt.get_latest_frame, args=(rtsp_path, 0.5,))
        t2 = threading.Thread(target=mt.process_latest_frame,
                              args=(model_dir, openpose_model, embeddings_dir, ))
        t3 = threading.Thread(target=mt.subscribe_json, args=('tcp://10.1.83.61:17719', ))
        t4 = threading.Thread(target=mt.send_json, args=())
        t4.start()
        t3.start()
        t2.start()
        t1.start()

        t3.join()
        t1.join()
        t2.join()
        t4.join()
        