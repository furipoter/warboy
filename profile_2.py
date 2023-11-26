from furiosa.runtime.sync import create_runner
from utils.preprocess import *
from utils.postprocess import *
import time

from furiosa.runtime.profiler import profile

# image_path = './data/22_Picnic_Picnic_22_10.jpg'
video_path = './demo/input_video/home_video_1.mp4'
video_path = './furi_test.mp4'



with open("yolov7_i8_2pe_trace.json", "w") as output:
    with profile(file=output) as profiler:
        with create_runner("yolov7_i8_2pe.enf", worker_num=8) as runner:
            # image = cv2.imread(image_path)
            cap = cv2.VideoCapture(video_path)

            start = time.time()
            i = 0
            for j in range(10):
                print(j)
                while True:
                    hasFrame, frame = cap.read()
                    if not hasFrame:
                        break
                    image_tensor, preproc_params = preproc(frame)
                    output = runner.run(image_tensor)
                    predictions = postproc(output, 0.65, 0.35)
                    predictions = predictions[0]
                    bboxed_img = draw_bbox(frame, predictions, preproc_params)
                    cv2.imwrite(f'./output/output_{i}.png', bboxed_img)
                    i += 1
            print(time.time() - start)

    # start = time.time()
    # for i in range(30):
    #     image_tensor, preproc_params = preproc(image)
    #     output = await asyncio.gather(runner.run(image_tensor))
    #     predictions = postproc(output, 0.65, 0.35)
    #     predictions = predictions[0]
    #     bboxed_img = draw_bbox(image, predictions, preproc_params)
    #     cv2.imwrite('./output.png', bboxed_img)
    # print(str(time.time() - start))
