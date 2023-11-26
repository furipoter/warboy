from furiosa.runtime.sync import create_runner
from utils.preprocess import *
from utils.postprocess import *
import time

# image_path = './data/22_Picnic_Picnic_22_10.jpg'
# video_path = './demo/input_video/home_video_1.mp4'



video_path = './furi_test.mp4'


with create_runner("yolov7_i8_1pe.enf", worker_num=2, device='warboy(1)*1') as runner:
    # image = cv2.imread(image_path)


    start = time.time()
    for i in range(1000):
        frame = torch.randint(256, (1, 3, 640, 640), dtype=torch.uint8)
        # image_tensor, preproc_params = preproc(frame)
        output = runner.run(frame)
        # predictions = postproc(output, 0.65, 0.35)
        # predictions = predictions[0]
        # bboxed_img = draw_bbox(frame, predictions, preproc_params)
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
