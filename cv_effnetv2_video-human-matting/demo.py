

# 视频人像抠图模型-通用领域
#pip install moviepy

from modelscope.outputs import OutputKeys
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

video_matting = pipeline(Tasks.video_human_matting, model='damo/cv_effnetv2_video-human-matting')
result_status = video_matting({'video_input_path':'./videos/video_matting_test.mp4','output_path':'matting_out.mp4'})
result = result_status[OutputKeys.MASKS]