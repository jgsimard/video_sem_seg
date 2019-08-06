#RESUME_S=./run/isi_rgb/spatial-xception-adv/experiment_0/checkpoint.pth.tar
#python demo_video.py --resume  $RESUME_S\
#                     --backbone xception \
#                     --dataset isi_rgb \
#                     --demo_camera
#
#RESUME_T=./run/isi_rgb_temporal/deeplab-xception-adv/experiment_0/checkpoint.pth.tar
#python demo_video.py --resume  $RESUME_T\
#                     --backbone xception \
#                     --dataset isi_rgb \
#                     --demo_camera  \
#                     --demo_temporal \
#                     --demo_frame_fixed_schedule 10




RESUME_S=./run/isi_rgb/spatial-xception-adv/experiment_0/checkpoint.pth.tar
python demo_video.py --resume  $RESUME_S\
                     --backbone xception \
                     --dataset isi_rgb \
                     --demo_video_path  /home/deepsight2/DeepSightData/Feb_11_CDE_lab/GoPro_RGB_data/Room_camera_1/GH020005.MP4


#RESUME_T=./run/isi_rgb_temporal/deeplab-xception-adv/experiment_0/checkpoint.pth.tar
#python demo_video.py --resume  $RESUME_T\
#                     --backbone xception \
#                     --dataset isi_rgb \
#                     --demo_video_path  /home/deepsight2/DeepSightData/Feb_11_CDE_lab/GoPro_RGB_data/Room_camera_1/GH020005.MP4\
#                     --demo_temporal \
#                     --demo_frame_fixed_schedule 10
