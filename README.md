# Lightweight-Semantic-segmentation

사용한 데이터 : ICG Drone dataset
Class: 20

Data Location = './data/icg_drone/dataset/original'  ( 원본데이터 )
                './data/icg_drone/dataset/label'     ( 라벨데이터 )


## Image Convert 

 !python tools/Imageconvert.py 
 
 re_size 인자를 수정해서 이미지 output 수정 가능
 
 
 ## Image file index
 
 !python data/example_dataset/cityscapes/image_file_index/make_image_file_index.py
 
 
 
 ## TFrecording
 
 !python tools/cityscapes/make_cityscapes_tfrecords.py
 
 
 ## Training
 
 './config/cityscapes/cityscapes_bisenetv2.yaml' 파일에서 Parameter 수정 가능
 
 !CUDA_VISIBLE_DEVICES="0, 1, 2, 3" python train_bisenetv2_cityscapes.py
 
 
 
