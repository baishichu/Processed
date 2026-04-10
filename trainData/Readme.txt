The SMIC_crop folder contains the structure:
- subject id
  - video_name
    + image

The SMIC_HS_E_spotting.xlsx is the annotation file:
- subject: the subject id
- video: the index of video in one subject
- video_name: the video name corresponded to one folder in 
- NumME: the number of Micro-expression in one video
- StartFrame: the start frame in one video
- Endframe: the end frame in one video
- ME_Type: The type of Micro-expression: sur is surprise, po is positive emotion, ne is negative emotion
- onset: the start frame of Micro-expression 
- offset: the end frame of Micro-expression
