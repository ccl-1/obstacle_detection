# import os
# def extract_first_number_form_line(line):
#     for word in line.split():
#         if word.isdigit():
#             return word
#     return None
# def process_txt_files(input_folder,output_file):
#     with open(output_file, 'w') as outfile:
#         for filename in os.listdir(input_folder):
#             file_path = os.path.join(input_folder, filename)
#             if filename.endswith('.txt'):
#                 with open(file_path,'r') as infile:
#                     first_line = infile.readline().strip()
#                     first_number = extract_first_number_form_line(first_line)
#                     if first_number:
#                         outfile.write(f"{filename} {first_number}\n")
# input_folder= '/media/ubuntu/zoro/ubuntu/data/railway_obstacle_detection/ObstacleDetection/images/val1'
# output_file = '/media/ubuntu/zoro/ubuntu/data/railway_obstacle_detection/ObstacleDetection/images/val1.txt'
# process_txt_files(input_folder, output_file)



import os
import cv2
import matplotlib.pyplot as plt

class_dict = {0:'Safe', 1:'Low', 2:'High'} 

mode = 'val'
data_path = "/media/ubuntu/zoro/ubuntu/data/railway_obstacle_detection/ObstacleDetection/images/"
train_gt = os.path.join('tmp', mode+'.txt')
with open(train_gt, 'r')as f:
    for line in f:
        file_name, cls = line.strip().split(' ')
        img_path = os.path.join(data_path, mode, file_name+'.jpg')
        img = cv2.imread(img_path)
        print(img_path.replace('jpg', 'txt'))
        if int(cls) == 1:
            info = "Risk Level: " + class_dict[int(cls)]
            font, font_scale, font_thickness, xx, yy = cv2.FONT_HERSHEY_SIMPLEX, 1, 2, 20, 40
            (text_w, text_h), _ = cv2.getTextSize(info, font, font_scale, font_thickness)
            plt.figure(figsize=(10,10))
            cv2.rectangle(img, (xx-5, yy-text_h-5), (xx+text_w, yy+15), (0, 0, 255), -1)
            cv2.putText(img, info, (xx, yy), font, font_scale, (255, 255, 255), font_thickness)
            plt.imshow(img)
            plt.show()