import base64
import json
import os
import os.path as osp

import numpy as np
import PIL.Image
from labelme import utils


if __name__ == '__main__':
    jpgs_path   = "datasets/JPEGImages"
    pngs_path   = "datasets/SegmentationClass"
    classes     = ["_background_","basaltic lava platform","trachyte","tongue"]
    
    count = os.listdir("./datasets/before/") 
    for i in range(0, len(count)):
        path = os.path.join("./datasets/before", count[i])

        if os.path.isfile(path) and path.endswith('json'):
            #data = json.load(open(path))
            with open(path,'r',encoding='utf-8') as f:
                data = json.load(f)
            if data['imageData']:
                imageData = data['imageData']
            else:
                imagePath = os.path.join(os.path.dirname(path), data['imagePath'])
                with open(imagePath, 'rb') as f:
                    imageData = f.read()
                    imageData = base64.b64encode(imageData).decode('utf-8')

            img = utils.img_b64_to_arr(imageData)

            label_name_to_value = {'_background_': 0}    #存储标签映射
            for shape in data['shapes']:

                label_name = shape['label']
                if label_name in label_name_to_value:
                    label_value = label_name_to_value[label_name]
                else:
                    label_value = len(label_name_to_value)
                    label_name_to_value[label_name] = label_value
            
            # label_values must be dense
            label_values, label_names = [], []
            for ln, lv in sorted(label_name_to_value.items(), key=lambda x: x[1]):
                label_values.append(lv)

                label_names.append(ln)

            assert label_values == list(range(len(label_values)))
            
            lbl,ins = utils.shapes_to_label(img.shape, data['shapes'], label_name_to_value)

            PIL.Image.fromarray(img).save(osp.join(jpgs_path, count[i].split(".")[0]+'.jpg'))

            new = np.zeros([np.shape(img)[0],np.shape(img)[1]])
            print(new.shape)
            for name in label_names:
                index_json = label_names.index(name)
                print("Label name causing the error:", name)
                index_all = classes.index(name)
                print(index_all,"1111",index_json,np.array(lbl))
                print(new.shape)

                print("11111",np.array(lbl).shape)
                new = new + index_all*(np.array(lbl) == index_json)
                print(new.shape)
            print(new.shape)

            utils.lblsave(osp.join(pngs_path, count[i].split(".")[0]+'.png'), new)
            print('Saved ' + count[i].split(".")[0] + '.jpg and ' + count[i].split(".")[0] + '.png')
