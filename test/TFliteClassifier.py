import tensorflow as tf
import numpy as np
import cv2
_R_MEAN = 128
_G_MEAN = 128
_B_MEAN = 128
_CHANNEL_MEANS = [_R_MEAN, _G_MEAN, _B_MEAN]
class TFLiteClassifier(object):
    def __init__(self,tflite_model_file,input_size,quant=True):
        self.input_size=input_size
        self.quant=quant
        self.interpreter = tf.contrib.lite.Interpreter(model_path=str(tflite_model_file))
        self.interpreter.allocate_tensors()
        self.input_index = self.interpreter.get_input_details()[0]['index']
        self.outputs = self.interpreter.get_output_details()

        self.cls = self.outputs[0]['index']

    def inference(self,img):
        '''
        :param img: RGB-uint8
        :return: label
        '''
        img = cv2.resize(img, (self.input_size, self.input_size))
        if not self.quant:
            img=img-np.reshape(_CHANNEL_MEANS,(1,1,3))
            img=img.astype(np.uint8)
        self.interpreter.set_tensor(self.input_index, [img])
        self.interpreter.invoke()
        cls_val= self.interpreter.get_tensor(self.cls)[0]
        return np.argmax(cls_val)


if __name__ == '__main__':
    import glob
    import os
    import tqdm
    import shutil
    input_dir='/home/sixd-ailabs/Develop/Human/Hand/hand_dataset/hand-classification2'
    input_list_file='/home/sixd-ailabs/Develop/Human/Hand/hand_dataset/hand-classification2/val.txt'
    cate_dir={'background':0, 'hand':1}
    input_list=[]
    with open(input_list_file,'r') as f:
        input_list=f.read().split('\n')
    SIZE=64
    # classifier=TFLiteClassifier('/home/sixd-ailabs/Downloads/models_uqtf_eval/model_quant.tflite')
    classifier=TFLiteClassifier('/home/sixd-ailabs/Develop/DL/MobileDL/PocketFlow/models_uqtf_eval/model_quant.tflite',SIZE)
    total=0
    correct=0
    for input_file in tqdm.tqdm(input_list):
        # input_file=input_list[-100]
        if len(input_file)>1:
            total+=1
            words=os.path.join(input_dir, input_file).split(' ')
            input_file= words[0]
            label=cate_dir[words[1]]
            # if label==0:
            #     continue
            img=cv2.imread(input_file)
            orig=img
            img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
            cls=classifier.inference(img)
            if cls==label:
                correct+=1
            else:
                print(input_file,'ret={}'.format(cls),'gt={}'.format(label))
                # shutil.copyfile(input_file,)
                # cv2.imshow("img", orig)
                # cv2.waitKey(0)

    print('accuracy:{}'.format(correct*1.0/total))