import os
os.environ["KERAS_BACKEND"] = "tensorflow"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import numpy as np
import keras
from keras.regularizers import *
import mltools,dataset2016
import BLOCK as bl
import argparse
import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.5
keras.backend.tensorflow_backend.set_session(tf.Session(config=config))

if __name__ == "__main__":
    # Set up some params
    parser = argparse.ArgumentParser(description="BLOCK")
    parser.add_argument("--epoch", type=int, default=500, help='Max number of training epochs')
    parser.add_argument("--batch_size", type=int, default=400, help="Training batch size")
    parser.add_argument("--filepath", type=str, default='./weights.h5', help='Path for saving and reloading the weight')
    parser.add_argument("--datasetpath", type=str, default='../../../Dataset/RML2016.10a_dict.pkl', help='Path for the dataset')
    parser.add_argument("--data", type=int, default=0, help='Select the RadioML2016.10a or RadioML2016.10b, 0 or 1')
    opt = parser.parse_args()

    K.set_image_data_format('channels_last')
    print(K.image_data_format())

    (mods,snrs,lbl),(X_train,Y_train),(X_val,Y_val),(X_test,Y_test),(train_idx,val_idx,test_idx) = \
        dataset2016.load_data(opt.datasetpath,opt.data)

    X1_train=np.expand_dims(X_train[:,0,:], axis=2)
    X1_test=np.expand_dims(X_test[:,0,:], axis=2)
    X1_val=np.expand_dims(X_val[:,0,:],axis=2)

    X2_train=np.expand_dims(X_train[:,1,:], axis=2)
    X2_test=np.expand_dims(X_test[:,1,:], axis=2)
    X2_val=np.expand_dims(X_val[:,1,:],axis=2)

    X_train=np.expand_dims(X_train,axis=3)
    X_test=np.expand_dims(X_test,axis=3)
    X_val=np.expand_dims(X_val,axis=3)

    # Build model
    if opt.data==0:
        model = bl.BLOCK(classes=11)
    elif opt.data==1:
        model = bl.BLOCK(classes=10)
    else:
        print('use correct data number: 0 or 1')
    model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')
    model.summary()
    history = model.fit([X_train,X1_train,X2_train],
        Y_train,
        batch_size=opt.batch_size,
        epochs=opt.epoch,
        verbose=2,
        validation_data=([X_val,X1_val,X2_val],Y_val),
        callbacks = [
                    keras.callbacks.ModelCheckpoint(opt.filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='auto'),
                    keras.callbacks.ReduceLROnPlateau(monitor='val_loss',factor=0.8,verbose=1,patince=10,min_lr=0.0000001),
                    keras.callbacks.EarlyStopping(monitor='val_loss', patience=60, verbose=1, mode='auto'),
                    # keras.callbacks.TensorBoard(histogram_freq=1,write_graph=True,write_images=True,batch_size=opt.batch_size)
                    ]
                        )
    model.load_weights(opt.filepath)
    mltools.show_history(history)
    score = model.evaluate([X_test,X1_test,X2_test], Y_test, verbose=1, batch_size=opt.batch_size)
    print(score)


