import os,random
import time
from keras.utils import plot_model
os.environ["KERAS_BACKEND"] = "tensorflow"
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
import numpy as np
import keras
from keras.regularizers import *
import mltools,dataset2016
import BLOCK as bl
import argparse
import tensorflow as tf
import pandas as pd

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.5
keras.backend.tensorflow_backend.set_session(tf.Session(config=config))

if __name__ == "__main__":
    # Set up some params
    parser = argparse.ArgumentParser(description="MCLDNN")
    parser.add_argument("--epoch", type=int, default=500, help='Max number of training epochs')
    parser.add_argument("--batch_size", type=int, default=400, help="Training batch size")
    parser.add_argument("--filepath", type=str, default='./localmodel/GlobalModel.hdf5', help='Path for saving and reloading the weight')
    parser.add_argument("--datasetpath", type=str, default='../../../Dataset/RML2016.10a_dict.pkl', help='Path for the dataset')
    parser.add_argument("--data", type=int, default=0, help='Select the RadioML2016.10a or RadioML2016.10b, 0 or 1')
    opt = parser.parse_args()

    # Set Keras data format as channels_last
    K.set_image_data_format('channels_last')
    print(K.image_data_format())

    (mods,snrs,lbl),(X_train,Y_train),(X_val,Y_val),(X_test,Y_test),(train_idx,val_idx,test_idx) = \
        dataset2016.load_data(opt.datasetpath,opt.data)

    # Select the data set of the real part and the imaginary part, separately
    # and expand the data set dimension
    X1_train=np.expand_dims(X_train[:,0,:], axis=2)
    X1_test=np.expand_dims(X_test[:,0,:], axis=2)
    X1_val=np.expand_dims(X_val[:,0,:],axis=2)
    #We assume that there are 10 EDs and 1 CD. Hence,
    #Each dataset is divided into 10 equal parts to simulate 10 different EDs.
    X1_train_d1 = X1_train[0:13200]
    X1_train_d2 = X1_train[13200:26400]
    X1_train_d3 = X1_train[26400:39600]
    X1_train_d4 = X1_train[39600:52800]
    X1_train_d5 = X1_train[52800:66000]
    X1_train_d6 = X1_train[66000:79200]
    X1_train_d7 = X1_train[79200:92400]
    X1_train_d8 = X1_train[92400:105600]
    X1_train_d9 = X1_train[105600:118800]
    X1_train_d10 = X1_train[118800:132000]
    X1_val_d1 = X1_val[0:4400]
    X1_val_d2 = X1_val[4400:8800]
    X1_val_d3 = X1_val[8800:13200]
    X1_val_d4 = X1_val[13200:17600]
    X1_val_d5 = X1_val[17600:22000]
    X1_val_d6 = X1_val[22000:26400]
    X1_val_d7 = X1_val[26400:30800]
    X1_val_d8 = X1_val[30800:35200]
    X1_val_d9 = X1_val[35200:39600]
    X1_val_d10 = X1_val[39600:44000]

    X2_train=np.expand_dims(X_train[:,1,:], axis=2)
    X2_test=np.expand_dims(X_test[:,1,:], axis=2)
    X2_val=np.expand_dims(X_val[:,1,:],axis=2)
    X2_train_d1 = X2_train[0:13200]
    X2_train_d2 = X2_train[13200:26400]
    X2_train_d3 = X2_train[26400:39600]
    X2_train_d4 = X2_train[39600:52800]
    X2_train_d5 = X2_train[52800:66000]
    X2_train_d6 = X2_train[66000:79200]
    X2_train_d7 = X2_train[79200:92400]
    X2_train_d8 = X2_train[92400:105600]
    X2_train_d9 = X2_train[105600:118800]
    X2_train_d10 = X2_train[118800:132000]
    X2_val_d1 = X2_val[0:4400]
    X2_val_d2 = X2_val[4400:8800]
    X2_val_d3 = X2_val[8800:13200]
    X2_val_d4 = X2_val[13200:17600]
    X2_val_d5 = X2_val[17600:22000]
    X2_val_d6 = X2_val[22000:26400]
    X2_val_d7 = X2_val[26400:30800]
    X2_val_d8 = X2_val[30800:35200]
    X2_val_d9 = X2_val[35200:39600]
    X2_val_d10 = X2_val[39600:44000]

    X_train=np.expand_dims(X_train,axis=3)
    X_test=np.expand_dims(X_test,axis=3)
    X_val=np.expand_dims(X_val,axis=3)
    X_train_d1 = X_train[0:13200]
    X_train_d2 = X_train[13200:26400]
    X_train_d3 = X_train[26400:39600]
    X_train_d4 = X_train[39600:52800]
    X_train_d5 = X_train[52800:66000]
    X_train_d6 = X_train[66000:79200]
    X_train_d7 = X_train[79200:92400]
    X_train_d8 = X_train[92400:105600]
    X_train_d9 = X_train[105600:118800]
    X_train_d10 = X_train[118800:132000]
    X_val_d1 = X_val[0:4400]
    X_val_d2 = X_val[4400:8800]
    X_val_d3 = X_val[8800:13200]
    X_val_d4 = X_val[13200:17600]
    X_val_d5 = X_val[17600:22000]
    X_val_d6 = X_val[22000:26400]
    X_val_d7 = X_val[26400:30800]
    X_val_d8 = X_val[30800:35200]
    X_val_d9 = X_val[35200:39600]
    X_val_d10 = X_val[39600:44000]

    Y_train_d1 = Y_train[0:13200]
    Y_train_d2 = Y_train[13200:26400]
    Y_train_d3 = Y_train[26400:39600]
    Y_train_d4 = Y_train[39600:52800]
    Y_train_d5 = Y_train[52800:66000]
    Y_train_d6 = Y_train[66000:79200]
    Y_train_d7 = Y_train[79200:92400]
    Y_train_d8 = Y_train[92400:105600]
    Y_train_d9 = Y_train[105600:118800]
    Y_train_d10 = Y_train[118800:132000]
    Y_val_d1 = Y_val[0:4400]
    Y_val_d2 = Y_val[4400:8800]
    Y_val_d3 = Y_val[8800:13200]
    Y_val_d4 = Y_val[13200:17600]
    Y_val_d5 = Y_val[17600:22000]
    Y_val_d6 = Y_val[22000:26400]
    Y_val_d7 = Y_val[26400:30800]
    Y_val_d8 = Y_val[30800:35200]
    Y_val_d9 = Y_val[35200:39600]
    Y_val_d10 = Y_val[39600:44000]

    # Build framework (model)
    if opt.data==0:
        model = bl.BLOCK(classes=11)
        # plot_model(model, to_file='CNNmodel.png', show_shapes=True, show_layer_names=False)
    elif opt.data==1:
        model = bl.BLOCK(classes=10)
    else:
        print('use correct data number: 0 or 1')

    model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')
    model.summary()
    model.save("m.hdf5")
    model.save("./localmodel/LocalModel1.hdf5")
    model.save("./localmodel/LocalModel2.hdf5")
    model.save("./localmodel/LocalModel3.hdf5")
    model.save("./localmodel/LocalModel4.hdf5")
    model.save("./localmodel/LocalModel5.hdf5")
    model.save("./localmodel/LocalModel6.hdf5")
    model.save("./localmodel/LocalModel7.hdf5")
    model.save("./localmodel/LocalModel8.hdf5")
    model.save("./localmodel/LocalModel9.hdf5")
    model.save("./localmodel/LocalModel10.hdf5")
    model.save("./localmodel/GlobalModel.hdf5")

    # training process
    def averagemodel(weightpath, model):
        weights = []
        new_weights = list()
        for i in range(len(weightpath)):
            model.load_weights(weightpath[i])
            weight = model.get_weights()
            weights.append(weight)
        for weights_list_tuple in zip(*weights):
            new_weights.append([np.array(weights_).mean(axis=0) \
                                for weights_ in zip(*weights_list_tuple)])
        model.set_weights(new_weights)
        model.save_weights('./localmodel/GlobalModel.hdf5')
        return model
    weightpath \
        = ['./localmodel/LocalModel1.hdf5', './localmodel/LocalModel2.hdf5', './localmodel/LocalModel3.hdf5',
           './localmodel/LocalModel4.hdf5',
           './localmodel/LocalModel5.hdf5', './localmodel/LocalModel6.hdf5', './localmodel/LocalModel7.hdf5',
           './localmodel/LocalModel8.hdf5',
           './localmodel/LocalModel9.hdf5', './localmodel/LocalModel10.hdf5']
    averagemodel(weightpath, model)
    accy = []
    lossy = []
    val_accy = []
    val_lossy = []
    start_epoch = -1  # 0 train -1 retrain
    sub_epoch = 1
    train_start_time = time.time()
    for epoch in range(opt.epoch):
        print("***************epoch=" + str(epoch + 1) + "***************")
        # SubModel1
        print("***************epoch=" + str(epoch + 1) + "**************1")
        if epoch != start_epoch:
            model.load_weights("./localmodel/GlobalModel.hdf5")
        history1 = model.fit([X_train_d1, X1_train_d1, X2_train_d1],
                            Y_train_d1,
                            batch_size=opt.batch_size,
                            epochs=sub_epoch,
                            verbose=2,
                            validation_data=([X_val_d1, X1_val_d1, X2_val_d1], Y_val_d1),
                            callbacks=[
                                keras.callbacks.ModelCheckpoint(weightpath[0], monitor='val_loss', verbose=1,
                                                                save_best_only=True, mode='auto'),
                                keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.8, verbose=1, patince=5,
                                                                  min_lr=0.0000001)])
        accy.append(history1.history['acc'])
        lossy.append(history1.history['loss'])
        val_accy.append(history1.history['val_acc'])
        val_lossy.append(history1.history['val_loss'])

        # SubModel2
        print("***************epoch=" + str(epoch + 1) + "**************2")
        if epoch != start_epoch:
            model.load_weights("./localmodel/GlobalModel.hdf5")
        model.fit([X_train_d2, X1_train_d2, X2_train_d2],
                            Y_train_d2,
                            batch_size=opt.batch_size,
                            epochs=sub_epoch,
                            verbose=2,
                            validation_data=([X_val_d2, X1_val_d2, X2_val_d2], Y_val_d2),
                            callbacks=[
                                keras.callbacks.ModelCheckpoint(weightpath[1], monitor='val_loss', verbose=1,
                                                                save_best_only=True, mode='auto'),
                                keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.8, verbose=1, patince=5,
                                                                  min_lr=0.0000001)])

        # SubModel3
        print("***************epoch=" + str(epoch + 1) + "**************3")
        if epoch != start_epoch:
            model.load_weights("./localmodel/GlobalModel.hdf5")
        model.fit([X_train_d3, X1_train_d3, X2_train_d3],
                            Y_train_d3,
                            batch_size=opt.batch_size,
                            epochs=sub_epoch,
                            verbose=2,
                            validation_data=([X_val_d3, X1_val_d3, X2_val_d3], Y_val_d3),
                            callbacks=[
                                keras.callbacks.ModelCheckpoint(weightpath[2], monitor='val_loss', verbose=1,
                                                                save_best_only=True, mode='auto'),
                                keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.8, verbose=1, patince=5,
                                                                  min_lr=0.0000001)])

        # SubModel4
        print("***************epoch=" + str(epoch + 1) + "**************4")
        if epoch != start_epoch:
            model.load_weights("./localmodel/GlobalModel.hdf5")
        model.fit([X_train_d4, X1_train_d4, X2_train_d4],
                            Y_train_d4,
                            batch_size=opt.batch_size,
                            epochs=sub_epoch,
                            verbose=2,
                            validation_data=([X_val_d4, X1_val_d4, X2_val_d4], Y_val_d4),
                            callbacks=[
                                keras.callbacks.ModelCheckpoint(weightpath[3], monitor='val_loss', verbose=1,
                                                                save_best_only=True, mode='auto'),
                                keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.8, verbose=1, patince=5,
                                                                  min_lr=0.0000001)])

        # SubModel5
        print("***************epoch=" + str(epoch + 1) + "**************5")
        if epoch != start_epoch:
            model.load_weights("./localmodel/GlobalModel.hdf5")
        model.fit([X_train_d5, X1_train_d5, X2_train_d5],
                            Y_train_d5,
                            batch_size=opt.batch_size,
                            epochs=sub_epoch,
                            verbose=2,
                            validation_data=([X_val_d5, X1_val_d5, X2_val_d5], Y_val_d5),
                            callbacks=[
                                keras.callbacks.ModelCheckpoint(weightpath[4], monitor='val_loss', verbose=1,
                                                                save_best_only=True, mode='auto'),
                                keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.8, verbose=1, patince=5,
                                                                  min_lr=0.0000001)])

        # SubModel6
        print("***************epoch=" + str(epoch + 1) + "**************6")
        if epoch != start_epoch:
            model.load_weights("./localmodel/GlobalModel.hdf5")
        model.fit([X_train_d6, X1_train_d6, X2_train_d6],
                            Y_train_d6,
                            batch_size=opt.batch_size,
                            epochs=sub_epoch,
                            verbose=2,
                            validation_data=([X_val_d6, X1_val_d6, X2_val_d6], Y_val_d6),
                            callbacks=[
                                keras.callbacks.ModelCheckpoint(weightpath[5], monitor='val_loss', verbose=1,
                                                                save_best_only=True, mode='auto'),
                                keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.8, verbose=1, patince=5,
                                                                  min_lr=0.0000001)])

        # SubModel7
        print("***************epoch=" + str(epoch + 1) + "**************7")
        if epoch != start_epoch:
            model.load_weights("./localmodel/GlobalModel.hdf5")
        model.fit([X_train_d7, X1_train_d7, X2_train_d7],
                            Y_train_d7,
                            batch_size=opt.batch_size,
                            epochs=sub_epoch,
                            verbose=2,
                            validation_data=([X_val_d7, X1_val_d7, X2_val_d7], Y_val_d7),
                            callbacks=[
                                keras.callbacks.ModelCheckpoint(weightpath[6], monitor='val_loss', verbose=1,
                                                                save_best_only=True, mode='auto'),
                                keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.8, verbose=1, patince=5,
                                                                  min_lr=0.0000001)])

        # SubModel8
        print("***************epoch=" + str(epoch + 1) + "**************8")
        if epoch != start_epoch:
            model.load_weights("./localmodel/GlobalModel.hdf5")
        model.fit([X_train_d8, X1_train_d8, X2_train_d8],
                            Y_train_d8,
                            batch_size=opt.batch_size,
                            epochs=sub_epoch,
                            verbose=2,
                            validation_data=([X_val_d8, X1_val_d8, X2_val_d8], Y_val_d8),
                            callbacks=[
                                keras.callbacks.ModelCheckpoint(weightpath[7], monitor='val_loss', verbose=1,
                                                                save_best_only=True, mode='auto'),
                                keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.8, verbose=1, patince=5,
                                                                  min_lr=0.0000001)])

        # SubModel9
        print("***************epoch=" + str(epoch + 1) + "**************9")
        if epoch != start_epoch:
            model.load_weights("./localmodel/GlobalModel.hdf5")
        model.fit([X_train_d9, X1_train_d9, X2_train_d9],
                            Y_train_d9,
                            batch_size=opt.batch_size,
                            epochs=sub_epoch,
                            verbose=2,
                            validation_data=([X_val_d9, X1_val_d9, X2_val_d9], Y_val_d9),
                            callbacks=[
                                keras.callbacks.ModelCheckpoint(weightpath[8], monitor='val_loss', verbose=1,
                                                                save_best_only=True, mode='auto'),
                                keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.8, verbose=1, patince=5,
                                                                  min_lr=0.0000001)])

        # SubModel10
        print("***************epoch=" + str(epoch + 1) + "**************10")
        if epoch != start_epoch:
            model.load_weights("./localmodel/GlobalModel.hdf5")
        model.fit([X_train_d10, X1_train_d10, X2_train_d10],
                            Y_train_d10,
                            batch_size=opt.batch_size,
                            epochs=sub_epoch,
                            verbose=2,
                            validation_data=([X_val_d10, X1_val_d10, X2_val_d10], Y_val_d10),
                            callbacks=[
                                keras.callbacks.ModelCheckpoint(weightpath[9], monitor='val_loss', verbose=1,
                                                                save_best_only=True, mode='auto'),
                                keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.8, verbose=1, patince=5,
                                                                  min_lr=0.0000001)])

        # gobel model
        model = averagemodel(weightpath, model)

    train_time = time.time() - train_start_time
    # store np_accy
    data_np_accy = pd.DataFrame(accy)
    writer = pd.ExcelWriter("./excel/DecentAMC_train_acc.xlsx")
    data_np_accy.to_excel(writer, 'page_1', float_format='%.5f')
    writer.save()
    writer.close()

    # store np_lossy
    data_np_lossy = pd.DataFrame(lossy)
    writer = pd.ExcelWriter("./excel/DecentAMC_train_loss.xlsx")
    data_np_lossy.to_excel(writer, 'page_1', float_format='%.5f')
    writer.save()
    writer.close()

    # store np_val_accy
    data_np_val_accy = pd.DataFrame(val_accy)
    writer = pd.ExcelWriter("./excel/DecentAMC_val_acc.xlsx")
    data_np_val_accy.to_excel(writer, 'page_1', float_format='%.5f')
    writer.save()
    writer.close()

    # store np_val_lossy
    data_np_val_lossy = pd.DataFrame(val_lossy)
    writer = pd.ExcelWriter("./excel/DecentAMC_val_loss.xlsx")
    data_np_val_lossy.to_excel(writer, 'page_1', float_format='%.5f')
    writer.save()
    writer.close()

    # We re-load the best weights once training is finished
    model.load_weights("./localmodel/GlobalModel.hdf5")
    # Show simple version of performance
    score = model.evaluate([X_test,X1_test,X2_test], Y_test, verbose=1, batch_size=opt.batch_size)




