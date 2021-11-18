import argparse
import numpy as np
import tensorflow 
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from utils import mnist_keras, myCallback, architecture, save_model


#command line argument parser
def parse_args():

    # Creates the parser 
    parser = argparse.ArgumentParser(
        description="Train Livliness model", 
        allow_abbrev=False)

    #Adding arguments for model training 
    parser.add_argument('--train_dataset_path', type=str,
                        default=None,
                        help='Path to directory containing the different classes for the train data')

    parser.add_argument('--val_dataset_path', type=str,
                        default=None,
                        help='Path to directory containing the different classes for the val data')

    parser.add_argument('--target_size', type=int,
                        default=28,
                        help='The size of the input to be fed into the neural network')

    parser.add_argument('--batch_size', type=int,
                        default=128,
                        help='Number of training examples utilized in one iteration')
    

    parser.add_argument('--epoch', type=int,
                        default=20,
                        help='Number of complete passes through the training dataset')

    parser.add_argument('--class_mode', type=str,
                        default="categorical",
                        help='Type of class. binary or categorical')

    parser.add_argument('--save_path', type=str,
                        default="saved_model",
                        help='Directory to save model artifacts')

    parser.add_argument('--num_classes', type=int,
                        default=10,
                        help='Number of classes')
                        

    args =parser.parse_args()

    return args

# Using in-built tensorflow MNIST dataset for model training 
def train_mnist_conv():

    args = parse_args()
    model = architecture()
    callback = myCallback() 
    x_train, x_test, y_train, y_test = mnist_keras()
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', 
                  metrics=['accuracy'])
    
    # model fitting
    history = model.fit(
        x_train, y_train, epochs=args.epoch, batch_size=args.batch_size,
        validation_data=(x_test, y_test), callbacks=[callback]
    )

    #save model
    save_model(args.save_path , model)

    return history, history.history['accuracy'][-1]

# Training with custom dataset 
def train_custom_dataset():

    args = parse_args()
    model = architecture(input_shape=(args.target_size, args.target_size, 3), 
                         classes=args.num_classes)
    # callback = myCallback() 

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy']) 
  
    train_dir = args.train_dataset_path
    val_dir = args.val_dataset_path

    # Generate Train data
    train_datagen = ImageDataGenerator(rescale = 1./255)
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size = (args.target_size, args.target_size),
        batch_size =  args.batch_size,
        class_mode = args.class_mode,
        shuffle = True
        )

    # Generate Validation data
    validation_datagen = ImageDataGenerator(rescale = 1./255) 

    val_generator = validation_datagen.flow_from_directory(
        val_dir,
        target_size =(args.target_size, args.target_size),
        batch_size = args.batch_size,
        class_mode = args.class_mode,
        shuffle = True
    )

    STEP_SIZE_TRAIN=train_generator.n//train_generator.batch_size
    STEP_SIZE_VALID=val_generator.n//val_generator.batch_size
    
    # model fitting
    history = model.fit_generator(
        train_generator,
        steps_per_epoch = STEP_SIZE_TRAIN,
        validation_data=val_generator,
        validation_steps=STEP_SIZE_VALID,
        epochs = args.epoch,
        verbose = 2)
    
    #save model
    save_model(args.save_path , model)
    
    return history, history.history['accuracy'][-1]


if __name__ == "__main__":


    args = parse_args()

    if args.train_dataset_path == None or args.val_dataset_path == None:
        _, _ = train_mnist_conv()

    else:
        _, _ = train_custom_dataset()
    