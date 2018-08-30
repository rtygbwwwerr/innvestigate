import warnings
warnings.simplefilter('ignore')

import numpy as np
import imp
import time

import keras
import keras.backend
import keras.models

import innvestigate
import innvestigate.applications
import innvestigate.applications.mnist
import innvestigate.utils as iutils
import innvestigate.utils.visualizations as ivis


if __name__ == "__main__":
    eutils = imp.load_source("utils", "utils.py")
    mnistutils = imp.load_source("utils_mnist", "utils_mnist.py")
    
    
    # Load data
    channels_first = keras.backend.image_data_format() == "channels_first"
    data = mnistutils.fetch_data(channels_first) #returns x_train, y_train, x_test, y_test as numpy.ndarray
    num_classes = len(np.unique(data[1]))
    
    # Test samples for illustrations
    images = [(data[2][i].copy(), data[3][i]) for i in range(num_classes)]
    label_to_class_name = [str(i) for i in range(num_classes)]
    
    
    # Select a model architecture and configure training parameters.
    modelname = 'mlp_3dense'
    activation_type = 'relu'
    input_range = [-1, 1]
    epochs = 3
    batch_size = 256
    create_model_kwargs = {'dense_units':1024, 'dropout_rate':0.25}
    
    # Preprocess data
    data_preprocessed = (mnistutils.preprocess(data[0], input_range), data[1],
                         mnistutils.preprocess(data[2], input_range), data[3])
    
    
    # Create & (optionally) train model
    model, modelp = mnistutils.create_model(channels_first, modelname, **create_model_kwargs)
    mnistutils.train_model(modelp, data_preprocessed, batch_size=batch_size, epochs=epochs)
    model.set_weights(modelp.get_weights())
    
    
    
    # Determine analysis methods and properties
    methods = [
        # NAME                    OPT.PARAMS               POSTPROC FXN                TITLE
    
        # Show input
#         ("input",                 {},                       mnistutils.image,          "Input"),
#     
#         # Function
#         ("gradient",              {},                       mnistutils.graymap,        "Gradient"),
#         ("smoothgrad",            {"noise_scale": 50},      mnistutils.graymap,        "SmoothGrad"),
#         ("integrated_gradients",  {},                       mnistutils.graymap,        "Integrated Gradients"),
#     
#         # Signal
#         ("deconvnet",             {},                       mnistutils.bk_proj,        "Deconvnet"),
#         ("guided_backprop",       {},                       mnistutils.bk_proj,        "Guided Backprop",),
#         ("pattern.net",           {},                       mnistutils.bk_proj,        "PatternNet"),
    
        # Interaction
#         ("lrp.z",                 {},                       mnistutils.heatmap,         "LRP-Z"),
        ("lrp.alpha_1_beta_0",                 {},                       mnistutils.heatmap,         "LRP-alpha_1_beta_0")
#         ("lrp.epsilon",           {"epsilon": 1},           mnistutils.heatmap,         "LRP-Epsilon"),
        ]

    # Create analyzers.
    analyzers = []
    print('Creating analyzer instances. ')
    for method in methods:
        analyzer = innvestigate.create_analyzer(method[0],   # analysis method identifier
                                                model,       # model without softmax output
                                                **method[1]) # optional analysis parameters
        # some analyzers require additional training. For those
        analyzer.fit(data_preprocessed[0],
                     pattern_type=activation_type,
                     batch_size=256, verbose=1)
        analyzers.append(analyzer)
    
    print('Running analyses.')
    # Apply analyzers to trained model.
    analysis = np.zeros([len(images), len(analyzers), 28, 28, 3])
    text = []
    for i, (image, y) in enumerate(images):
        print('Image {}: '.format(i), end='')
        t_start = time.time()
        image = image[None, :, :, :]
        
        # Predict label.
        x = mnistutils.preprocess(image, input_range)
        presm = model.predict_on_batch(x)[0] #forward pass without softmax
        prob = modelp.predict_on_batch(x)[0] #forward pass with softmax
        y_hat = prob.argmax()
        
        # Save prediction info:
        text.append(("%s" %label_to_class_name[y],    # ground truth label
                     "%.2f" %presm.max(),             # pre-softmax logits
                     "%.2f" %prob.max(),              # probabilistic softmax output  
                     "%s" %label_to_class_name[y_hat] # predicted label
                    ))
        
        for aidx, analyzer in enumerate(analyzers):
            
            is_input_analyzer = methods[aidx][0] == "input"
            # Analyze.
            a = analyzer.analyze(image if is_input_analyzer else x)
            
            # Postprocess.
            if not is_input_analyzer:
                a = mnistutils.postprocess(a)
            a = methods[aidx][2](a)
            analysis[i, aidx] = a[0]
        t_elapsed = time.time() - t_start
        print('{:.4f}s'.format(t_elapsed))
        
        
        

    # Plot the analysis.# Plot  
    grid = [[analysis[i, j] for j in range(analysis.shape[1])]
            for i in range(analysis.shape[0])]
    label, presm, prob, pred = zip(*text)
    row_labels_left = [('label: {}'.format(label[i]),'pred: {}'.format(pred[i])) for i in range(len(label))]
    row_labels_right = [('logit: {}'.format(presm[i]),'prob: {}'.format(prob[i])) for i in range(len(label))]
    col_labels = [''.join(method[3]) for method in methods]
    
    eutils.plot_image_grid(grid, row_labels_left, row_labels_right, col_labels)
        
