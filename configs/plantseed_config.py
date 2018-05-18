'''
contains parameters ranging from location of i/o files to model paramters
'''
class config():
    print("[Using Plant seedling config]")
    # experiement_name = "cifar_one_cycle_10"
    input_size = (224, 224)
    input_space = "RGB"
    use_tencrop = False #For inference
    rotation = False
    h_flip = True
    v_flip = False
    brightness, hue, saturation, contrast = 0.0, 0.0, 0.0, 0.0

    normalize = True
    mean =  ( 0.3191,  0.2870,  0.1943) #(0.485, 0.456, 0.406)
    std = (0.083571,  0.087252,  0.094464) #(0.229, 0.224, 0.225)
    use_gpu = True
    use_multi_gpu = False
    num_workers = 4

    # model_name = "resnet50"
    num_classes = 12
    save_loc = "/fractal/home/kunal/kaggleplantseedlings/checkpoints/"
    # save_name = "cifar_iter10.pth"


    # use_imagenet_pretrained_weights = True
    # last_epoch = 0
    # pretrained_weights_loc= None
    # global_pooling = True #object() or pooling fucntion u want to use
    # drop_out = False
    # classifier_factory = False
    # use_original_classifier = True if classifier_factory is None else False
    # metric = "accuracy"
    # metric_n = 3 # Set this to zero if u are not using this. # I am very sick about the way this is done. Sorry.
    # metric_maximize = True
    # which_to_maximize = "metric_n" ## Takes two values, metric_n and metric.

    home_loc = "/fractal/home/kunal/kaggleplantseedlings/"
    train_data_loc = home_loc+"data/x_train.txt"
    # test_data_loc = home_loc+"x_test.txt"
    val_data_loc = home_loc+"data/x_valid.txt"
    inference_correct_loc = "/fractal/home/kunal/kaggleplantseedlings/correct_prediction.txt"
    inference_incorrect_loc = "/fractal/home/kunal/kaggleplantseedlings/incorrect_prediction.txt"
    #labels_loc = home_loc+"labels.txt"
    # batch_size = 1024
    # epochs = 300
    # loss_type = "cross_entropy_loss"
    # class_weight = None


    ## Schedulars and learning rates
    # use_scheduler = True
    lr = 0.002
    weight_decay = 0.0005
    momentum = 0.90
    batch_size = 64
    nesterov = False
    use_checkpoint = False
    epochs = 15
    # optimizer = "adam" #sgd and adam are available for now
    # scheduler_name = "one_cycle"
    # scheduler_type = "batch" # takes  "epoch" and "batch"
    # early_stopping = 100
    # use_differential_training = False
    # kwargs = { "scheduler_name": scheduler_name, "max_iterations": 10000, "base_lr": 0.01, "max_lr": 0.5,
    #          "mom_max": 0.95, "mom_min":0.85} 
    # #kwargs = {"gamma": 0.1, "lr_decay_epoch": 30,} # Use this in general for "epoch" scheduler type
    # #kwargs = {"base_lr": 1e-3, "max_lr": 6e-3, "step_size": 2000, "mode": "triangular", "gamma": 1.,"scale_fn": None, "scale_mode": "cycle"}
    # # Use this in general for "batch" schedular type.

    # ## Resume training
    # epoch_completed = 0
    # use_visdom = True
    # inference_imgs=10000