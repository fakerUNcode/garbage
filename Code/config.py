
class DefaultConfig(object):
    TrainWithGPU = True
    LogDir = "checkpoint/exp3"
    DataSetDir = "./TrashBig"
    SetGroup = {'Recyclables': ('cardboard', 'glass', 'metal', 'plastic', 'clothes', 'paper'),
                       'Kitchen waste': ('bananapeel', 'vegetable'),
                       'Hazardous waste': ('battery', 'lightbulb', 'drugs'),
                       'Other': ('papercup',)}

    EpochNum = 150
    BatchSize = 12
    LearningRate = 0.0002
    LRScheduleStep = 40

    # inference.py
    DataSetInfoPath = "checkpoint/exp2/DataSetInfo.pth"
    CkptPath = 'checkpoint/exp2/ckpt_149.pth'
    InferWithGPU = False


