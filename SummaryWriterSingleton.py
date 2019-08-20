from torch.utils.tensorboard import SummaryWriter

class SummaryWriterSingleton:
    """This wrapper class ensures there is only one SummaryWriter per process"""

    #class variable to hold the Summary Writer instance
    instance = None

    def __init__(self, *args, **kwargs):

        #Check if the Summary Writer has already been created
        if not SummaryWriterSingleton.instance:

            #Create the instance of the summary writer and assign it to the class instance
            SummaryWriterSingleton.instance = SummaryWriter(*args, **kwargs)
        
    #map attributes requests for this singleton instance to the single summary writer instance
    def __getattr__(self, name):

        return getattr(SummaryWriterSingleton.instance, name)