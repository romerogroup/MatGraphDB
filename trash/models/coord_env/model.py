import os

from matgraphdb.utils import Timer
from matgraphdb.models.coord_env.classify import Trainer

class COORD_ENV:
    """
    COORD ENV object detection model.
    """

    def train(self,config_file=None,**kwargs):
        timer = Timer()
        trainer=Trainer()

        timer.start_event("Initializing Configurations")
        trainer.load_config(input_file=config_file)
        timer.end_event()

        timer.start_event("Initializing Datasets")
        trainer.initialize_datasets()
        timer.end_event()

        timer.start_event("Initializing Model")
        trainer.initialize_model()
        timer.end_event()
        
        timer.start_event("Estimating Required Memory")
        trainer.get_required_memory()
        timer.end_event()

        timer.start_event("Starting Train Loop")
        trainer.train()
        timer.end_event()

        timer.start_event("Saving in Train directory")
        trainer.save_training()
        timer.end_event()

        timer.save_events(filename=os.path.join(trainer.run_dir,"time_profile.txt"))

    def val():
        pass

    def predict():
        pass

    def export():
        pass