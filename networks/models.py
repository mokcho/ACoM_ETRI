from trainers.dcase2023t2_ae import DCASE2023T2AE
from trainers.encodec_ae import EnCodecAETrainer


class Models:
    ModelsDic = {
        "DCASE2023T2-AE":DCASE2023T2AE,
        "EnCodec-AE" : EnCodecAETrainer
    }

    def __init__(self,models_str):
        self.net = Models.ModelsDic[models_str]

    def show_list(self):
        return Models.ModelsDic.keys()
