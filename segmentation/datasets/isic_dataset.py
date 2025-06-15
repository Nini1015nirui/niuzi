from mmseg.datasets.basesegdataset import BaseSegDataset
from mmseg.registry import DATASETS


@DATASETS.register_module()
class ISICDataset(BaseSegDataset):
    """ISIC2017 皮肤病变分割数据集"""
    
    METAINFO = dict(
        classes=('background', 'lesion'),
        palette=[[0, 0, 0], [255, 255, 255]]
    )
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)