from ultralytics.data.utils import autosplit

autosplit(  
    path="fish_no_fish/test_scripts/data_dump/images",
    weights=(0.8, 0.2, 0.0),  # (train, validation, test) fractional splits
    annotated_only=True,  # split only images with annotation file when True
)
