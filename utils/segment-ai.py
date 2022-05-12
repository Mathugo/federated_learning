
API = "6dd07692024b2ec218992550eb0c7084a5f018f2"

# pip install segments-ai
from segments import SegmentsClient, SegmentsDataset
from segments.utils import export_dataset

# Initialize a SegmentsDataset from the release file
client = SegmentsClient(API)
release = client.get_release('bobochelemioche1601/Orange', 'v0.1') # Alternatively: release = 'flowers-v1.0.json'
dataset = SegmentsDataset(release, labelset='ground-truth', filter_by=['labeled', 'reviewed'])

# Export to COCO panoptic format
export_dataset(dataset, export_format='semantic')