"""Segmentation model interface; DeepCrack-style multi-scale implementation will be added after data adapters."""
from .cnn_classifier import InfraCrackCNN
class InfraCrackSegmentor(InfraCrackCNN): pass
