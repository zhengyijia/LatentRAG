import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class IdentityCollator:
    def __call__(self, features): 
        return features[0]
