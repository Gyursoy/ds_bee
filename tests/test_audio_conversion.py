# import pytest
import sys
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.preprocessing.audio_conversion import pcm_to_wav, pcm_to_wav_batch

def test_func():
    pass


# if __name__ == "__main__":
#     pytest.main([__file__])
