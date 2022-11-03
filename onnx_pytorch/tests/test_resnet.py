import subprocess
import tempfile
from pathlib import Path

from onnx_pytorch import code_gen


def test_resnet():
    with tempfile.TemporaryDirectory() as tmpdirname:
        out_dir = Path(tmpdirname)
        resnet_url = 'https://github.com/onnx/models/raw/main/vision/classification/resnet/model/resnet18-v2-7.onnx'
        subprocess.run([f'wget {resnet_url} -P {out_dir}'], shell=True)

        model_fn = out_dir / Path(resnet_url).name
        assert model_fn.exists()

        code_gen.gen(model_fn, './')


if __name__ == '__main__':
    test_resnet()
