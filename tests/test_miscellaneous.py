import wget
from tiatoolbox.wsicore import wsireader


def test_conda_env(tmp_path):
    url = "https://tiatoolbox.dcs.warwick.ac.uk/sample_wsis/CMU-1-Small-Region.svs"
    path = tmp_path / "slide.svs"
    wget.download(url, out=path.as_posix())

    # assert torch.cuda.is_available()

    img = wsireader.OpenSlideWSIReader(path)
    info = img.info.as_dict()

    assert info["mpp"] == (0.499, 0.499)
    assert info["slide_dimensions"] == (2220, 2967)
