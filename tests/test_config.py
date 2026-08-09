from amber_thesis.config import get_paths, load_config


def test_local_smoke_config_loads():
    config = load_config("configs/local_smoke.yaml")
    paths = get_paths(config)

    assert config["run"]["name"] == "local_smoke"
    assert paths.results.name == "results"
